# Customer Churn Prediction — MLOps Pipeline

An end-to-end, reproducible machine learning pipeline that predicts customer churn from
demographics, account information, service usage and billing details.

The point of this repository is not the model. It is the **engineering around** the model:
any developer can clone it, run four commands, and land on byte-identical data splits,
identical experiment records and the same promoted artifact.

> **Course:** MLOps (STDE 301) · Vijaybhoomi School of Science and Technology · Mid Term, August 2026

---

## The core idea

Three different things get versioned, by three different tools:

| Tool | Versions | Lives in |
|------|----------|----------|
| **Git** | Code, configuration, pipeline definition | GitHub |
| **DVC** | Datasets and model artifacts (too large/binary for Git) | DagsHub remote; hashes in `dvc.lock` |
| **MLflow** | Experiments — parameters, metrics, model artifacts | Local SQLite store + Model Registry |

Nothing is committed to the wrong one. That separation is what makes the workflow reproducible.

---

## Quick start

Common to both routes below — **Python 3.12 or newer is required** (numpy and xgboost set
that floor; 3.14 is what CI runs):

```bash
git clone https://github.com/MeghVyas3132/churn-mlops-pipeline.git
cd churn-mlops-pipeline
make setup
source .venv/bin/activate
```

### Route A — pull the exact data (no Kaggle account needed)

Recommended if you have been granted access to the DagsHub remote. Credentials must be set
**before** `dvc pull`, or it fails with `HTTP 'basic' authentication require both 'user'
and 'password'`.

```bash
dvc remote modify origin --local user     <YOUR_DAGSHUB_USERNAME>
dvc remote modify origin --local password <YOUR_DAGSHUB_TOKEN>

dvc pull      # ~50 MB: raw CSVs, processed splits, all model artifacts
dvc repro     # every stage is already cached, so this reports "up to date"
```

### Route B — rebuild everything from source

No DVC remote access needed, but you need a Kaggle token at `~/.kaggle/access_token`
(see [Credentials](#credentials)). Takes a few minutes.

```bash
dvc repro     # downloads from Kaggle, then runs all four stages
```

### Then inspect the results

```bash
dvc metrics show                     # metrics from the last run
make mlflow                          # MLflow UI at http://localhost:5000
```

Plots land in `reports/plots/` (`roc_curves.png`, `confusion_matrix_*.png`).

> Both routes were verified from a clean `git clone` into an empty directory. Route A
> reproduced `reports/metrics.json` byte-identically and reported all four stages cached.

---

## Pipeline

```
ingest ──▶ preprocess ──▶ train ──▶ evaluate
```

`dvc repro` runs only the stages whose inputs changed. Edit a hyperparameter under
`train:` in `params.yaml` and DVC re-runs **train + evaluate** while reusing the cached
ingest and preprocess outputs.

| Stage | Script | Does | Produces |
|-------|--------|------|----------|
| **ingest** | [ingest.py](src/data/ingest.py) | Downloads from Kaggle, validates the schema against a 12-column contract, reports nulls and churn rate | `data/raw/` |
| **preprocess** | [preprocess.py](src/data/preprocess.py) | Drops unlabelled rows, deduplicates, imputes, one-hot encodes, scales, stratified split | `data/processed/` + `preprocessor.joblib` |
| **train** | [train.py](src/models/train.py) | Trains 3 candidates, 3-fold stratified CV, logs every run to MLflow | `models/candidates/` |
| **evaluate** | [evaluate.py](src/models/evaluate.py) | Scores on the held-out test set, applies a quality gate, promotes the winner | `models/best_model.joblib`, `reports/` |

### Models compared

Logistic Regression (interpretable baseline), Random Forest (non-linear, robust) and
XGBoost (gradient boosting). All three train on the same split with the same seed, and
selection is driven by **ROC-AUC** — not accuracy, which flatters a model that simply
predicts the majority class on imbalanced churn data.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.8962 | 0.9395 | 0.8738 | 0.9054 | 0.9596 |
| Random Forest | 0.9921 | 0.9999 | 0.9862 | 0.9930 | 0.9999 |
| **XGBoost** (promoted) | **0.9999** | **1.0000** | **0.9997** | **0.9999** | **1.0000** |

### Why the scores are near-perfect — and why that is not a bug

A ROC-AUC of 1.000 normally means label leakage. Here it does not. **This Kaggle dataset is
synthetically generated and contains a hard deterministic rule.** Checking the raw file:

```
Support Calls    churn rate    rows
      5            0.947       24,918
      6            1.000       23,554
      7            1.000       23,870
      8            1.000       23,559
      9            1.000       23,630
     10            1.000       23,900
```

Every one of the ~118,000 customers with 6 or more support calls churned, without a single
exception. That is a rule the data generator applied, not a pattern learned from behaviour.

Two pieces of evidence that our pipeline is not leaking:

1. **Logistic Regression only reaches 0.96.** A leak would lift *every* model to ~1.0. A
   linear model cannot express a sharp step at `Support Calls >= 6`, so it lags — precisely
   the signature of a threshold rule rather than a leaked label.
2. **The leakage tests pass.** `test_transformer_is_fitted_on_training_data_only` and
   `test_split_produces_disjoint_rows` assert the two failure modes directly.

The practical reading: these metrics say the pipeline is wired correctly, not that churn
prediction is solved. On real customer data expect ROC-AUC in the 0.75–0.85 range, and the
`min_roc_auc` gate in `params.yaml` is set to a realistic 0.70 rather than to these numbers.

---

## Design decisions worth knowing

**One config file, no CLI arguments.** Every tunable lives in [params.yaml](params.yaml).
Scripts take no arguments. That is what lets DVC hash *just* the parameters a stage
actually reads and skip stages precisely.

**Preprocessing is fitted on training data only.** Medians, modes, category vocabularies
and scaler statistics are learned from the training split, then applied to test. Fitting on
the full frame first would leak test information and inflate every reported metric.
`test_transformer_is_fitted_on_training_data_only` asserts this directly.

**Train and evaluate are separate stages.** The test set is untouched until evaluation.
Model selection therefore cannot see it.

**A quality gate blocks promotion.** If the best model scores below `evaluate.min_roc_auc`,
the stage exits non-zero — `dvc repro` fails and no model is promoted. Bad models fail
loudly rather than shipping quietly.

**Metrics live in Git, data lives in DVC.** `reports/metrics.json` is declared
`cache: false`, so it is committed as text. That is what makes
`dvc metrics diff main` able to compare two commits.

**Tests use synthetic fixtures, never the real dataset.** CI has no DVC credentials, so a
suite that skipped when data was absent would give a green tick proving nothing. The
fixtures in [conftest.py](tests/conftest.py) generate schema-accurate data — including the
malformed all-NaN row the published CSV actually contains.

---

## Reproducibility

| Mechanism | Where |
|-----------|-------|
| Single global seed threaded through split, models, numpy and stdlib `random` | `params.yaml` → `set_global_seed()` |
| Exact dependency pins | [requirements.txt](requirements.txt) |
| Data version hashes | `dvc.lock` (committed) |
| Stage dependency graph | [dvc.yaml](dvc.yaml) |
| Full experiment history | `mlflow.db` + Model Registry |

Verify it yourself — run twice and diff:

```bash
dvc repro --force && cp reports/metrics.json /tmp/run1.json
dvc repro --force && diff /tmp/run1.json reports/metrics.json && echo "identical"
```

---

## Testing

```bash
make test          # or: pytest --cov=src
```

95 tests across five modules, covering configuration contracts, schema validation, the
no-leakage guarantee, split integrity, model construction, prediction contracts and metric
correctness. Highlights:

- `test_preprocessor_eliminates_all_nulls` — nothing null reaches an estimator
- `test_transformer_is_fitted_on_training_data_only` — the anti-leakage assertion
- `test_split_produces_disjoint_rows` — no customer in both train and test
- `test_split_is_reproducible_with_a_fixed_seed` — same seed, same split
- `test_metrics_match_a_hand_worked_example` — metrics checked against known answers
- `test_a_degenerate_all_negative_prediction_does_not_crash` — the classic imbalanced-data trap

---

## Continuous integration

[.github/workflows/ci.yml](.github/workflows/ci.yml) runs on **every push and every pull
request** to `main`:

1. **Lint** — flake8, with syntax errors and undefined names as a hard gate
2. **Tests** — full pytest suite with coverage, uploaded as an artifact
3. **Pipeline check** — `dvc dag` proves `dvc.yaml` parses and the DAG resolves; the build
   fails if `dvc.lock` is missing, since without it nobody can reproduce the data version

---

## Credentials

No secret is ever committed. Everything below writes either to your home directory or to
`.dvc/config.local`, both of which are outside Git's reach.

### Kaggle (needed only for `dvc repro` from scratch)

Get a token from <https://www.kaggle.com/settings> → *API* → *Create New Token*, then:

```bash
mkdir -p ~/.kaggle && printf '%s' "KGAT_your_token_here" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

The `chmod` is not optional — the client refuses to run on a world-readable token file.

Kaggle issues an opaque `KGAT_…` token; the older `kaggle.json` username/key pair still
works and is also detected. Non-interactive environments can export `KAGGLE_API_TOKEN`
instead. None of this is needed if you use `dvc pull`.

### DVC remote and MLflow server (DagsHub)

[DagsHub](https://dagshub.com) hosts both a DVC remote and an MLflow tracking server for a
repository, so data, models and experiment history all live in one place a reviewer can
open in a browser.

1. Sign in at <https://dagshub.com> with GitHub
2. **+** → **Connect a Repository** → **GitHub** → select this repo
3. Generate a token at <https://dagshub.com/user/settings/tokens>

The remote URL and `auth = basic` are already in `.dvc/config`, so only the credentials are
left to supply — and those go in the local, gitignored config:

```bash
dvc remote modify origin --local user     <YOUR_DAGSHUB_USERNAME>
dvc remote modify origin --local password <YOUR_DAGSHUB_TOKEN>
dvc push
```

`--local` writes to `.dvc/config.local`, which is gitignored. Never drop a token into
`.dvc/config` — that file *is* committed.

To send experiment runs to the hosted MLflow server instead of the local SQLite file:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/MeghVyas3132/churn-mlops-pipeline.mlflow
export MLFLOW_TRACKING_USERNAME=<YOUR_DAGSHUB_USERNAME>
export MLFLOW_TRACKING_PASSWORD=<YOUR_DAGSHUB_TOKEN>
dvc repro --force
```

`MLFLOW_TRACKING_URI` overrides `params.yaml`, so the same commit logs locally on a laptop
and to the server in CI without a URL or credential ever entering Git. Unset the variables
and everything falls back to `sqlite:///mlflow.db`.

> **Prefer no accounts at all?** A local directory works as a drop-in replacement and still
> demonstrates the full `dvc push` / `dvc pull` cycle:
> `dvc remote add -d --force storage ~/dvc-storage/churn-mlops`

---

## Layout

```
churn-mlops-pipeline/
├── .github/workflows/ci.yml     # CI: lint, test, validate pipeline
├── data/                        # DVC-tracked, gitignored
│   ├── raw/                     #   downloaded CSVs
│   └── processed/               #   splits + fitted preprocessor
├── models/
│   ├── candidates/              #   every trained model
│   └── best_model.joblib        #   the promoted one
├── reports/
│   ├── metrics.json             #   DVC metrics (committed to Git)
│   ├── model_comparison.csv     #   side-by-side table
│   └── plots/                   #   ROC curves, confusion matrices
├── src/
│   ├── data/{ingest,preprocess}.py
│   ├── models/{train,evaluate}.py
│   └── utils/{config,logger,tracking}.py
├── tests/                       # 95 tests, synthetic fixtures
├── params.yaml                  # every tunable, one file
├── dvc.yaml / dvc.lock          # pipeline DAG + locked data versions
├── requirements.txt             # exact pins
└── Makefile                     # command shortcuts
```

---

## Common commands

| Command | Does |
|---------|------|
| `dvc repro` | Run stale stages only |
| `dvc repro --force` | Re-run everything |
| `dvc dag` | Print the stage graph |
| `dvc metrics show` | Metrics from the last run |
| `dvc metrics diff main` | Compare metrics against another commit |
| `dvc push` / `dvc pull` | Sync data and models with the remote |
| `dvc status` | What is out of date |
| `make mlflow` | MLflow UI |
| `make test` | Tests with coverage |

---

## Requirement coverage

| # | Requirement | Where |
|---|-------------|-------|
| 1 | Ingest and preprocess | [src/data/](src/data/) |
| 2 | Train and evaluate multiple models | [train.py](src/models/train.py), [evaluate.py](src/models/evaluate.py) |
| 3 | Track data with DVC | [dvc.yaml](dvc.yaml), `dvc.lock`, DagsHub remote |
| 4 | Track experiments with MLflow | [tracking.py](src/utils/tracking.py) + both model stages |
| 5 | Automated tests with Pytest | [tests/](tests/) — 95 tests |
| 6 | Git version control | Commit history, branches, PRs |
| 7 | GitHub Actions on push and PR | [ci.yml](.github/workflows/ci.yml) |
| 8 | Reproducible by another developer | This README, `params.yaml`, pinned deps, `dvc.lock` |
