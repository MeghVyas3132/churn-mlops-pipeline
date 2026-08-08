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
| **DVC** | Datasets and model artifacts (too large/binary for Git) | Google Drive remote; hashes in `dvc.lock` |
| **MLflow** | Experiments — parameters, metrics, model artifacts | Local SQLite store + Model Registry |

Nothing is committed to the wrong one. That separation is what makes the workflow reproducible.

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/MeghVyas3132/churn-mlops-pipeline.git
cd churn-mlops-pipeline

# 2. Environment (Python 3.14; 3.11+ works)
make setup
source .venv/bin/activate

# 3. Credentials — see "Credentials" below
#    Kaggle token at ~/.kaggle/kaggle.json, and `dvc remote modify` for Drive

# 4. Run the whole pipeline
dvc repro
```

Already have access to the DVC remote? Skip the Kaggle download entirely:

```bash
dvc pull        # fetches the exact data version recorded in dvc.lock
dvc repro       # every stage is cached, so this is a no-op
```

Then inspect the results:

```bash
make metrics                # metrics from the last run
make mlflow                 # MLflow UI at http://localhost:5000
open reports/plots/roc_curves.png
```

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

### Kaggle (needed for `dvc repro` from scratch)

Get a token from <https://www.kaggle.com/settings> → *API* → *Create New Token*, then:

```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

CI and other non-interactive environments can instead export `KAGGLE_USERNAME` and
`KAGGLE_KEY`. Not needed at all if you use `dvc pull`.

### DVC remote (Google Drive)

```bash
dvc remote add -d storage gdrive://<YOUR_FOLDER_ID>
dvc remote modify storage gdrive_acknowledge_abuse true
dvc push
```

`<YOUR_FOLDER_ID>` is the trailing path segment of the Drive folder URL
(`https://drive.google.com/drive/folders/`**`1a2b3c...`**). The first `dvc push` opens a
browser for OAuth consent.

> Since 2024 Google requires each user to supply their own OAuth client for gdrive
> remotes. If the default flow is rejected, create a client in Google Cloud Console and
> set `gdrive_client_id` / `gdrive_client_secret` — DVC's
> [gdrive documentation](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive)
> has the walkthrough. A local remote (`dvc remote add -d storage /path/to/folder`) is a
> working fallback.

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
| 3 | Track data with DVC | [dvc.yaml](dvc.yaml), `dvc.lock`, Drive remote |
| 4 | Track experiments with MLflow | [tracking.py](src/utils/tracking.py) + both model stages |
| 5 | Automated tests with Pytest | [tests/](tests/) — 95 tests |
| 6 | Git version control | Commit history, branches, PRs |
| 7 | GitHub Actions on push and PR | [ci.yml](.github/workflows/ci.yml) |
| 8 | Reproducible by another developer | This README, `params.yaml`, pinned deps, `dvc.lock` |
