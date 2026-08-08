"""Stage 2 -- clean, encode and split the raw data.

The two rules this module exists to enforce:

1. **No leakage.** Every statistic used to transform the data (medians, modes,
   category vocabularies, scaler means) is learned from the training split only
   and then applied to the test split. Fitting on the full frame first would let
   test-set information bleed into training and inflate the reported metrics.

2. **No hidden state.** The fitted transformer is written to disk alongside the
   splits, so scoring a new customer later uses byte-identical preprocessing.

Run:
    python -m src.data.preprocess
"""

from __future__ import annotations

import json
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import ensure_dir, load_params, resolve_path, set_global_seed
from src.utils.logger import get_logger

LOGGER = get_logger("data.preprocess")

NUMERIC_FEATURES = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Subscription Type",
    "Contract Length",
]


def drop_unusable_rows(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Remove rows with no target label.

    The published training CSV contains a malformed record that pandas parses as
    all-NaN. A row without a label teaches a supervised model nothing and would
    crash `fit`, so it is dropped before anything else happens.
    """
    before = len(frame)
    cleaned = frame.dropna(subset=[target_column]).copy()
    dropped = before - len(cleaned)

    if dropped:
        LOGGER.warning("Dropped %d row(s) with a missing '%s' value", dropped, target_column)

    return cleaned


def deduplicate(frame: pd.DataFrame, id_column: str) -> pd.DataFrame:
    """Drop duplicate customer records, keeping the first occurrence.

    A customer appearing in both splits would leak directly, and duplicates also
    silently over-weight those customers during training.
    """
    if id_column not in frame.columns:
        return frame

    before = len(frame)
    deduped = frame.drop_duplicates(subset=[id_column], keep="first").copy()
    removed = before - len(deduped)

    if removed:
        LOGGER.warning("Removed %d duplicate '%s' row(s)", removed, id_column)

    return deduped


def build_preprocessor(
    numeric_strategy: str, categorical_strategy: str, scale: bool
) -> ColumnTransformer:
    """Assemble the impute -> encode/scale transformer.

    Returned unfitted: the caller fits it on train only. `handle_unknown="ignore"`
    means a category seen at scoring time but not during training produces an
    all-zero block instead of raising -- the right behaviour for a model that has
    to keep serving in production.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy=numeric_strategy))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps = [
        ("imputer", SimpleImputer(strategy=categorical_strategy)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("categorical", Pipeline(categorical_steps), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def to_frame(matrix, preprocessor: ColumnTransformer) -> pd.DataFrame:
    """Wrap the transformed matrix back into a named DataFrame.

    Keeping feature names attached is what makes the processed CSVs readable and
    lets feature-importance output later be interpreted without a lookup table.
    """
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    return pd.DataFrame(matrix, columns=names)


def main() -> int:
    params = load_params()
    ingest_cfg = params["ingest"]
    prep_cfg = params["preprocess"]
    seed = params["seed"]

    set_global_seed(seed)

    raw_path = resolve_path(ingest_cfg["raw_dir"]) / ingest_cfg["train_file"]
    if not raw_path.exists():
        LOGGER.error("Raw file not found: %s. Run `python -m src.data.ingest` first.", raw_path)
        return 1

    processed_dir = ensure_dir(prep_cfg["processed_dir"])
    target_column = ingest_cfg["target_column"]
    id_column = ingest_cfg["id_column"]

    LOGGER.info("Reading %s", raw_path.name)
    frame = pd.read_csv(raw_path)
    LOGGER.info("Loaded %d rows x %d columns", len(frame), frame.shape[1])

    if prep_cfg.get("drop_rows_missing_target", True):
        frame = drop_unusable_rows(frame, target_column)

    frame = deduplicate(frame, id_column)

    # Sub-sample for tractable CI runtimes. Seeded, so the sample is the same on
    # every machine and every rerun.
    sample_size = prep_cfg.get("sample_size")
    if sample_size and len(frame) > sample_size:
        LOGGER.info("Sampling %d of %d rows (seed=%d)", sample_size, len(frame), seed)
        frame = frame.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    features = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = frame[target_column].astype(int)

    LOGGER.info("Class balance before split: %s", target.value_counts().to_dict())

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        features,
        target,
        test_size=prep_cfg["test_size"],
        random_state=seed,
        stratify=target if prep_cfg.get("stratify", True) else None,
    )
    LOGGER.info("Split -> train=%d rows, test=%d rows", len(x_train_raw), len(x_test_raw))

    preprocessor = build_preprocessor(
        numeric_strategy=prep_cfg["numeric_impute_strategy"],
        categorical_strategy=prep_cfg["categorical_impute_strategy"],
        scale=prep_cfg.get("scale_numeric", True),
    )

    # fit_transform on train, transform on test. This asymmetry is the whole
    # point -- see the module docstring.
    x_train = to_frame(preprocessor.fit_transform(x_train_raw), preprocessor)
    x_test = to_frame(preprocessor.transform(x_test_raw), preprocessor)

    LOGGER.info("Engineered %d features", x_train.shape[1])

    x_train.to_csv(processed_dir / "X_train.csv", index=False)
    x_test.to_csv(processed_dir / "X_test.csv", index=False)
    y_train.to_csv(processed_dir / "y_train.csv", index=False)
    y_test.to_csv(processed_dir / "y_test.csv", index=False)

    joblib.dump(preprocessor, processed_dir / "preprocessor.joblib")

    metadata = {
        "n_train_rows": int(len(x_train)),
        "n_test_rows": int(len(x_test)),
        "n_features": int(x_train.shape[1]),
        "feature_names": list(x_train.columns),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "seed": seed,
        "test_size": prep_cfg["test_size"],
    }
    with (processed_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    LOGGER.info(
        "Churn rate -- train: %.4f, test: %.4f (stratification check)",
        metadata["train_churn_rate"],
        metadata["test_churn_rate"],
    )
    LOGGER.info("Preprocessing complete. Artifacts written to %s", processed_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
