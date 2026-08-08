"""Stage 3 -- train every candidate model and log each run to MLflow.

Model *selection* deliberately does not happen here. This stage trains each
candidate under identical conditions and records what it saw; `evaluate.py` then
scores them on the untouched test set and promotes a winner. Splitting the two
keeps the test set out of reach during training, which is the only way the final
metric means anything.

Every model is trained on the same split, with the same seed, and cross-validated
on the training data alone. Each gets its own nested MLflow run so the runs table
compares them side by side.

Run:
    python -m src.models.train
"""

from __future__ import annotations

import json
import sys
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.utils.config import ensure_dir, load_params, resolve_path, set_global_seed
from src.utils.logger import get_logger
from src.utils.tracking import log_model, setup_mlflow

LOGGER = get_logger("models.train")

# Registry of everything trainable. Adding a fourth candidate means adding a
# builder here and a block in params.yaml -- no other file changes.
MODEL_BUILDERS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}


def build_model(name: str, model_params: dict[str, Any], seed: int):
    """Instantiate an estimator with its params from `params.yaml` plus the seed."""
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Known: {sorted(MODEL_BUILDERS)}")

    return MODEL_BUILDERS[name](**{**model_params, "random_state": seed})


def load_training_data(processed_dir):
    """Read the training split produced by the preprocessing stage."""
    x_train = pd.read_csv(processed_dir / "X_train.csv")
    # squeeze() collapses the single-column frame to the Series sklearn expects.
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze("columns")
    return x_train, y_train


def cross_validate(model, x_train, y_train, folds: int, seed: int) -> dict[str, float]:
    """Score the model on the training split via stratified k-fold.

    This is the honest in-training estimate. Stratified because churn is
    imbalanced and unstratified folds can end up with wildly different positive
    rates, making the fold scores incomparable.
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(model, x_train, y_train, cv=splitter, scoring="roc_auc")

    return {
        "cv_roc_auc_mean": float(scores.mean()),
        "cv_roc_auc_std": float(scores.std()),
    }


def main() -> int:
    params = load_params()
    prep_cfg = params["preprocess"]
    train_cfg = params["train"]
    mlflow_cfg = params["mlflow"]
    seed = params["seed"]

    set_global_seed(seed)

    processed_dir = resolve_path(prep_cfg["processed_dir"])
    if not (processed_dir / "X_train.csv").exists():
        LOGGER.error(
            "Processed data not found in %s. Run `python -m src.data.preprocess` first.",
            processed_dir,
        )
        return 1

    models_dir = ensure_dir(train_cfg["models_dir"])

    # A local store keeps the whole experiment history inside the repo working
    # tree -- no server to stand up before a reviewer can look at the runs.
    setup_mlflow(mlflow_cfg)

    x_train, y_train = load_training_data(processed_dir)
    LOGGER.info("Training data: %d rows x %d features", len(x_train), x_train.shape[1])

    enabled = {
        name: cfg for name, cfg in train_cfg["models"].items() if cfg.get("enabled", True)
    }
    if not enabled:
        LOGGER.error("No models enabled in params.yaml -- nothing to train.")
        return 1

    LOGGER.info("Training %d model(s): %s", len(enabled), ", ".join(enabled))

    summary: dict[str, Any] = {}

    with mlflow.start_run(run_name="training-session") as parent_run:
        mlflow.log_params(
            {
                "seed": seed,
                "n_train_rows": len(x_train),
                "n_features": x_train.shape[1],
                "cv_folds": train_cfg["cv_folds"],
                "test_size": prep_cfg["test_size"],
            }
        )
        mlflow.set_tag("stage", "training")

        for name, model_cfg in enabled.items():
            LOGGER.info("--- %s ---", name)

            with mlflow.start_run(run_name=name, nested=True):
                model = build_model(name, model_cfg.get("params", {}), seed)

                mlflow.log_param("model_type", name)
                mlflow.log_params(
                    {f"{name}__{k}": v for k, v in model_cfg.get("params", {}).items()}
                )

                cv_metrics = cross_validate(
                    model, x_train, y_train, train_cfg["cv_folds"], seed
                )
                LOGGER.info(
                    "CV ROC-AUC: %.4f (+/- %.4f)",
                    cv_metrics["cv_roc_auc_mean"],
                    cv_metrics["cv_roc_auc_std"],
                )

                model.fit(x_train, y_train)

                mlflow.log_metrics(cv_metrics)
                # float64 on purpose: one-hot columns round-trip through CSV as
                # int64, which would bake an integer input signature into the
                # model and make MLflow reject float input at serving time.
                log_model(model, name=name, input_example=x_train.head(5).astype("float64"))

                artifact_path = models_dir / f"{name}.joblib"
                joblib.dump(model, artifact_path)
                LOGGER.info("Saved %s", artifact_path)

                summary[name] = {
                    **cv_metrics,
                    "artifact": str(artifact_path.relative_to(resolve_path("."))),
                    "params": model_cfg.get("params", {}),
                }

        mlflow.set_tag("models_trained", ",".join(summary))
        LOGGER.info("Parent MLflow run: %s", parent_run.info.run_id)

    with (models_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    best = max(summary.items(), key=lambda item: item[1]["cv_roc_auc_mean"])
    LOGGER.info("Best by CV ROC-AUC: %s (%.4f)", best[0], best[1]["cv_roc_auc_mean"])
    LOGGER.info("Training complete. Run `mlflow ui` to inspect the runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
