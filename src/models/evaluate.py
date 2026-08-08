"""Stage 4 -- score every trained model on the held-out test set and promote one.

This is the first and only time the test split is touched. Each candidate is
scored on identical data, the winner is chosen by the metric named in
`params.yaml`, and a quality gate rejects the promotion outright if even the best
model is too weak to ship.

Outputs `reports/metrics.json`, which is registered as a DVC metrics file -- so
`dvc metrics diff` shows exactly how a code or parameter change moved the numbers.

Run:
    python -m src.models.evaluate
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

# Headless backend: this runs on a CI runner with no display server.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.config import ensure_dir, load_params, resolve_path, set_global_seed  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.tracking import log_model, setup_mlflow  # noqa: E402

LOGGER = get_logger("models.evaluate")


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Five metrics, because accuracy alone hides the failure that matters.

    A model that predicts "nobody churns" can post high accuracy on skewed data
    while being useless to a retention team. Recall says how many at-risk
    customers were actually caught; ROC-AUC is threshold-independent and drives
    model selection.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, output_path: Path) -> None:
    """Save a labelled confusion matrix.

    The off-diagonal cells are the business story: false negatives are churners
    the retention team never got warned about.
    """
    matrix = confusion_matrix(y_true, y_pred)

    fig, axis = plt.subplots(figsize=(5, 4.5))
    axis.imshow(matrix, cmap="Blues")
    axis.set_title(f"Confusion Matrix -- {model_name}")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks([0, 1], ["Retained", "Churned"])
    axis.set_yticks([0, 1], ["Retained", "Churned"])

    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(
                col,
                row,
                f"{matrix[row, col]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_roc_curves(curves: dict[str, tuple], output_path: Path) -> None:
    """Overlay every candidate's ROC curve on one axis for direct comparison."""
    fig, axis = plt.subplots(figsize=(6, 5))

    for name, (fpr, tpr, auc) in curves.items():
        axis.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curves -- Test Set")
    axis.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def main() -> int:
    params = load_params()
    prep_cfg = params["preprocess"]
    train_cfg = params["train"]
    eval_cfg = params["evaluate"]
    mlflow_cfg = params["mlflow"]
    seed = params["seed"]

    set_global_seed(seed)

    processed_dir = resolve_path(prep_cfg["processed_dir"])
    models_dir = resolve_path(train_cfg["models_dir"])
    reports_dir = ensure_dir(eval_cfg["reports_dir"])
    plots_dir = ensure_dir(Path(eval_cfg["reports_dir"]) / "plots")

    summary_path = models_dir / "train_summary.json"
    if not summary_path.exists():
        LOGGER.error("%s not found. Run `python -m src.models.train` first.", summary_path)
        return 1

    with summary_path.open("r", encoding="utf-8") as handle:
        train_summary = json.load(handle)

    x_test = pd.read_csv(processed_dir / "X_test.csv")
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze("columns")
    LOGGER.info("Test set: %d rows x %d features", len(x_test), x_test.shape[1])

    setup_mlflow(mlflow_cfg)

    primary_metric = train_cfg["primary_metric"]
    results: dict[str, dict[str, float]] = {}
    roc_curves: dict[str, tuple] = {}

    with mlflow.start_run(run_name="evaluation") as run:
        mlflow.set_tag("stage", "evaluation")

        for name in train_summary:
            model_path = models_dir / f"{name}.joblib"
            if not model_path.exists():
                LOGGER.warning("Skipping %s -- artifact missing at %s", name, model_path)
                continue

            model = joblib.load(model_path)
            y_pred = model.predict(x_test)
            y_proba = model.predict_proba(x_test)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_proba)
            results[name] = metrics

            LOGGER.info(
                "%-20s acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f",
                name,
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["roc_auc"],
            )

            mlflow.log_metrics({f"test_{name}_{k}": v for k, v in metrics.items()})

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_curves[name] = (fpr, tpr, metrics["roc_auc"])

            plot_confusion_matrix(y_test, y_pred, name, plots_dir / f"confusion_matrix_{name}.png")

        if not results:
            LOGGER.error("No models could be evaluated -- no artifacts found in %s.", models_dir)
            return 1

        plot_roc_curves(roc_curves, plots_dir / "roc_curves.png")

        # ---- Model selection ----
        best_name = max(results, key=lambda name: results[name][primary_metric])
        best_metrics = results[best_name]
        LOGGER.info(
            "Best model by %s: %s (%.4f)", primary_metric, best_name, best_metrics[primary_metric]
        )

        # ---- Quality gate ----
        # Exits non-zero so `dvc repro` and CI both fail rather than silently
        # promoting a model that is worse than the agreed floor.
        threshold = eval_cfg["min_roc_auc"]
        if best_metrics["roc_auc"] < threshold:
            LOGGER.error(
                "QUALITY GATE FAILED: best ROC-AUC %.4f is below the required %.4f. "
                "No model promoted.",
                best_metrics["roc_auc"],
                threshold,
            )
            return 1

        LOGGER.info(
            "Quality gate passed (ROC-AUC %.4f >= %.4f)", best_metrics["roc_auc"], threshold
        )

        promoted_path = resolve_path(eval_cfg["promoted_model_path"])
        promoted_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(models_dir / f"{best_name}.joblib", promoted_path)
        LOGGER.info("Promoted %s -> %s", best_name, promoted_path)

        mlflow.log_params({"best_model": best_name, "primary_metric": primary_metric})
        mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})
        mlflow.log_artifacts(str(plots_dir), artifact_path="plots")
        mlflow.set_tag("best_model", best_name)

        # Register the promoted model so the Model Registry carries a versioned
        # history of what was considered production-ready, and when.
        log_model(
            joblib.load(promoted_path),
            name="best_model",
            input_example=x_test.head(5).astype("float64"),
            registered_model_name=mlflow_cfg["registered_model_name"],
        )
        LOGGER.info(
            "Registered '%s' in the MLflow Model Registry", mlflow_cfg["registered_model_name"]
        )

        LOGGER.info("Evaluation MLflow run: %s", run.info.run_id)

    # ---- Persist reports ----
    # Flat top-level keys are what `dvc metrics show` renders as a table.
    metrics_payload: dict[str, Any] = {
        "best_model": best_name,
        **{f"best_{k}": v for k, v in best_metrics.items()},
        "per_model": results,
    }
    with (reports_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    comparison = pd.DataFrame(results).T.sort_values(primary_metric, ascending=False)
    comparison.index.name = "model"
    comparison.to_csv(reports_dir / "model_comparison.csv")

    LOGGER.info("\n%s", comparison.to_string())
    LOGGER.info("Reports written to %s", reports_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
