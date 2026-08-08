"""Tests for metric computation and the reporting artifacts.

Metrics drive model promotion, so a silent bug here would ship the wrong model
without anything looking obviously broken. Each metric is checked against a
hand-worked example where the correct answer is known by inspection.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curves
from src.utils.config import load_params


# --------------------------------------------------------------------------
# Metric correctness
# --------------------------------------------------------------------------

def test_perfect_predictions_score_one():
    y_true = np.array([0, 1, 0, 1, 1])
    metrics = compute_metrics(y_true, y_true, np.array([0.1, 0.9, 0.2, 0.8, 0.95]))

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_metrics_match_a_hand_worked_example():
    """TP=2, FP=1, FN=1, TN=2 -> precision 2/3, recall 2/3, accuracy 4/6."""
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.4, 0.6, 0.3, 0.2])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)


def test_all_metrics_are_present():
    y_true = np.array([0, 1, 0, 1])
    metrics = compute_metrics(y_true, y_true, np.array([0.1, 0.9, 0.2, 0.8]))
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"} == set(metrics)


@pytest.mark.parametrize("metric", ["accuracy", "precision", "recall", "f1", "roc_auc"])
def test_metrics_stay_within_zero_and_one(metric):
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, 200)
    y_proba = rng.random(200)

    assert 0.0 <= compute_metrics(y_true, (y_proba > 0.5).astype(int), y_proba)[metric] <= 1.0


def test_metrics_are_json_serialisable_floats():
    """numpy scalars break json.dump -- compute_metrics must cast to float."""
    y_true = np.array([0, 1, 0, 1])
    for value in compute_metrics(y_true, y_true, np.array([0.1, 0.9, 0.2, 0.8])).values():
        assert type(value) is float


def test_a_degenerate_all_negative_prediction_does_not_crash():
    """The classic imbalanced-data failure: high accuracy, zero recall.

    zero_division=0 must keep this from raising so the run still reports.
    """
    y_true = np.array([0, 0, 0, 0, 1])
    y_pred = np.zeros(5, dtype=int)

    metrics = compute_metrics(y_true, y_pred, np.array([0.1, 0.1, 0.2, 0.1, 0.3]))

    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["recall"] == pytest.approx(0.0)
    assert metrics["precision"] == pytest.approx(0.0)


def test_inverted_probabilities_score_below_chance():
    """Confirms roc_auc is fed the positive-class column, not the negative one."""
    y_true = np.array([0, 0, 1, 1])
    inverted = np.array([0.9, 0.8, 0.2, 0.1])
    assert compute_metrics(y_true, y_true, inverted)["roc_auc"] < 0.5


# --------------------------------------------------------------------------
# Selection logic and gating
# --------------------------------------------------------------------------

def test_best_model_selection_picks_the_highest_primary_metric():
    """Mirrors the `max(...)` used in evaluate.main()."""
    results = {
        "logistic_regression": {"roc_auc": 0.81, "f1": 0.90},
        "random_forest": {"roc_auc": 0.93, "f1": 0.70},
        "xgboost": {"roc_auc": 0.88, "f1": 0.85},
    }
    assert max(results, key=lambda name: results[name]["roc_auc"]) == "random_forest"


def test_quality_gate_threshold_is_a_valid_probability():
    threshold = load_params()["evaluate"]["min_roc_auc"]
    assert 0.5 <= threshold <= 1.0, "a gate below 0.5 would accept worse-than-random models"


def test_promoted_model_lives_outside_the_candidates_directory():
    """The train and evaluate DVC stages must own disjoint outputs.

    If the promoted model were written inside `train.models_dir`, both stages
    would claim the same path and `dvc repro` would refuse to build the DAG.
    """
    params = load_params()
    candidates_dir = params["train"]["models_dir"].rstrip("/")
    promoted = params["evaluate"]["promoted_model_path"]

    assert not promoted.startswith(f"{candidates_dir}/")


# --------------------------------------------------------------------------
# Plot artifacts
# --------------------------------------------------------------------------

def test_confusion_matrix_plot_is_written(tmp_path):
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    output = tmp_path / "cm.png"

    plot_confusion_matrix(y_true, y_pred, "test_model", output)

    assert output.exists() and output.stat().st_size > 0


def test_roc_curve_plot_is_written(tmp_path):
    output = tmp_path / "roc.png"
    curves = {
        "model_a": (np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.8, 1.0]), 0.85),
        "model_b": (np.array([0.0, 0.3, 1.0]), np.array([0.0, 0.9, 1.0]), 0.91),
    }

    plot_roc_curves(curves, output)

    assert output.exists() and output.stat().st_size > 0
