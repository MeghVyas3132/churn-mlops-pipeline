"""Tests for model construction and training behaviour.

These are contract tests, not accuracy tests. They check that a model can be
built from `params.yaml`, that it fits, and that its output has the shape and
range the evaluation stage assumes -- the failures that would otherwise surface
as a confusing crash three stages downstream.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data.preprocess import build_preprocessor, to_frame
from src.models.train import MODEL_BUILDERS, build_model, cross_validate
from src.utils.config import load_params

SEED = 42


@pytest.fixture
def training_matrix(feature_target):
    """Fully preprocessed features plus the target, ready for `fit`."""
    features, target = feature_target
    preprocessor = build_preprocessor("median", "most_frequent", scale=True)
    return to_frame(preprocessor.fit_transform(features), preprocessor), target


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------

def test_every_model_in_params_has_a_builder():
    """params.yaml and MODEL_BUILDERS must not drift apart."""
    for name in load_params()["train"]["models"]:
        assert name in MODEL_BUILDERS, f"'{name}' is configured but has no builder"


def test_build_model_returns_the_right_class():
    assert isinstance(build_model("logistic_regression", {"C": 1.0}, SEED), LogisticRegression)
    forest = build_model("random_forest", {"n_estimators": 10}, SEED)
    assert isinstance(forest, RandomForestClassifier)


def test_build_model_applies_params_from_config():
    model = build_model("random_forest", {"n_estimators": 33, "max_depth": 4}, SEED)
    assert model.n_estimators == 33
    assert model.max_depth == 4


def test_build_model_injects_the_seed():
    """Without this the run is not reproducible, whatever params.yaml says."""
    assert build_model("random_forest", {"n_estimators": 10}, 123).random_state == 123


def test_build_model_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("magic_forest", {}, SEED)


@pytest.mark.parametrize("name", sorted(MODEL_BUILDERS))
def test_all_registered_models_are_constructible(name):
    assert build_model(name, {}, SEED) is not None


# --------------------------------------------------------------------------
# Fitting and prediction contracts
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(MODEL_BUILDERS))
def test_every_model_trains_and_predicts(name, training_matrix):
    x_train, y_train = training_matrix
    model = build_model(name, {}, SEED)
    model.fit(x_train, y_train)

    predictions = model.predict(x_train)
    assert len(predictions) == len(x_train)


@pytest.mark.parametrize("name", sorted(MODEL_BUILDERS))
def test_predictions_are_binary_labels(name, training_matrix):
    """Evaluation assumes a 0/1 classifier; anything else breaks the metrics."""
    x_train, y_train = training_matrix
    model = build_model(name, {}, SEED)
    model.fit(x_train, y_train)

    assert set(np.unique(model.predict(x_train))).issubset({0, 1})


@pytest.mark.parametrize("name", sorted(MODEL_BUILDERS))
def test_predicted_probabilities_are_valid(name, training_matrix):
    """ROC-AUC needs calibrated-shaped probabilities in [0, 1] summing to 1."""
    x_train, y_train = training_matrix
    model = build_model(name, {}, SEED)
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_train)
    assert proba.shape == (len(x_train), 2)
    assert (proba >= 0).all() and (proba <= 1).all()
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_training_is_reproducible_with_a_fixed_seed(training_matrix):
    """Two models, same seed, same data -> identical predictions."""
    x_train, y_train = training_matrix

    first = build_model("random_forest", {"n_estimators": 20}, SEED).fit(x_train, y_train)
    second = build_model("random_forest", {"n_estimators": 20}, SEED).fit(x_train, y_train)

    assert np.array_equal(first.predict(x_train), second.predict(x_train))


def test_model_beats_random_guessing(training_matrix):
    """A sanity floor -- catches a scrambled target or all-zero features."""
    x_train, y_train = training_matrix
    model = build_model("random_forest", {"n_estimators": 50}, SEED).fit(x_train, y_train)

    assert model.score(x_train, y_train) > 0.5


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------

def test_cross_validate_returns_the_expected_keys(training_matrix):
    x_train, y_train = training_matrix
    metrics = cross_validate(
        build_model("logistic_regression", {"max_iter": 200}, SEED), x_train, y_train, 3, SEED
    )
    assert {"cv_roc_auc_mean", "cv_roc_auc_std"} == set(metrics)


def test_cross_validated_auc_is_in_range(training_matrix):
    x_train, y_train = training_matrix
    metrics = cross_validate(
        build_model("logistic_regression", {"max_iter": 200}, SEED), x_train, y_train, 3, SEED
    )
    assert 0.0 <= metrics["cv_roc_auc_mean"] <= 1.0
    assert metrics["cv_roc_auc_std"] >= 0.0


def test_cross_validate_is_deterministic(training_matrix):
    x_train, y_train = training_matrix
    args = (x_train, y_train, 3, SEED)

    first = cross_validate(build_model("logistic_regression", {"max_iter": 200}, SEED), *args)
    second = cross_validate(build_model("logistic_regression", {"max_iter": 200}, SEED), *args)

    assert first == second
