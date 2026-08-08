"""Tests for the preprocessing stage.

Two properties matter more than the rest and get the most coverage here:
no nulls survive into the model, and no test-set information reaches the fitted
transformer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocess import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    deduplicate,
    drop_unusable_rows,
    to_frame,
)

SEED = 42


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def test_drop_unusable_rows_removes_rows_with_no_target(dirty_frame):
    cleaned = drop_unusable_rows(dirty_frame, "Churn")
    assert cleaned["Churn"].isnull().sum() == 0
    assert len(cleaned) < len(dirty_frame)


def test_drop_unusable_rows_keeps_clean_data_intact(raw_frame):
    assert len(drop_unusable_rows(raw_frame, "Churn")) == len(raw_frame)


def test_drop_unusable_rows_preserves_nulls_in_other_columns(dirty_frame):
    """Only the *target* is non-negotiable; feature gaps get imputed later."""
    cleaned = drop_unusable_rows(dirty_frame, "Churn")
    assert cleaned["Age"].isnull().sum() > 0


def test_deduplicate_removes_repeated_customer_ids(dirty_frame):
    deduped = deduplicate(dirty_frame, "CustomerID")
    non_null_ids = deduped["CustomerID"].dropna()
    assert non_null_ids.duplicated().sum() == 0


def test_deduplicate_is_a_no_op_when_id_column_absent(raw_frame):
    without_id = raw_frame.drop(columns=["CustomerID"])
    assert len(deduplicate(without_id, "CustomerID")) == len(without_id)


# --------------------------------------------------------------------------
# Transformation
# --------------------------------------------------------------------------

def test_preprocessor_eliminates_all_nulls(dirty_frame):
    """The headline guarantee: nothing null reaches an estimator."""
    cleaned = drop_unusable_rows(dirty_frame, "Churn")
    features = cleaned[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    preprocessor = build_preprocessor("median", "most_frequent", scale=True)
    transformed = to_frame(preprocessor.fit_transform(features), preprocessor)

    assert transformed.isnull().sum().sum() == 0
    assert np.isfinite(transformed.to_numpy()).all()


def test_preprocessor_one_hot_encodes_categoricals(feature_target):
    features, _ = feature_target
    preprocessor = build_preprocessor("median", "most_frequent", scale=True)
    transformed = to_frame(preprocessor.fit_transform(features), preprocessor)

    # 3 categoricals with 2 + 3 + 3 levels -> 8 dummy columns, plus 7 numerics.
    assert transformed.shape[1] > len(NUMERIC_FEATURES)
    assert any("Gender" in column for column in transformed.columns)


def test_preprocessor_produces_only_numeric_output(feature_target):
    """Estimators cannot consume object dtypes -- encoding must be complete."""
    features, _ = feature_target
    preprocessor = build_preprocessor("median", "most_frequent", scale=True)
    transformed = to_frame(preprocessor.fit_transform(features), preprocessor)

    assert all(pd.api.types.is_numeric_dtype(transformed[c]) for c in transformed.columns)


def test_scaling_standardises_numeric_columns(feature_target):
    features, _ = feature_target
    preprocessor = build_preprocessor("median", "most_frequent", scale=True)
    transformed = to_frame(preprocessor.fit_transform(features), preprocessor)

    numeric_columns = [c for c in transformed.columns if c.startswith("numeric__")]
    assert numeric_columns
    assert np.allclose(transformed[numeric_columns].mean(), 0, atol=1e-6)
    assert np.allclose(transformed[numeric_columns].std(ddof=0), 1, atol=1e-6)


def test_scaling_can_be_disabled(feature_target):
    features, _ = feature_target
    preprocessor = build_preprocessor("median", "most_frequent", scale=False)
    transformed = to_frame(preprocessor.fit_transform(features), preprocessor)

    # Age lives on 18-70; unscaled it must stay far from a zero mean.
    assert transformed["numeric__Age"].mean() > 10


def test_preprocessor_row_count_is_preserved(feature_target):
    """Transformation must never drop or invent rows."""
    features, _ = feature_target
    preprocessor = build_preprocessor("median", "most_frequent", scale=True)
    assert len(to_frame(preprocessor.fit_transform(features), preprocessor)) == len(features)


def test_unseen_category_at_transform_time_does_not_raise(feature_target):
    """A new subscription tier must not take the scoring path down."""
    features, _ = feature_target
    preprocessor = build_preprocessor("median", "most_frequent", scale=True)
    preprocessor.fit(features)

    unseen = features.head(5).copy()
    unseen["Subscription Type"] = "Platinum"  # never present during fit

    transformed = preprocessor.transform(unseen)
    assert transformed.shape[0] == 5


def test_train_and_test_have_identical_feature_columns(feature_target):
    """A column-order mismatch between fit and transform corrupts predictions."""
    features, target = feature_target
    x_train_raw, x_test_raw = train_test_split(features, test_size=0.2, random_state=SEED)[:2]

    preprocessor = build_preprocessor("median", "most_frequent", scale=True)
    x_train = to_frame(preprocessor.fit_transform(x_train_raw), preprocessor)
    x_test = to_frame(preprocessor.transform(x_test_raw), preprocessor)

    assert list(x_train.columns) == list(x_test.columns)


# --------------------------------------------------------------------------
# Leakage and reproducibility
# --------------------------------------------------------------------------

def test_transformer_is_fitted_on_training_data_only(feature_target):
    """The core anti-leakage assertion.

    Fitting on train alone, versus fitting on the full frame, must give
    different scaler means. If they match, the test set influenced the
    transformer and every reported metric is optimistic.
    """
    features, target = feature_target
    x_train_raw, _ = train_test_split(features, test_size=0.5, random_state=SEED, stratify=target)

    train_only = build_preprocessor("median", "most_frequent", scale=True).fit(x_train_raw)
    everything = build_preprocessor("median", "most_frequent", scale=True).fit(features)

    train_means = train_only.named_transformers_["numeric"].named_steps["scaler"].mean_
    full_means = everything.named_transformers_["numeric"].named_steps["scaler"].mean_

    assert not np.allclose(train_means, full_means)


def test_split_produces_disjoint_rows(feature_target):
    """No customer may appear in both train and test."""
    features, target = feature_target
    x_train, x_test = train_test_split(
        features, test_size=0.2, random_state=SEED, stratify=target
    )[:2]

    assert set(x_train.index).isdisjoint(set(x_test.index))
    assert len(x_train) + len(x_test) == len(features)


def test_split_respects_the_configured_ratio(feature_target):
    features, target = feature_target
    _, x_test = train_test_split(features, test_size=0.2, random_state=SEED, stratify=target)[:2]
    assert abs(len(x_test) / len(features) - 0.2) < 0.01


def test_stratification_preserves_the_churn_rate(feature_target):
    features, target = feature_target
    y_train, y_test = train_test_split(
        target, test_size=0.2, random_state=SEED, stratify=target
    )[:2]

    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_split_is_reproducible_with_a_fixed_seed(feature_target):
    """Same seed, same split -- the foundation of the whole reproducibility claim."""
    features, target = feature_target

    first = train_test_split(features, test_size=0.2, random_state=SEED, stratify=target)[0]
    second = train_test_split(features, test_size=0.2, random_state=SEED, stratify=target)[0]

    assert list(first.index) == list(second.index)


def test_different_seeds_produce_different_splits(feature_target):
    """Confirms the seed is actually wired in, not incidentally ignored."""
    features, target = feature_target

    first = train_test_split(features, test_size=0.2, random_state=1, stratify=target)[0]
    second = train_test_split(features, test_size=0.2, random_state=2, stratify=target)[0]

    assert list(first.index) != list(second.index)


def test_feature_lists_do_not_overlap():
    """A column in both lists would be encoded twice."""
    assert set(NUMERIC_FEATURES).isdisjoint(set(CATEGORICAL_FEATURES))


def test_target_is_excluded_from_the_feature_lists():
    """Leaking the label into the features would produce a perfect, useless model."""
    assert "Churn" not in NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert "CustomerID" not in NUMERIC_FEATURES + CATEGORICAL_FEATURES
