"""Shared pytest fixtures.

Every fixture here is synthetic. The tests must pass on a fresh clone in CI,
where `data/` is empty because the real CSVs live in DVC storage that the runner
has no credentials for. Tests that silently skip when data is absent would give a
green tick that proves nothing -- so the suite generates its own data instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES

SEED = 42


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Seeded generator so fixture data is identical on every run and machine."""
    return np.random.default_rng(SEED)


def _make_frame(n_rows: int, seed: int = SEED) -> pd.DataFrame:
    """Build a frame matching the real Kaggle schema exactly.

    Churn is derived from support calls and payment delay rather than drawn at
    random -- a genuinely unlearnable target would make every model score ~0.5
    and the model-quality assertions would be meaningless.
    """
    generator = np.random.default_rng(seed)

    support_calls = generator.integers(0, 11, n_rows)
    payment_delay = generator.integers(0, 31, n_rows)

    churn_logit = 0.35 * support_calls + 0.10 * payment_delay - 3.0
    churn_prob = 1 / (1 + np.exp(-churn_logit))

    return pd.DataFrame(
        {
            "CustomerID": np.arange(1, n_rows + 1),
            "Age": generator.integers(18, 70, n_rows),
            "Gender": generator.choice(["Male", "Female"], n_rows),
            "Tenure": generator.integers(1, 61, n_rows),
            "Usage Frequency": generator.integers(1, 31, n_rows),
            "Support Calls": support_calls,
            "Payment Delay": payment_delay,
            "Subscription Type": generator.choice(["Basic", "Standard", "Premium"], n_rows),
            "Contract Length": generator.choice(["Monthly", "Quarterly", "Annual"], n_rows),
            "Total Spend": generator.uniform(100, 1000, n_rows).round(2),
            "Last Interaction": generator.integers(1, 31, n_rows),
            "Churn": (generator.random(n_rows) < churn_prob).astype(int),
        }
    )


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A clean, well-formed sample of the raw dataset."""
    return _make_frame(500)


@pytest.fixture
def dirty_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """The same data with the defects the real CSV actually contains.

    Mirrors three concrete problems: an all-NaN row (present in the published
    training file), scattered nulls in numeric and categorical columns, and a
    duplicated CustomerID.
    """
    frame = raw_frame.copy()

    frame.loc[0, :] = np.nan
    frame.loc[1, "Age"] = np.nan
    frame.loc[2, "Total Spend"] = np.nan
    frame.loc[3, "Subscription Type"] = np.nan

    duplicate = frame.iloc[10].copy()
    frame = pd.concat([frame, duplicate.to_frame().T], ignore_index=True)

    return frame


@pytest.fixture
def feature_target(raw_frame: pd.DataFrame):
    """Raw features and target, split apart but not yet transformed."""
    return raw_frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES], raw_frame["Churn"].astype(int)
