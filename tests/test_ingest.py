"""Tests for the ingestion stage.

The job here is guarding the boundary: catching a changed upstream schema at
ingest time, rather than letting it turn into wrong features six stages later.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.ingest import (
    EXPECTED_COLUMNS,
    kaggle_credentials_available,
    summarise,
    validate_schema,
)


def test_expected_columns_cover_the_documented_schema():
    """Guards against a careless edit to EXPECTED_COLUMNS."""
    assert "Churn" in EXPECTED_COLUMNS, "target column must be part of the expected schema"
    assert "CustomerID" in EXPECTED_COLUMNS
    assert len(EXPECTED_COLUMNS) == 12


def test_validate_schema_accepts_a_conforming_frame(raw_frame):
    validate_schema(raw_frame, "raw_frame")  # must not raise


def test_validate_schema_rejects_a_missing_column(raw_frame):
    incomplete = raw_frame.drop(columns=["Total Spend"])
    with pytest.raises(ValueError, match="Total Spend"):
        validate_schema(incomplete, "incomplete")


def test_validate_schema_rejects_a_missing_target(raw_frame):
    with pytest.raises(ValueError, match="Churn"):
        validate_schema(raw_frame.drop(columns=["Churn"]), "no_target")


def test_validate_schema_tolerates_extra_columns(raw_frame):
    """Extra columns are survivable -- preprocessing selects explicitly."""
    extended = raw_frame.copy()
    extended["MarketingSegment"] = "A"
    validate_schema(extended, "extended")  # warns, must not raise


def test_validate_schema_error_names_every_missing_column(raw_frame):
    incomplete = raw_frame.drop(columns=["Age", "Tenure"])
    with pytest.raises(ValueError) as excinfo:
        validate_schema(incomplete, "incomplete")
    assert "Age" in str(excinfo.value)
    assert "Tenure" in str(excinfo.value)


def test_summarise_handles_a_frame_containing_nulls(raw_frame, caplog):
    """Must log and continue, not crash -- the real CSV has nulls."""
    with_nulls = raw_frame.copy()
    with_nulls.loc[0, "Age"] = None
    summarise(with_nulls, "with_nulls", "Churn")


def test_summarise_handles_an_empty_frame():
    empty = pd.DataFrame({column: [] for column in EXPECTED_COLUMNS})
    summarise(empty, "empty", "Churn")


def test_kaggle_credentials_check_reads_env_vars(monkeypatch, tmp_path):
    """Env-var auth is the path CI uses via repository secrets."""
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("KAGGLE_USERNAME", "someone")
    monkeypatch.setenv("KAGGLE_KEY", "a-key")
    assert kaggle_credentials_available() is True


def test_kaggle_credentials_check_reads_config_file(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))

    assert kaggle_credentials_available() is False

    (tmp_path / "kaggle.json").write_text('{"username": "someone", "key": "a-key"}')
    assert kaggle_credentials_available() is True
