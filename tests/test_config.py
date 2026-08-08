"""Tests for configuration loading and reproducibility helpers.

`params.yaml` is the contract between every stage. If a key is renamed or
deleted, the pipeline breaks at runtime -- these tests turn that into an
immediate, obvious failure instead.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from src.utils.config import (
    PROJECT_ROOT,
    ensure_dir,
    load_params,
    resolve_path,
    set_global_seed,
)
from src.utils.logger import get_logger


def test_params_file_exists():
    assert (PROJECT_ROOT / "params.yaml").exists(), "params.yaml is missing from the project root"


def test_load_params_returns_mapping():
    params = load_params()
    assert isinstance(params, dict)
    assert params, "params.yaml parsed to an empty mapping"


@pytest.mark.parametrize(
    "section", ["seed", "ingest", "preprocess", "train", "evaluate", "mlflow"]
)
def test_required_top_level_sections_present(section):
    """Each pipeline stage reads one of these -- none may go missing."""
    assert section in load_params(), f"params.yaml is missing the '{section}' section"


def test_seed_is_an_integer():
    """A non-integer seed silently breaks reproducibility rather than erroring."""
    assert isinstance(load_params()["seed"], int)


def test_test_size_is_a_valid_fraction():
    test_size = load_params()["preprocess"]["test_size"]
    assert 0 < test_size < 1, f"test_size must be a fraction between 0 and 1, got {test_size}"


def test_at_least_two_models_enabled():
    """Requirement: train and compare *multiple* models."""
    models = load_params()["train"]["models"]
    enabled = [name for name, cfg in models.items() if cfg.get("enabled", True)]
    assert len(enabled) >= 2, f"Need at least two enabled models to compare, found {enabled}"


def test_primary_metric_is_recognised():
    metric = load_params()["train"]["primary_metric"]
    assert metric in {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_load_params_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_params(tmp_path / "does_not_exist.yaml")


def test_resolve_path_makes_relative_paths_absolute():
    resolved = resolve_path("data/raw")
    assert resolved.is_absolute()
    assert resolved == PROJECT_ROOT / "data" / "raw"


def test_resolve_path_leaves_absolute_paths_alone(tmp_path):
    """The test suite depends on this to redirect outputs into tmp_path."""
    assert resolve_path(tmp_path) == Path(tmp_path)


def test_ensure_dir_creates_missing_directories(tmp_path):
    target = tmp_path / "nested" / "deeper"
    assert not target.exists()
    assert ensure_dir(target).exists()


def test_ensure_dir_is_idempotent(tmp_path):
    """Stages re-run under `dvc repro`; a second call must not raise."""
    target = tmp_path / "repeated"
    ensure_dir(target)
    ensure_dir(target)
    assert target.exists()


def test_set_global_seed_makes_numpy_deterministic():
    set_global_seed(42)
    first = np.random.rand(10)
    set_global_seed(42)
    assert np.array_equal(first, np.random.rand(10))


def test_set_global_seed_makes_stdlib_random_deterministic():
    set_global_seed(7)
    first = [random.random() for _ in range(5)]
    set_global_seed(7)
    assert first == [random.random() for _ in range(5)]


def test_get_logger_does_not_stack_handlers():
    """Repeated imports under pytest must not duplicate every log line."""
    first = get_logger("test.duplication")
    handler_count = len(first.handlers)
    second = get_logger("test.duplication")
    assert second is first
    assert len(second.handlers) == handler_count
