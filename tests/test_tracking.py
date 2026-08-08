"""Tests for MLflow tracking configuration.

A tracking URI that silently resolves to the wrong place sends runs to a store
nobody looks at -- the experiment tracking requirement fails without anything
appearing to be broken. These tests pin the resolution rules.
"""

from __future__ import annotations

from src.utils.config import PROJECT_ROOT, load_params
from src.utils.tracking import is_local_store, resolve_tracking_uri


def test_relative_sqlite_path_resolves_against_the_project_root():
    """Running a stage from a subdirectory must not create a second database."""
    resolved = resolve_tracking_uri("sqlite:///mlflow.db")
    assert resolved == f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


def test_absolute_sqlite_path_is_preserved(tmp_path):
    absolute = tmp_path / "custom.db"
    assert resolve_tracking_uri(f"sqlite:///{absolute}") == f"sqlite:///{absolute}"


def test_remote_tracking_server_uri_passes_through():
    """Swapping in a hosted tracking server must need no code change."""
    assert resolve_tracking_uri("http://localhost:5000") == "http://localhost:5000"


def test_bare_directory_becomes_a_file_uri():
    resolved = resolve_tracking_uri("mlruns")
    assert resolved.startswith("file://")
    assert resolved.endswith("/mlruns")


def test_configured_tracking_uri_is_resolvable():
    """Guards against a typo in params.yaml reaching the pipeline."""
    resolved = resolve_tracking_uri(load_params()["mlflow"]["tracking_uri"])
    assert "://" in resolved


def test_experiment_and_registered_model_names_are_configured():
    mlflow_cfg = load_params()["mlflow"]
    assert mlflow_cfg["experiment_name"]
    assert mlflow_cfg["registered_model_name"]
    assert mlflow_cfg["artifact_location"]


def test_sqlite_and_file_uris_are_local_stores():
    assert is_local_store("sqlite:////tmp/mlflow.db") is True
    assert is_local_store("file:///tmp/mlruns") is True


def test_http_tracking_server_is_not_a_local_store():
    """Decides whether we may hand MLflow a local artifact path.

    Passing one to a hosted server (DagsHub, Databricks) points the experiment at
    a directory the server cannot write to, and artifact logging then fails.
    """
    assert is_local_store("https://dagshub.com/user/repo.mlflow") is False
    assert is_local_store("http://localhost:5000") is False


def test_configured_backend_is_recognised_as_local():
    """The committed default must never require a running server."""
    resolved = resolve_tracking_uri(load_params()["mlflow"]["tracking_uri"])
    assert is_local_store(resolved) is True
