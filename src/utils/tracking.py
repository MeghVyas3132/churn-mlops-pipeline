"""MLflow setup shared by the training and evaluation stages.

Both stages must agree on the tracking backend and experiment, otherwise their
runs land in different stores and the comparison view is useless. Centralising
that here means there is exactly one place to change it.

Note on the backend: MLflow 3.x has put the plain-directory file store into
maintenance mode and refuses to open one without an opt-out env var. A local
SQLite file is the supported replacement, needs no server, and has the side
benefit of enabling the Model Registry -- which the file store never supported.
"""

from __future__ import annotations

import os

import mlflow

from src.utils.config import ensure_dir, resolve_path
from src.utils.logger import get_logger

LOGGER = get_logger("utils.tracking")

_SQLITE_PREFIX = "sqlite:///"


def is_local_store(tracking_uri: str) -> bool:
    """True if runs are written to this machine rather than a tracking server.

    Drives two decisions: whether we may set an artifact location (a server owns
    its own storage) and whether credentials are needed.
    """
    return tracking_uri.startswith((_SQLITE_PREFIX, "file://"))


def resolve_tracking_uri(raw_uri: str) -> str:
    """Turn the `params.yaml` tracking URI into something MLflow accepts anywhere.

    Relative paths in the config are resolved against the project root so a run
    started from a subdirectory still writes to the same store.
    """
    if raw_uri.startswith(_SQLITE_PREFIX):
        return f"{_SQLITE_PREFIX}{resolve_path(raw_uri[len(_SQLITE_PREFIX):])}"

    if "://" in raw_uri:
        # A remote tracking server (http://, databricks://) is passed through.
        return raw_uri

    # A bare path means a local directory store.
    return resolve_path(raw_uri).as_uri()


def setup_mlflow(mlflow_cfg: dict) -> str:
    """Point MLflow at the configured backend and select the experiment.

    Returns the resolved tracking URI so the caller can log it.
    """
    # MLFLOW_TRACKING_URI wins over params.yaml. That is what lets the same
    # commit log to a local SQLite file on a laptop and to a hosted tracking
    # server in CI -- without a server URL or credentials ever entering Git.
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        tracking_uri = env_uri
        LOGGER.info("Using MLFLOW_TRACKING_URI from the environment")
    else:
        tracking_uri = resolve_tracking_uri(mlflow_cfg["tracking_uri"])

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = mlflow_cfg["experiment_name"]

    # set_experiment() cannot specify an artifact location, so the experiment is
    # created explicitly the first time to keep artifacts inside the project.
    # A remote server manages its own artifact storage, so we must not try to
    # hand it a local path it cannot write to.
    if mlflow.get_experiment_by_name(experiment_name) is None:
        if is_local_store(tracking_uri):
            artifact_location = ensure_dir(mlflow_cfg["artifact_location"]).as_uri()
            mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
            LOGGER.info("Created MLflow experiment '%s' at %s", experiment_name, artifact_location)
        else:
            mlflow.create_experiment(experiment_name)
            LOGGER.info("Created MLflow experiment '%s' on the tracking server", experiment_name)

    mlflow.set_experiment(experiment_name)
    LOGGER.info("MLflow tracking URI: %s (experiment: %s)", tracking_uri, experiment_name)

    return tracking_uri


def log_model(model, name: str, input_example=None, registered_model_name: str | None = None):
    """Log an estimator using the MLflow flavor that matches it.

    Not cosmetic. MLflow 3.x serialises `mlflow.sklearn` models with skops, which
    refuses to write types it does not trust -- and `xgboost.Booster` is on that
    list, so routing an XGBClassifier through the sklearn flavor raises. Sending
    it to `mlflow.xgboost` instead uses XGBoost's own format and additionally
    records the correct flavor metadata for anything loading the model later.
    """
    import mlflow.sklearn
    import mlflow.xgboost
    from xgboost import XGBModel

    flavor = mlflow.xgboost if isinstance(model, XGBModel) else mlflow.sklearn

    return flavor.log_model(
        model,
        name=name,
        input_example=input_example,
        registered_model_name=registered_model_name,
    )
