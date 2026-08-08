"""Stage 1 -- pull the raw dataset from Kaggle and validate its shape.

This stage is deliberately thin: download, check the schema, report. It does not
clean or transform anything. Keeping ingestion separate from preprocessing means
DVC can cache the (slow, network-bound) download and skip it entirely when only
preprocessing parameters change.

Run:
    python -m src.data.ingest
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

from src.utils.config import ensure_dir, load_params, resolve_path
from src.utils.logger import get_logger

LOGGER = get_logger("data.ingest")

# Columns the downstream pipeline is written against. Ingestion fails loudly if
# Kaggle ever republishes this dataset with a different shape, rather than
# letting preprocessing produce silently wrong features.
EXPECTED_COLUMNS = {
    "CustomerID",
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
    "Churn",
}


def kaggle_credentials_available() -> bool:
    """True if the Kaggle client will be able to authenticate.

    Four mechanisms are accepted, because the client supports two generations of
    credential and the older one is still widely documented:

      * KAGGLE_API_TOKEN            -- current format, single opaque token
      * ~/.kaggle/access_token      -- the same token stored on disk
      * KAGGLE_USERNAME + KAGGLE_KEY -- legacy pair, still honoured
      * ~/.kaggle/kaggle.json       -- legacy pair stored on disk

    CI supplies the env var from a repository secret; a local machine normally
    uses the file. Checking all four means the error message below only appears
    when authentication genuinely cannot succeed.
    """
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True

    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (config_dir / "access_token").exists() or (config_dir / "kaggle.json").exists()


def download_from_kaggle(dataset: str, destination: Path) -> None:
    """Download and unzip a Kaggle dataset into `destination`.

    Imported lazily because `kaggle` authenticates at import time -- a top-level
    import would make this module unimportable (and therefore untestable) on any
    machine without credentials.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    LOGGER.info("Downloading %s from Kaggle...", dataset)
    api.dataset_download_files(dataset, path=str(destination), unzip=True)

    # Older kaggle clients ignore unzip=True in some paths; unpack any leftovers.
    for archive in destination.glob("*.zip"):
        LOGGER.info("Extracting %s", archive.name)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(destination)
        archive.unlink()


def validate_schema(frame: pd.DataFrame, source_name: str) -> None:
    """Assert the frame carries every column the pipeline depends on."""
    missing = EXPECTED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"{source_name} is missing expected column(s): {sorted(missing)}. "
            f"Found: {sorted(frame.columns)}"
        )

    unexpected = set(frame.columns) - EXPECTED_COLUMNS
    if unexpected:
        # Extra columns are survivable -- preprocessing selects explicitly -- so
        # this is a warning, not a failure.
        LOGGER.warning("%s has unexpected extra column(s): %s", source_name, sorted(unexpected))

    LOGGER.info("%s schema OK (%d columns)", source_name, len(frame.columns))


def summarise(frame: pd.DataFrame, source_name: str, target_column: str) -> None:
    """Log the handful of facts worth seeing in a pipeline log."""
    LOGGER.info("%s: %d rows x %d columns", source_name, len(frame), frame.shape[1])

    null_counts = frame.isnull().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls:
        offenders = null_counts[null_counts > 0].to_dict()
        LOGGER.warning("%s contains %d null value(s): %s", source_name, total_nulls, offenders)
    else:
        LOGGER.info("%s contains no null values", source_name)

    if target_column in frame.columns:
        rate = frame[target_column].mean()
        LOGGER.info("%s churn rate: %.4f", source_name, rate)


def main() -> int:
    params = load_params()
    ingest_cfg = params["ingest"]

    raw_dir = ensure_dir(ingest_cfg["raw_dir"])
    train_path = raw_dir / ingest_cfg["train_file"]
    test_path = raw_dir / ingest_cfg["test_file"]

    if train_path.exists() and test_path.exists():
        # Idempotent by design: re-running the stage must not re-download 400MB.
        LOGGER.info("Raw files already present in %s -- skipping download.", raw_dir)
    elif kaggle_credentials_available():
        download_from_kaggle(ingest_cfg["kaggle_dataset"], raw_dir)
    else:
        LOGGER.error(
            "Raw data missing and no Kaggle credentials found.\n"
            "  Fix by either:\n"
            "    1. Saving your API token to ~/.kaggle/access_token "
            "(chmod 600), or\n"
            "    2. Exporting KAGGLE_API_TOKEN, or\n"
            "    3. Downloading the CSVs manually from "
            "https://www.kaggle.com/datasets/%s into %s\n"
            "  Get a token at https://www.kaggle.com/settings -> API.",
            ingest_cfg["kaggle_dataset"],
            raw_dir,
        )
        return 1

    for path in (train_path, test_path):
        if not path.exists():
            LOGGER.error("Expected file not found after ingestion: %s", path)
            LOGGER.error("Files present: %s", sorted(p.name for p in raw_dir.iterdir()))
            return 1

    target_column = ingest_cfg["target_column"]
    for path in (train_path, test_path):
        frame = pd.read_csv(path)
        validate_schema(frame, path.name)
        summarise(frame, path.name, target_column)

    LOGGER.info("Ingestion complete. Raw data available at %s", resolve_path(raw_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
