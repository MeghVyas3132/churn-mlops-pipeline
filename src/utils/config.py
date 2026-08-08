"""Configuration loading and path resolution.

Every script in this pipeline reads its settings through :func:`load_params`
rather than taking command-line arguments. That keeps `params.yaml` the single
place DVC has to watch to know a stage is stale, and it means a reviewer can
read one file to understand exactly how a run was configured.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# src/utils/config.py -> src/utils -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params(path: Path | str | None = None) -> dict[str, Any]:
    """Read `params.yaml` and return it as a plain dict.

    Raises FileNotFoundError with an actionable message rather than letting an
    obscure IOError surface three frames deeper.
    """
    params_path = Path(path) if path is not None else PARAMS_PATH
    if not params_path.exists():
        raise FileNotFoundError(
            f"params.yaml not found at {params_path}. "
            "Run pipeline commands from the project root."
        )

    with params_path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)

    if not isinstance(params, dict):
        raise ValueError(f"{params_path} did not parse to a mapping.")

    return params


def resolve_path(relative: str | Path) -> Path:
    """Turn a params.yaml path into an absolute one.

    Paths in params.yaml are written relative to the project root so the file
    stays portable across machines and CI runners. Absolute inputs pass through
    untouched, which is what the test suite relies on to redirect output into a
    temporary directory.
    """
    candidate = Path(relative)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if absent and return the resolved path."""
    resolved = resolve_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def set_global_seed(seed: int) -> None:
    """Pin every RNG this project can reach.

    scikit-learn and XGBoost take an explicit ``random_state``, but pandas
    sampling and any incidental shuffling fall back to the global numpy and
    stdlib generators -- so those need pinning too for a run to be repeatable.
    """
    random.seed(seed)
    np.random.seed(seed)
