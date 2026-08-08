"""Shared logging setup.

Pipeline stages run unattended under `dvc repro` and in CI, where the console
output is the only record of what happened. A consistent timestamped format
across every stage makes that log readable after the fact.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to stdout exactly once.

    Guarded against duplicate handlers: pytest imports these modules repeatedly
    within one process, and without the check every log line would be emitted
    once per import.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger
