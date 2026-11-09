"""Logging configuration for Matomeru."""

from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    """Configure Python logging for cron-friendly output.

    Format: `%(asctime)s %(levelname)s %(name)s %(message)s`
    Level: from MATOMERU_LOG_LEVEL env var (case-insensitive).
           Valid levels: DEBUG, INFO, WARNING, ERROR, CRITICAL.
           Default: INFO. Invalid values fall back to INFO with a warning.
    """
    level_name = os.environ.get("MATOMERU_LOG_LEVEL", "INFO").upper()
    valid_levels = logging.getLevelNamesMapping()
    if level_name in valid_levels:
        level = valid_levels[level_name]
    else:
        level = logging.INFO
        logging.getLogger(__name__).warning(f"Invalid MATOMERU_LOG_LEVEL '{level_name}', falling back to INFO")

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
