"""CLI entry points for Matomeru pipeline sources."""

from __future__ import annotations

import argparse
import logging

from matomeru.config import ConfigError, load_app_config, load_runtime_config
from matomeru.logging_setup import configure_logging
from matomeru.state import begin_run
from matomeru.window import RunWindow, compute_run_window

logger = logging.getLogger(__name__)


def run_source_stub(source: str, window: RunWindow) -> None:
    """Stub pipeline function for a source.

    Logs that the pipeline is not yet implemented and returns without
    marking the run completed (status stays 'running' in phase 2b).
    """
    logger.info(
        "pipeline not yet implemented for source=%s report_date=%s window=[%s, %s]",
        source,
        window.report_date.isoformat(),
        window.start_utc.isoformat(),
        window.end_utc.isoformat(),
    )


def _build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--force", action="store_true", help="Restart the run even if already completed")
    parser.add_argument("--config", default=None, help="Path to matomeru.toml config file")
    parser.add_argument("--dry-run", action="store_true", help="Print the window plan and exit without touching state")
    return parser


def _run_source(source: str, argv: list[str] | None) -> int:
    """Shared CLI flow for a pipeline source.

    Returns:
        0 on success (dry-run, skipped, or stub run), 1 on ConfigError.
    """
    args = _build_parser(f"Run the {source} pipeline").parse_args(argv)

    try:
        configure_logging()
        load_app_config(args.config)
        runtime = load_runtime_config()

        window = compute_run_window(runtime.tz_iana)

        if args.dry_run:
            logger.info(
                "dry-run for source=%s: report_date=%s window=[%s, %s] (state not touched)",
                source,
                window.report_date.isoformat(),
                window.start_utc.isoformat(),
                window.end_utc.isoformat(),
            )
            return 0

        report_date, action = begin_run(runtime.state_db_path, source, window, force=args.force)

        if action == "skipped":
            logger.info(
                "skipping source=%s: run for %s already completed (use --force to restart)", source, report_date
            )
            return 0

        logger.info(
            "source=%s action=%s report_date=%s window=[%s, %s]",
            source,
            action,
            report_date,
            window.start_utc.isoformat(),
            window.end_utc.isoformat(),
        )

        run_source_stub(source, window)

        # Phase 2b: pipeline is a stub, so the run is intentionally left in
        # 'running' state (complete_run is not called yet).
        return 0

    except ConfigError as e:
        logging.error("%s", e)
        return 1


def main_reddit(argv: list[str] | None = None) -> int:
    """Entry point for the matomeru-reddit script."""
    return _run_source("reddit", argv)


def main_arxiv(argv: list[str] | None = None) -> int:
    """Entry point for the matomeru-arxiv script."""
    return _run_source("arxiv", argv)
