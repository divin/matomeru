"""Preflight checks for Matomeru dry runs.

A dry run verifies that a real run could start: required configuration is
present and consistent, and the state database location is writable. All
offline checks are side-effect-free: nothing is created, no state is
touched, no network call is made. With ``live=True`` an additional set of
reachability probes runs (TCP connect to SMTP; HTTP probes of the Sagashi,
Notes, LLM, and reranker endpoints).

Each check produces a ``CheckResult`` with status ``pass``, ``warn``, or
``fail``. Warnings describe conditions that are legal but suspicious (e.g.
an unauthenticated endpoint); failures mean the run cannot start. The dry
run exits 1 when any check fails.
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from pathlib import Path

import httpx

from matomeru.config import AppConfig, RuntimeConfig

logger = logging.getLogger(__name__)

PASS = "pass"
WARN = "warn"
FAIL = "fail"

_PROBE_TIMEOUT_S = 5.0


@dataclass
class CheckResult:
    """Outcome of one preflight check."""

    name: str
    status: str
    detail: str


def preflight_ok(checks: list[CheckResult]) -> bool:
    """Return True when no check has status ``fail``."""
    return not any(check.status == FAIL for check in checks)


def run_preflight(app: AppConfig, runtime: RuntimeConfig, source: str, *, live: bool = False) -> list[CheckResult]:
    """Run the offline preflight checks, plus live probes when ``live``.

    Offline checks never touch state or the network, preserving the dry-run
    contract. Live probes are opt-in: they attempt connections but never
    send a digest or modify anything.
    """
    checks: list[CheckResult] = [
        _check_state_db(runtime.state_db_path),
        _check_smtp_config(runtime),
        _check_llm_config(runtime),
        _check_sagashi_config(runtime),
        _check_notes_config(app, runtime, source),
        _check_reranker_config(runtime),
        _check_source_config(app, source),
    ]
    if live:
        checks.append(_check_smtp_reachable(runtime))
        checks.append(_check_http_endpoint("sagashi-mcp", runtime.sagashi_mcp_url, runtime.sagashi_mcp_token))
        checks.append(_check_http_endpoint("notes-mcp", runtime.notes_mcp_url, runtime.notes_mcp_token))
        checks.append(_check_http_endpoint("llm-api", runtime.llm_base_url, runtime.llm_api_key))
        if runtime.reranker_url:
            checks.append(_check_http_endpoint("reranker", runtime.reranker_url, runtime.reranker_token))
    return checks


def _check_state_db(state_db_path: str) -> CheckResult:
    """The state database's parent directory must be creatable and writable.

    Nothing is created: the nearest existing ancestor is probed for
    writability, so a dry run never materializes the state directory.
    """
    path = Path(state_db_path).expanduser()
    if path.exists() and path.is_dir():
        return CheckResult("state-db", FAIL, f"{path} is a directory")
    probe = path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.is_dir() or not os.access(probe, os.W_OK | os.X_OK):
        return CheckResult("state-db", FAIL, f"nearest existing ancestor {probe} of {path} is not writable")
    return CheckResult("state-db", PASS, f"parent dir of {path} is writable (nearest ancestor {probe})")


def _check_smtp_config(runtime: RuntimeConfig) -> CheckResult:
    """SMTP credentials must be consistent: user and password come as a pair."""
    if bool(runtime.smtp_user) != bool(runtime.smtp_password):
        return CheckResult("smtp-config", FAIL, "SMTP_USER and SMTP_PASSWORD must be set together or both unset")
    detail = f"user {'set' if runtime.smtp_user else 'unset'}, from={runtime.smtp_from}, to={runtime.smtp_to}"
    return CheckResult("smtp-config", PASS, detail)


def _check_llm_config(runtime: RuntimeConfig) -> CheckResult:
    """LLM endpoint and model are required by config; the API key is optional."""
    if runtime.llm_api_key is None:
        return CheckResult("llm-config", WARN, "LLM_API_KEY not set; only valid for unauthenticated endpoints")
    return CheckResult("llm-config", PASS, f"model={runtime.llm_model}, api key set")


def _check_sagashi_config(runtime: RuntimeConfig) -> CheckResult:
    """Sagashi MCP token is optional (some deployments are unauthenticated)."""
    if runtime.sagashi_mcp_token is None:
        return CheckResult("sagashi-config", WARN, "SAGASHI_MCP_TOKEN not set")
    return CheckResult("sagashi-config", PASS, "token set")


def _check_notes_config(app: AppConfig, runtime: RuntimeConfig, source: str) -> CheckResult:
    """Notes MCP token is optional; an empty folder skips notes delivery."""
    checks: list[str] = []
    folder = app.notes_reddit_folder if source == "reddit" else app.notes_arxiv_folder
    if not folder:
        checks.append(f"notes folder not configured for source={source}; notes delivery will be skipped")
    if runtime.notes_mcp_token is None:
        checks.append("NOTES_MCP_TOKEN not set")
    if checks:
        return CheckResult("notes-config", WARN, "; ".join(checks))
    return CheckResult("notes-config", PASS, f"folder={folder}, token set")


def _check_reranker_config(runtime: RuntimeConfig) -> CheckResult:
    """Reranker is optional; when set without a token, warn."""
    if runtime.reranker_url is None:
        return CheckResult("reranker-config", PASS, "not configured; funnel disabled")
    if runtime.reranker_token is None:
        return CheckResult("reranker-config", WARN, "RERANKER_URL set but RERANKER_TOKEN not set")
    return CheckResult("reranker-config", PASS, "url and token set")


def _check_source_config(app: AppConfig, source: str) -> CheckResult:
    """Per-source app config sanity: what would actually be fetched."""
    if source == "reddit":
        return CheckResult(
            "source-config",
            PASS,
            f"{len(app.reddit_subreddits)} subreddit(s): {', '.join(app.reddit_subreddits)}",
        )
    if source == "arxiv":
        if not app.arxiv_categories:
            return CheckResult("source-config", WARN, "no arxiv categories configured; nothing to fetch")
        return CheckResult(
            "source-config",
            PASS,
            f"{len(app.arxiv_categories)} category(ies), "
            f"{len(app.arxiv_search_terms)} search term(s), max_results={app.arxiv_max_results}",
        )
    return CheckResult("source-config", FAIL, f"unknown source: {source}")


def _check_smtp_reachable(runtime: RuntimeConfig) -> CheckResult:
    """TCP-connect to the SMTP server. Handshake and auth are not attempted."""
    try:
        with socket.create_connection((runtime.smtp_host, runtime.smtp_port), timeout=_PROBE_TIMEOUT_S):
            return CheckResult("smtp-reachable", PASS, f"tcp connect ok to {runtime.smtp_host}:{runtime.smtp_port}")
    except OSError as e:
        return CheckResult(
            "smtp-reachable", FAIL, f"tcp connect to {runtime.smtp_host}:{runtime.smtp_port} failed: {e}"
        )


def _check_http_endpoint(name: str, url: str, token: str | None) -> CheckResult:
    """Probe an HTTP endpoint for reachability.

    Any response status counts as reachable (MCP and LLM APIs commonly
    answer GET with 4xx); only transport-level errors are failures.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else None
    try:
        response = httpx.get(url, headers=headers, timeout=_PROBE_TIMEOUT_S, follow_redirects=False)
        return CheckResult(f"{name}-reachable", PASS, f"HTTP {response.status_code}")
    except httpx.RequestError as e:
        return CheckResult(f"{name}-reachable", FAIL, str(e))


def log_checks(checks: list[CheckResult]) -> int:
    """Log each check result; return the number of failures."""
    failures = 0
    for check in checks:
        if check.status == PASS:
            logger.info("preflight %s: %s (%s)", check.name, check.status, check.detail)
        else:
            logger.warning("preflight %s: %s (%s)", check.name, check.status, check.detail)
            if check.status == FAIL:
                failures += 1
    return failures
