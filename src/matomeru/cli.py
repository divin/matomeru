"""CLI entry points for Matomeru pipeline sources.

Phase 5b orchestration: each CLI collects its own source, summarizes items
sequentially via the LLM, renders the digest deterministically, delivers it
to two independent destinations (email + notes vault), and records run state.

Semantics:
- ``--dry-run`` prints the window plan, then runs preflight checks (offline
  by default; ``--live`` adds network reachability probes) and exits 0 only
  if no check failed. State is never touched and no digest is fetched.
- ``--live`` requires ``--dry-run``.
- ``begin_run`` returning ``skipped`` exits 0 without touching anything;
  ``restart`` (also reached via ``--force``) proceeds normally.
- Item-level LLM failures exclude that item and mark the run partial; the
  remaining items are still delivered. Destination failures are isolated:
  one failing destination never prevents the other.
- A run with partial content or any failed destination is finalized with
  ``fail_run`` and exits 1; a fully successful run (including an empty
  day, which still renders a no-results digest) is finalized with
  ``complete_run`` and exits 0.
- Every state update after ``begin_run`` is best-effort: a state glitch is
  logged but must not prevent delivery.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import httpx
from pydantic_ai import Agent

from matomeru.arxiv import collect_arxiv
from matomeru.config import (
    AppConfig,
    ConfigError,
    RuntimeConfig,
    load_app_config,
    load_runtime_config,
    render_profile_markdown,
)
from matomeru.digest import (
    render_arxiv_digest,
    render_arxiv_digest_html,
    render_reddit_digest,
    render_reddit_digest_html,
)
from matomeru.funnel import funnel_arxiv_papers, funnel_reddit_threads
from matomeru.llm import (
    ArxivPaperSummary,
    ArxivRelevanceDecision,
    RedditItemSummary,
    make_agent,
    screen_arxiv_paper,
    summarize_arxiv_paper,
    summarize_reddit_item,
)
from matomeru.logging_setup import configure_logging
from matomeru.mail import MailError, build_digest_message, digest_subject, send_mail
from matomeru.models import ArxivPaper, RedditThread, ThreadFetchError
from matomeru.notes import NotesClientError, NotesMcpClient, build_note_path
from matomeru.preflight import log_checks, run_preflight
from matomeru.reddit import collect_reddit
from matomeru.reranker_client import RerankerClient
from matomeru.sagashi_client import SagashiClient
from matomeru.state import begin_run, complete_run, fail_run, mark_delivery, record_rerank_scores
from matomeru.window import RunWindow, compute_run_window

logger = logging.getLogger(__name__)


@dataclass
class _DigestData:
    """Intermediate result of collection + LLM processing for one source."""

    body: str
    html_body: str
    partial: bool
    item_count: int
    extra_summary: str


def _best_effort_state(description: str, action: Callable[[], None]) -> None:
    """Run a state-store update, logging instead of raising on failure.

    State glitches after ``begin_run`` must never crash delivery; genuine
    programming errors still surface as logged warnings here.
    """
    try:
        action()
    except Exception as e:  # noqa: BLE001 - best-effort by contract
        logger.warning("state update failed (%s): %s", description, e)


async def _pipeline_reddit(
    app: AppConfig, runtime: RuntimeConfig, profile_text: str, window: RunWindow, report_date: str
) -> _DigestData:
    """Collect Reddit threads, funnel them, summarize, and render the digest.

    Collection failure propagates (aborts the run); per-thread LLM failures
    are logged, excluded from the digest, and mark the run partial. When the
    reranker is configured, only the funnel-selected threads are summarized;
    dropped threads are excluded by design, not failures. With no reranker
    URL the funnel is skipped and v1 behavior is unchanged.
    """
    agent = make_agent(runtime)
    client = SagashiClient(runtime.sagashi_mcp_url, runtime.sagashi_mcp_token)
    threads, fetch_errors = await collect_reddit(
        client,
        app.reddit_subreddits,
        window,
        posts_per_subreddit=app.reddit_posts_per_subreddit,
        comments_per_post=app.reddit_comments_per_post,
    )
    logger.info("collected %d reddit thread(s), %d fetch error(s)", len(threads), len(fetch_errors))

    # Reranker funnel (v1.1.0): when configured, score every collected thread
    # and keep only the top X per subreddit for summarization. Funnel failures
    # propagate and fail the run (fail-early, like collection failures);
    # dropped threads are excluded by design and never mark the run partial.
    # Without a reranker URL the funnel is skipped entirely (v1 behavior).
    funnel_line: str | None = None
    if runtime.reranker_url is not None and threads:
        collected_count = len(threads)
        reranker = RerankerClient(runtime.reranker_url, runtime.reranker_token)
        outcome = await funnel_reddit_threads(
            threads,
            [f"{section.name}\n\n{section.text}" for section in app.profile_sections],
            reranker,
            top_x=app.reranker_reddit_top_x,
            min_score=app.reranker_min_score,
            report_date=report_date,
        )
        _best_effort_state(
            "record rerank scores",
            lambda: record_rerank_scores(runtime.state_db_path, "reddit", report_date, outcome.scored),
        )
        for scored_item in outcome.scored:
            if not scored_item.kept:
                logger.debug(
                    "funnel dropped thread id=%s subreddit=%s score=%s",
                    scored_item.item_id,
                    scored_item.subreddit,
                    scored_item.score,
                )
        logger.info("funnel kept %d of %d reddit thread(s)", len(outcome.kept), collected_count)
        threads = outcome.kept
        funnel_line = f"Ranked {collected_count} threads, kept {len(outcome.kept)}"

    items: list[tuple[RedditThread, RedditItemSummary]] = []
    llm_failures: list[ThreadFetchError] = []
    for thread in threads:
        try:
            summary = await summarize_reddit_item(agent, profile_text, thread)
        except Exception as e:  # noqa: BLE001 - one broken item must not kill the run
            logger.warning("summarization failed for thread id=%s: %s; excluding from digest", thread.id, e)
            llm_failures.append(ThreadFetchError(index=len(llm_failures), item=thread.id, error=str(e)))
            continue
        items.append((thread, summary))

    partial = bool(fetch_errors or llm_failures)
    body = render_reddit_digest(
        report_date,
        window,
        items,
        [*fetch_errors, *llm_failures],
        partial=partial,
        funnel_line=funnel_line,
    )
    html_body = render_reddit_digest_html(
        report_date,
        window,
        items,
        [*fetch_errors, *llm_failures],
        partial=partial,
        funnel_line=funnel_line,
    )
    return _DigestData(body=body, html_body=html_body, partial=partial, item_count=len(items), extra_summary="")


async def _pipeline_arxiv(
    app: AppConfig, runtime: RuntimeConfig, profile_text: str, window: RunWindow, report_date: str
) -> _DigestData:
    """Collect arXiv papers, screen/funnel them, summarize the relevant, render.

    Collection failure propagates (aborts the run). When the reranker is
    configured, the deterministic threshold funnel replaces the LLM screening
    stage entirely (zero screening calls): a funnel failure propagates and
    fails the run (fail-early, like collection failures), while dropped
    papers are excluded by design and never mark the run partial. Without a
    reranker URL the v1 behavior is unchanged: screening failure is treated
    as excluded (the conservative default already excludes the paper, so it
    does NOT mark the run partial — an Errors-section entry is still
    rendered). Summarization failure on an included paper excludes that item
    and marks the run partial.
    """
    # make_agent is a single entry point whose annotation reflects the Reddit
    # use case; the same OpenAI-compatible agent serves both structured output
    # types, so narrow via cast at the call sites (no llm.py API change).
    agent = cast("Agent[None, ArxivRelevanceDecision]", make_agent(runtime))
    summary_agent = cast("Agent[None, ArxivPaperSummary]", agent)
    async with httpx.AsyncClient() as http:
        papers, fetched_total = await collect_arxiv(
            http,
            app.arxiv_categories,
            app.arxiv_search_terms,
            window,
            max_results=app.arxiv_max_results,
            category_terms=app.arxiv_category_terms,
        )
    logger.info("collected %d arxiv paper(s) in window (fetched=%d)", len(papers), fetched_total)

    fetch_errors: list[str] = []
    included: list[ArxivPaper] = []

    # Reranker funnel (v1.1.0): when configured, score every collected paper
    # and keep those at or above the threshold for summarization — the LLM
    # screening stage is bypassed entirely (zero screening calls). Funnel
    # failures propagate and fail the run (fail-early, like collection
    # failures); dropped papers are excluded by design and never mark the run
    # partial. Without a reranker URL the v1 LLM screening stage runs as
    # before.
    funnel_line: str | None = None
    if runtime.reranker_url is not None and papers:
        collected_count = len(papers)
        reranker = RerankerClient(runtime.reranker_url, runtime.reranker_token)
        outcome = await funnel_arxiv_papers(
            papers,
            [f"{section.name}\n\n{section.text}" for section in app.profile_sections],
            reranker,
            min_score=app.reranker_arxiv_min_score,
            report_date=report_date,
        )
        _best_effort_state(
            "record rerank scores",
            lambda: record_rerank_scores(runtime.state_db_path, "arxiv", report_date, outcome.scored),
        )
        for scored_item in outcome.scored:
            if not scored_item.kept:
                logger.debug(
                    "funnel dropped paper id=%s category=%s score=%s",
                    scored_item.item_id,
                    scored_item.subreddit,
                    scored_item.score,
                )
        logger.info("funnel kept %d of %d arxiv paper(s)", len(outcome.kept), collected_count)
        included = outcome.kept
        funnel_line = f"Ranked {collected_count} papers, kept {len(outcome.kept)}"
    else:
        for paper in papers:
            try:
                decision = await screen_arxiv_paper(agent, profile_text, paper)
            except Exception as e:  # noqa: BLE001 - conservative default excludes the paper
                logger.warning(
                    "screening failed for paper arxiv_id=%s: %s; treating as excluded (conservative default)",
                    paper.arxiv_id,
                    e,
                )
                fetch_errors.append(f"screening failed for {paper.arxiv_id}: {e}")
                continue
            if decision.include:
                included.append(paper)

    items: list[tuple[ArxivPaper, ArxivPaperSummary]] = []
    for paper in included:
        try:
            summary = await summarize_arxiv_paper(summary_agent, profile_text, paper)
        except Exception as e:  # noqa: BLE001 - one broken item must not kill the run
            logger.warning("summarization failed for paper arxiv_id=%s: %s; excluding from digest", paper.arxiv_id, e)
            continue
        items.append((paper, summary))

    partial = len(items) < len(included)
    screened_out = len(papers) - len(items)
    body = render_arxiv_digest(
        report_date, window, items, screened_out, fetch_errors, partial=partial, funnel_line=funnel_line
    )
    html_body = render_arxiv_digest_html(
        report_date, window, items, screened_out, fetch_errors, partial=partial, funnel_line=funnel_line
    )
    return _DigestData(
        body=body,
        html_body=html_body,
        partial=partial,
        item_count=len(items),
        extra_summary=f"screened_out={screened_out}",
    )


async def _deliver_and_finalize(
    source: str, app: AppConfig, runtime: RuntimeConfig, window: RunWindow, report_date: str, digest: _DigestData
) -> bool:
    """Deliver the digest to email and notes, then finalize the run row.

    Destinations are independent: a failure in one never prevents the other.
    Returns True when the run should be marked completed (exit 0), False
    when partial content or a failed destination means the run failed.
    """
    # --- Email ---
    email_state = "failed"
    recipients = [r.strip() for r in runtime.smtp_to.split(",") if r.strip()]
    try:
        subject = digest_subject(source, report_date)
        message = build_digest_message(
            sender=runtime.smtp_from,
            recipients=recipients,
            subject=subject,
            body=digest.body,
            sent_at=window.end_utc,
            html_body=digest.html_body,
        )
        await send_mail(
            runtime.smtp_host,
            runtime.smtp_port,
            runtime.smtp_user or None,
            runtime.smtp_password or None,
            runtime.smtp_from,
            recipients,
            message,
        )
        email_state = "sent"
    except MailError as e:
        logger.error("email delivery failed for source=%s: %s", source, e)
    _best_effort_state(
        "mark email_sent",
        lambda: mark_delivery(runtime.state_db_path, source, report_date, email_sent=(email_state == "sent")),
    )

    # --- Notes ---
    folder = app.notes_reddit_folder if source == "reddit" else app.notes_arxiv_folder
    notes_state = "skipped"
    if not folder:
        logger.warning("notes folder not configured for source=%s; skipping notes delivery", source)
    else:
        notes_state = "failed"
        try:
            client = NotesMcpClient(runtime.notes_mcp_url, runtime.notes_mcp_token)
            await client.ensure_folder(folder)
            note_path = build_note_path(folder, report_date)
            await client.save_note(note_path, digest.body)
            logger.info("saved note %s", note_path)
            notes_state = "saved"
        except NotesClientError as e:
            logger.error("notes delivery failed for source=%s: %s", source, e)
        _best_effort_state(
            "mark note_saved",
            lambda: mark_delivery(runtime.state_db_path, source, report_date, note_saved=(notes_state == "saved")),
        )

    # --- Finalize ---
    run_failed = digest.partial or email_state == "failed" or notes_state == "failed"
    if run_failed:
        _best_effort_state("fail_run", lambda: fail_run(runtime.state_db_path, source, report_date))
        logger.error(
            "run incomplete for source=%s report_date=%s: items=%d partial=%s email=%s notes=%s %s",
            source,
            report_date,
            digest.item_count,
            digest.partial,
            email_state,
            notes_state,
            digest.extra_summary,
        )
        return False

    _best_effort_state("complete_run", lambda: complete_run(runtime.state_db_path, source, report_date))
    logger.info(
        "run completed for source=%s report_date=%s: items=%d email=%s notes=%s %s",
        source,
        report_date,
        digest.item_count,
        email_state,
        notes_state,
        digest.extra_summary,
    )
    return True


async def _execute_run(
    source: str, app: AppConfig, runtime: RuntimeConfig, profile_text: str, window: RunWindow, report_date: str
) -> bool:
    """Run the collect/process/render/deliver pipeline for one source.

    Collection failures (network, API, deterministic config/data errors)
    abort this run: the run row is marked failed and False is returned.
    """
    try:
        if source == "reddit":
            digest = await _pipeline_reddit(app, runtime, profile_text, window, report_date)
        elif source == "arxiv":
            digest = await _pipeline_arxiv(app, runtime, profile_text, window, report_date)
        else:
            raise ValueError(f"unknown source: {source}")
    except Exception as e:  # noqa: BLE001 - any collection failure fails the run
        logger.error("collection failed for source=%s: %s", source, e)
        _best_effort_state("fail_run", lambda: fail_run(runtime.state_db_path, source, report_date))
        return False
    return await _deliver_and_finalize(source, app, runtime, window, report_date, digest)


def _build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--force", action="store_true", help="Restart the run even if already completed")
    parser.add_argument("--config", default=None, help="Path to matomeru.toml config file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run preflight checks and print the window plan without touching state"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="With --dry-run, also probe network endpoints (SMTP TCP connect, MCP/LLM reachability)",
    )
    return parser


def _run_source(source: str, argv: list[str] | None) -> int:
    """Shared CLI flow for a pipeline source.

    Returns:
        0 on success (dry-run, skipped, or fully successful run),
        1 on ConfigError or an incomplete (partial/failed) run.
    """
    parser = _build_parser(f"Run the {source} pipeline")
    args = parser.parse_args(argv)
    if args.live and not args.dry_run:
        parser.error("--live requires --dry-run")

    runtime: RuntimeConfig | None = None
    report_date: str | None = None
    try:
        configure_logging()
        app = load_app_config(args.config)
        runtime = load_runtime_config()
        profile_text = render_profile_markdown(app.profile_sections)

        window = compute_run_window(runtime.tz_iana)

        if args.dry_run:
            logger.info(
                "dry-run for source=%s: report_date=%s window=[%s, %s] (state not touched)",
                source,
                window.report_date.isoformat(),
                window.start_utc.isoformat(),
                window.end_utc.isoformat(),
            )
            failures = log_checks(run_preflight(app, runtime, source, live=args.live))
            if failures:
                logger.error("dry-run preflight failed for source=%s: %d check(s) failed", source, failures)
                return 1
            logger.info("dry-run preflight passed for source=%s", source)
            return 0

        begin_report_date, action = begin_run(runtime.state_db_path, source, window, force=args.force)
        report_date = begin_report_date

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

        ok = asyncio.run(_execute_run(source, app, runtime, profile_text, window, report_date))
        return 0 if ok else 1

    except ConfigError as e:
        logging.error("%s", e)
        return 1
    except Exception:
        logger.exception("unexpected failure for source=%s", source)
        if runtime is not None and report_date is not None:
            _best_effort_state("fail_run", lambda: fail_run(runtime.state_db_path, source, report_date))
        return 1


def main_reddit(argv: list[str] | None = None) -> int:
    """Entry point for the matomeru-reddit script."""
    return _run_source("reddit", argv)


def main_arxiv(argv: list[str] | None = None) -> int:
    """Entry point for the matomeru-arxiv script."""
    return _run_source("arxiv", argv)
