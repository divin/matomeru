"""Deterministic digest rendering for Reddit and arXiv sources.

Templates are code-owned: the LLM text is slotted in verbatim, and all
paths, links, and layout come from code. These functions are pure — no I/O
and no clock access — so output is a deterministic function of their inputs.

The ``partial`` flag is a parameter passed by orchestration (phase 5b); it is
never computed here.
"""

from __future__ import annotations

from matomeru.llm import ArxivPaperSummary, RedditItemSummary
from matomeru.models import ArxivPaper, RedditThread, ThreadFetchError
from matomeru.window import RunWindow

REDDIT_NO_RESULTS = "No posts matched the configured subreddits in this window."
REDDIT_ALL_FAILED = "Posts were collected in this window, but none could be summarized (see the Errors section below)."
ARXIV_NO_RESULTS = "No relevant arXiv papers matched the configured categories and search terms in this window."
ARXIV_ALL_FAILED = "Papers were collected in this window, but none could be summarized (see the Errors section below)."

PARTIAL_LINE = "Partial run: some items failed and are omitted."
SCREENED_OUT_LINE = "{count} further paper(s) were screened out as not relevant."


def _window_line(window: RunWindow) -> str:
    return f"Window (UTC): {window.start_utc.isoformat()} → {window.end_utc.isoformat()}"


def _errors_section(errors: list[str]) -> list[str]:
    lines: list[str] = ["## Errors"]
    for error in errors:
        lines.append(f"- {error}")
    return lines


def _render(
    *,
    heading: str,
    window: RunWindow,
    item_blocks: list[str],
    no_results_body: str,
    all_failed_body: str,
    errors: list[str],
    trailing_lines: list[str],
) -> str:
    """Assemble a digest body from already-rendered pieces (fixed template).

    With no items, a genuinely empty window renders ``no_results_body``;
    items that were collected but all lost to errors render
    ``all_failed_body`` instead, so the two situations read differently.
    """
    lines: list[str] = [heading, _window_line(window), ""]

    if item_blocks:
        lines.extend(item_blocks)
    elif errors:
        lines.append(all_failed_body)
        lines.append("")
    else:
        lines.append(no_results_body)
        lines.append("")

    if errors:
        lines.extend(_errors_section(errors))
        lines.append("")

    lines.extend(trailing_lines)

    # Collapse the final blank line(s): the last line is always real content.
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def render_reddit_digest(
    report_date: str,
    window: RunWindow,
    items: list[tuple[RedditThread, RedditItemSummary]],
    fetch_errors: list[ThreadFetchError],
    *,
    partial: bool,
) -> str:
    """Render the Reddit digest markdown.

    Args:
        report_date: Report date string (e.g. "2026-11-09").
        window: The run window (start/end rendered as UTC ISO timestamps).
        items: (thread, summary) pairs in the order they should appear.
        fetch_errors: Per-thread fetch errors, rendered in an Errors section.
        partial: Orchestration-supplied flag; when True a visible partial-run
            line is appended.

    Returns:
        The digest markdown string.
    """
    item_blocks: list[str] = []
    for thread, summary in items:
        author = thread.author if thread.author is not None else "unknown"
        block = [
            f"## r/{thread.subreddit} — {thread.title}",
            f"- Score: {thread.score}",
            f"- Author: {author}",
            f"- Link: https://reddit.com{thread.permalink}",
            f"- {thread.num_comments} comments",
            "",
            "Summary:",
            summary.summary,
            "",
            "Why relevant:",
            summary.why_relevant,
            "",
        ]
        item_blocks.extend(block)

    trailing: list[str] = []
    if partial:
        trailing.append(PARTIAL_LINE)

    return _render(
        heading=f"# Reddit digest — {report_date}",
        window=window,
        item_blocks=item_blocks,
        no_results_body=REDDIT_NO_RESULTS,
        all_failed_body=REDDIT_ALL_FAILED,
        errors=[f"{error.item}: {error.error}" for error in fetch_errors],
        trailing_lines=trailing,
    )


def render_arxiv_digest(
    report_date: str,
    window: RunWindow,
    items: list[tuple[ArxivPaper, ArxivPaperSummary]],
    screened_out: int,
    fetch_errors: list[str],
    *,
    partial: bool,
) -> str:
    """Render the arXiv digest markdown.

    Args:
        report_date: Report date string (e.g. "2026-11-09").
        window: The run window (start/end rendered as UTC ISO timestamps).
        items: (paper, summary) pairs in the order they should appear.
        screened_out: Number of papers excluded by relevance screening.
        fetch_errors: Fetch error strings, rendered in an Errors section.
        partial: Orchestration-supplied flag; when True a visible partial-run
            line is appended.

    Returns:
        The digest markdown string.
    """
    item_blocks: list[str] = []
    for paper, summary in items:
        block = [
            f"## {paper.title}",
            f"- Authors: {', '.join(paper.authors)}",
            f"- Categories: {', '.join(paper.categories)}",
            f"- Link: {paper.abs_url}",
            "",
            "Summary:",
            summary.summary,
            "",
            "Why relevant:",
            summary.why_relevant,
            "",
        ]
        item_blocks.extend(block)

    trailing: list[str] = []
    if screened_out > 0:
        trailing.append(SCREENED_OUT_LINE.format(count=screened_out))
    if partial:
        trailing.append(PARTIAL_LINE)

    return _render(
        heading=f"# arXiv digest — {report_date}",
        window=window,
        item_blocks=item_blocks,
        no_results_body=ARXIV_NO_RESULTS,
        all_failed_body=ARXIV_ALL_FAILED,
        errors=list(fetch_errors),
        trailing_lines=trailing,
    )
