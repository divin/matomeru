"""Deterministic digest rendering for Reddit and arXiv sources.

Templates are code-owned: the LLM text is slotted in verbatim, and all
paths, links, and layout come from code. These functions are pure — no I/O
and no clock access — so output is a deterministic function of their inputs.

The ``partial`` flag is a parameter passed by orchestration (phase 5b); it is
never computed here.
"""

from __future__ import annotations

import html

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
    funnel_line: str | None = None,
) -> str:
    """Assemble a digest body from already-rendered pieces (fixed template).

    With no items, a genuinely empty window renders ``no_results_body``;
    items that were collected but all lost to errors render
    ``all_failed_body`` instead, so the two situations read differently.

    ``funnel_line`` (reranker funnel count header) is rendered after the
    window line when supplied; ``None`` keeps the v1 layout byte-for-byte.
    """
    lines: list[str] = [heading, _window_line(window)]
    if funnel_line is not None:
        lines.append(funnel_line)
    lines.append("")

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
    funnel_line: str | None = None,
) -> str:
    """Render the Reddit digest markdown.

    Args:
        report_date: Report date string (e.g. "2026-11-09").
        window: The run window (start/end rendered as UTC ISO timestamps).
        items: (thread, summary) pairs in the order they should appear.
        fetch_errors: Per-thread fetch errors, rendered in an Errors section.
        partial: Orchestration-supplied flag; when True a visible partial-run
            line is appended.
        funnel_line: Optional reranker funnel count header (e.g.
            "Ranked 110 threads, kept 22"), rendered after the window line;
            ``None`` (funnel inactive) keeps the v1 output unchanged.

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
        funnel_line=funnel_line,
    )


def render_arxiv_digest(
    report_date: str,
    window: RunWindow,
    items: list[tuple[ArxivPaper, ArxivPaperSummary]],
    screened_out: int,
    fetch_errors: list[str],
    *,
    partial: bool,
    funnel_line: str | None = None,
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
        funnel_line: Optional reranker funnel count header (e.g.
            "Ranked 87 papers, kept 12"), rendered after the window line;
            ``None`` (funnel inactive) keeps the v1 output unchanged.

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
        funnel_line=funnel_line,
    )


# --- HTML variants -----------------------------------------------------------
#
# These mirror the text templates above with the SAME deterministic structure
# and slotting rules; every interpolated string is escaped via html.escape
# because source content is untrusted and must never inject markup.
# Duplication with the text templates is deliberate and preferred over a
# clever shared abstraction.

_HTML_STYLE = """    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #1a1a1a; max-width: 42rem; margin: 0 auto; padding: 1rem; line-height: 1.5; }
      li.meta { color: #555555; }
    </style>"""


def _esc(value: str) -> str:
    """Escape an untrusted string for safe interpolation into HTML."""
    return html.escape(value, quote=True)


def _html_window_line(window: RunWindow) -> str:
    return f'<p class="window">{_esc(_window_line(window))}</p>'


def _html_errors_section(errors: list[str]) -> list[str]:
    lines: list[str] = ["<h2>Errors</h2>", "<ul>"]
    for error in errors:
        lines.append(f"<li>{_esc(error)}</li>")
    lines.append("</ul>")
    return lines


def _html_link(href: str) -> str:
    """Render a link as an anchor whose visible text is the URL itself.

    Both the href attribute and the visible text are escaped.
    """
    safe = _esc(href)
    return f'<a href="{safe}">{safe}</a>'


def _render_html(
    *,
    heading: str,
    window: RunWindow,
    item_blocks: list[str],
    no_results_body: str,
    all_failed_body: str,
    errors: list[str],
    trailing_blocks: list[str],
    funnel_line: str | None = None,
) -> str:
    """Assemble an HTML digest body from already-rendered pieces (fixed template).

    Mirrors :func:`_render` exactly: no-results body when there are no items
    and no errors, all-failed body when items are empty but errors exist,
    an Errors section whenever errors are present, and trailing blocks last.
    ``funnel_line`` mirrors the text renderer's placement after the window
    line; ``None`` keeps the v1 output unchanged.
    """
    lines: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        _HTML_STYLE,
        "</head>",
        "<body>",
        f"<h1>{_esc(heading)}</h1>",
        _html_window_line(window),
    ]
    if funnel_line is not None:
        lines.append(f"<p>{_esc(funnel_line)}</p>")

    if item_blocks:
        lines.extend(item_blocks)
    elif errors:
        lines.append(f"<p>{_esc(all_failed_body)}</p>")
    else:
        lines.append(f"<p>{_esc(no_results_body)}</p>")

    if errors:
        lines.extend(_html_errors_section(errors))

    lines.extend(trailing_blocks)
    lines.append("</body>")
    lines.append("</html>")
    return "\n".join(lines) + "\n"


def _html_partial_paragraph() -> str:
    # Inline style so the highlight survives in mail clients that strip
    # <style> blocks; no external resources anywhere.
    return f'<p style="font-weight: bold; color: #b45309;">{_esc(PARTIAL_LINE)}</p>'


def render_reddit_digest_html(
    report_date: str,
    window: RunWindow,
    items: list[tuple[RedditThread, RedditItemSummary]],
    fetch_errors: list[ThreadFetchError],
    *,
    partial: bool,
    funnel_line: str | None = None,
) -> str:
    """Render the Reddit digest as a self-contained HTML document.

    Mirrors :func:`render_reddit_digest` exactly (same branching, same
    ordering); all interpolated strings are html-escaped.

    Args:
        report_date: Report date string (e.g. "2026-11-09").
        window: The run window (start/end rendered as UTC ISO timestamps).
        items: (thread, summary) pairs in the order they should appear.
        fetch_errors: Per-thread fetch errors, rendered in an Errors section.
        partial: Orchestration-supplied flag; when True a visible partial-run
            paragraph is appended.
        funnel_line: Optional reranker funnel count header, rendered after
            the window paragraph; ``None`` (funnel inactive) keeps the v1
            output unchanged.

    Returns:
        The digest HTML string (no external resources, no images, no JS).
    """
    item_blocks: list[str] = []
    for thread, summary in items:
        author = thread.author if thread.author is not None else "unknown"
        block = [
            f"<h2>{_esc(f'r/{thread.subreddit} — {thread.title}')}</h2>",
            "<ul>",
            f'<li class="meta">Score: {thread.score}</li>',
            f'<li class="meta">Author: {_esc(author)}</li>',
            f'<li class="meta">Link: {_html_link(f"https://reddit.com{thread.permalink}")}</li>',
            f'<li class="meta">{thread.num_comments} comments</li>',
            "</ul>",
            "<h3>Summary:</h3>",
            f"<p>{_esc(summary.summary)}</p>",
            "<h3>Why relevant:</h3>",
            f"<p>{_esc(summary.why_relevant)}</p>",
        ]
        item_blocks.extend(block)

    trailing: list[str] = []
    if partial:
        trailing.append(_html_partial_paragraph())

    return _render_html(
        heading=f"Reddit digest — {report_date}",
        window=window,
        item_blocks=item_blocks,
        no_results_body=REDDIT_NO_RESULTS,
        all_failed_body=REDDIT_ALL_FAILED,
        errors=[f"{error.item}: {error.error}" for error in fetch_errors],
        trailing_blocks=trailing,
        funnel_line=funnel_line,
    )


def render_arxiv_digest_html(
    report_date: str,
    window: RunWindow,
    items: list[tuple[ArxivPaper, ArxivPaperSummary]],
    screened_out: int,
    fetch_errors: list[str],
    *,
    partial: bool,
    funnel_line: str | None = None,
) -> str:
    """Render the arXiv digest as a self-contained HTML document.

    Mirrors :func:`render_arxiv_digest` exactly (same branching, same
    ordering); all interpolated strings are html-escaped.

    Args:
        report_date: Report date string (e.g. "2026-11-09").
        window: The run window (start/end rendered as UTC ISO timestamps).
        items: (paper, summary) pairs in the order they should appear.
        screened_out: Number of papers excluded by relevance screening.
        fetch_errors: Fetch error strings, rendered in an Errors section.
        partial: Orchestration-supplied flag; when True a visible partial-run
            paragraph is appended.
        funnel_line: Optional reranker funnel count header, rendered after
            the window paragraph; ``None`` (funnel inactive) keeps the v1
            output unchanged.

    Returns:
        The digest HTML string (no external resources, no images, no JS).
    """
    item_blocks: list[str] = []
    for paper, summary in items:
        block = [
            f"<h2>{_esc(paper.title)}</h2>",
            "<ul>",
            f'<li class="meta">Authors: {_esc(", ".join(paper.authors))}</li>',
            f'<li class="meta">Categories: {_esc(", ".join(paper.categories))}</li>',
            f'<li class="meta">Link: {_html_link(paper.abs_url)}</li>',
            "</ul>",
            "<h3>Summary:</h3>",
            f"<p>{_esc(summary.summary)}</p>",
            "<h3>Why relevant:</h3>",
            f"<p>{_esc(summary.why_relevant)}</p>",
        ]
        item_blocks.extend(block)

    trailing: list[str] = []
    if screened_out > 0:
        trailing.append(f"<p>{_esc(SCREENED_OUT_LINE.format(count=screened_out))}</p>")
    if partial:
        trailing.append(_html_partial_paragraph())

    return _render_html(
        heading=f"arXiv digest — {report_date}",
        window=window,
        item_blocks=item_blocks,
        no_results_body=ARXIV_NO_RESULTS,
        all_failed_body=ARXIV_ALL_FAILED,
        errors=list(fetch_errors),
        trailing_blocks=trailing,
        funnel_line=funnel_line,
    )
