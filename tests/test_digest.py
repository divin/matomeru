"""Tests for deterministic digest rendering."""

from __future__ import annotations

from datetime import date, datetime, timezone

from matomeru.digest import (
    ARXIV_NO_RESULTS,
    PARTIAL_LINE,
    REDDIT_NO_RESULTS,
    SCREENED_OUT_LINE,
    render_arxiv_digest,
    render_reddit_digest,
)
from matomeru.llm import ArxivPaperSummary, RedditItemSummary
from matomeru.models import ArxivPaper, RedditThread, ThreadFetchError
from matomeru.window import RunWindow

WINDOW = RunWindow(
    start_utc=datetime(2026, 11, 8, 12, 0, 0, tzinfo=timezone.utc),
    end_utc=datetime(2026, 11, 9, 12, 0, 0, tzinfo=timezone.utc),
    report_date=date(2026, 11, 9),
    tz_name="Europe/Berlin",
)

THREAD = RedditThread(
    id="t3_a",
    subreddit="python",
    title="Thread title",
    score=100,
    permalink="/r/python/comments/a/",
    num_comments=4,
    author="alice",
)

SUMMARY = RedditItemSummary(
    summary="The thread discusses X.",
    why_relevant="Matches the profile's interest in X.",
)

PAPER = ArxivPaper(
    arxiv_id="2601.00001",
    abs_url="https://arxiv.org/abs/2601.00001",
    pdf_url="https://arxiv.org/pdf/2601.00001",
    title="A Study of Y",
    abstract="Abstract text.",
    authors=["Alice", "Bob"],
    categories=["cs.AI", "cs.LG"],
    primary_category="cs.AI",
    published=datetime(2026, 11, 8, 12, 0, 0, tzinfo=timezone.utc),
    updated=datetime(2026, 11, 8, 12, 0, 0, tzinfo=timezone.utc),
)

ARXIV_SUMMARY = ArxivPaperSummary(
    summary="The paper studies Y.",
    why_relevant="Directly relevant to the profile.",
)


class TestRedditDigest:
    """Tests for render_reddit_digest."""

    def test_first_line_and_window(self) -> None:
        body = render_reddit_digest("2026-11-09", WINDOW, [], [], partial=False)
        lines = body.splitlines()
        assert lines[0] == "# Reddit digest — 2026-11-09"
        assert lines[1] == "Window (UTC): 2026-11-08T12:00:00+00:00 → 2026-11-09T12:00:00+00:00"

    def test_item_block_order_and_metadata(self) -> None:
        body = render_reddit_digest("2026-11-09", WINDOW, [(THREAD, SUMMARY)], [], partial=False)
        assert "## r/python — Thread title" in body
        assert "- Score: 100" in body
        assert "- Author: alice" in body
        assert "- Link: https://reddit.com/r/python/comments/a/" in body
        assert "- 4 comments" in body
        assert "Summary:" in body
        assert "The thread discusses X." in body
        assert "Why relevant:" in body
        assert "Matches the profile's interest in X." in body

    def test_items_rendered_in_given_order(self) -> None:
        other = RedditThread(id="t3_b", subreddit="rust", title="Other", permalink="/r/rust/comments/b/")
        body = render_reddit_digest("2026-11-09", WINDOW, [(THREAD, SUMMARY), (other, SUMMARY)], [], partial=False)
        assert body.index("## r/python — Thread title") < body.index("## r/rust — Other")

    def test_llm_slots_verbatim(self) -> None:
        weird = RedditItemSummary(summary="Line1\nLine2  spaced  |code|", why_relevant="trailing spaces   ")
        body = render_reddit_digest("2026-11-09", WINDOW, [(THREAD, weird)], [], partial=False)
        assert "Line1\nLine2  spaced  |code|" in body
        assert "trailing spaces   " in body

    def test_no_results_body(self) -> None:
        body = render_reddit_digest("2026-11-09", WINDOW, [], [], partial=False)
        assert REDDIT_NO_RESULTS in body
        assert "## r/" not in body

    def test_errors_section(self) -> None:
        errors = [
            ThreadFetchError(index=0, item="t3_x", error="boom"),
            ThreadFetchError(index=1, item="t3_y", error="gone"),
        ]
        body = render_reddit_digest("2026-11-09", WINDOW, [(THREAD, SUMMARY)], errors, partial=False)
        errors_idx = body.index("## Errors")
        assert "t3_x: boom" in body[errors_idx:]
        assert "t3_y: gone" in body[errors_idx:]

    def test_partial_line_is_final(self) -> None:
        body = render_reddit_digest("2026-11-09", WINDOW, [], [], partial=True)
        assert body.splitlines()[-1] == PARTIAL_LINE


class TestArxivDigest:
    """Tests for render_arxiv_digest."""

    def test_first_line_and_item_block(self) -> None:
        body = render_arxiv_digest("2026-11-09", WINDOW, [(PAPER, ARXIV_SUMMARY)], 0, [], partial=False)
        lines = body.splitlines()
        assert lines[0] == "# arXiv digest — 2026-11-09"
        assert lines[1].startswith("Window (UTC): ")
        assert "## A Study of Y" in body
        assert "- Authors: Alice, Bob" in body
        assert "- Categories: cs.AI, cs.LG" in body
        assert "- Link: https://arxiv.org/abs/2601.00001" in body
        assert "The paper studies Y." in body
        assert "Directly relevant to the profile." in body

    def test_no_results_body(self) -> None:
        body = render_arxiv_digest("2026-11-09", WINDOW, [], 0, [], partial=False)
        assert ARXIV_NO_RESULTS in body
        assert "## Errors" not in body

    def test_screened_out_line(self) -> None:
        body = render_arxiv_digest("2026-11-09", WINDOW, [], 7, [], partial=False)
        assert SCREENED_OUT_LINE.format(count=7) in body
        assert SCREENED_OUT_LINE.format(count=7) == "7 further paper(s) were screened out as not relevant."

    def test_no_screened_out_line_when_zero(self) -> None:
        body = render_arxiv_digest("2026-11-09", WINDOW, [(PAPER, ARXIV_SUMMARY)], 0, [], partial=False)
        assert "screened out" not in body

    def test_errors_section(self) -> None:
        body = render_arxiv_digest("2026-11-09", WINDOW, [], 0, ["fetch failed for cs.AI"], partial=False)
        assert "## Errors" in body
        assert "fetch failed for cs.AI" in body

    def test_partial_line_is_final(self) -> None:
        body = render_arxiv_digest("2026-11-09", WINDOW, [], 0, [], partial=True)
        assert body.splitlines()[-1] == PARTIAL_LINE

    def test_partial_after_screened_out(self) -> None:
        body = render_arxiv_digest("2026-11-09", WINDOW, [], 3, [], partial=True)
        lines = body.splitlines()
        assert lines[-1] == PARTIAL_LINE
        assert lines[-2] == SCREENED_OUT_LINE.format(count=3)


class TestAllFailedBody:
    """Empty items with errors must read differently from a quiet day."""

    def test_reddit_all_failed_shows_all_failed_body(self) -> None:
        body = render_reddit_digest(
            "2026-11-09",
            WINDOW,
            items=[],
            fetch_errors=[ThreadFetchError(index=0, item="t3_abc", error="Connection error.")],
            partial=True,
        )
        assert "No posts matched" not in body
        assert "none could be summarized" in body
        assert "## Errors" in body
        assert "Partial run" in body

    def test_reddit_quiet_day_keeps_no_results(self) -> None:
        body = render_reddit_digest("2026-11-09", WINDOW, items=[], fetch_errors=[], partial=False)
        assert "No posts matched" in body
        assert "## Errors" not in body

    def test_arxiv_all_failed_shows_all_failed_body(self) -> None:
        body = render_arxiv_digest(
            "2026-11-09",
            WINDOW,
            items=[],
            screened_out=0,
            fetch_errors=["screening failed for 2601.00001: boom"],
            partial=True,
        )
        assert "No relevant arXiv papers" not in body
        assert "none could be summarized" in body
        assert "## Errors" in body

    def test_arxiv_quiet_day_keeps_no_results(self) -> None:
        body = render_arxiv_digest("2026-11-09", WINDOW, items=[], screened_out=0, fetch_errors=[], partial=False)
        assert "No relevant arXiv papers" in body
        assert "## Errors" not in body
