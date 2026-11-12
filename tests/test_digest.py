"""Tests for deterministic digest rendering."""

from __future__ import annotations

from datetime import date, datetime, timezone

from matomeru.digest import (
    ARXIV_NO_RESULTS,
    PARTIAL_LINE,
    REDDIT_NO_RESULTS,
    SCREENED_OUT_LINE,
    render_arxiv_digest,
    render_arxiv_digest_html,
    render_reddit_digest,
    render_reddit_digest_html,
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


class TestRedditDigestHtml:
    """Tests for render_reddit_digest_html (mirrors the text branching)."""

    def test_happy_path_structure(self) -> None:
        body = render_reddit_digest_html("2026-11-09", WINDOW, [(THREAD, SUMMARY)], [], partial=False)
        assert body.startswith("<!DOCTYPE html>")
        assert '<html lang="en">' in body
        assert "<h1>Reddit digest — 2026-11-09</h1>" in body
        assert "Window (UTC): 2026-11-08T12:00:00+00:00 → 2026-11-09T12:00:00+00:00" in body
        assert "<h2>r/python — Thread title</h2>" in body
        assert "<li" in body and "Score: 100" in body
        assert "Author: alice" in body
        assert '<a href="https://reddit.com/r/python/comments/a/">https://reddit.com/r/python/comments/a/</a>' in body
        assert "4 comments" in body
        assert "<h3>Summary:</h3>" in body
        assert "The thread discusses X." in body
        assert "<h3>Why relevant:</h3>" in body
        assert "Matches the profile&#x27;s interest in X." in body
        assert "<h2>Errors</h2>" not in body

    def test_escapes_llm_slots_and_metadata(self) -> None:
        craft = RedditThread(
            id="t3_evil",
            subreddit="python",
            title="<script>alert('x')</script> & tildes",
            score=1,
            permalink="/r/python/comments/evil/?a=1&b=2",
            num_comments=0,
            author="<b>evil</b>",
        )
        crafted_summary = RedditItemSummary(
            summary="Bad <script>alert(1)</script> & payload",
            why_relevant="Why <em>not</em> & quotes \"double\" 'single'",
        )
        body = render_reddit_digest_html("2026-11-09", WINDOW, [(craft, crafted_summary)], [], partial=False)
        # No raw markup from untrusted content may leak into the document.
        assert "<script>" not in body
        assert "<b>evil</b>" not in body
        assert "&lt;script&gt;" in body
        assert "&amp;" in body
        # Escaped link href and visible text (query-string & escaped too).
        assert (
            '<a href="https://reddit.com/r/python/comments/evil/?a=1&amp;b=2">'
            "https://reddit.com/r/python/comments/evil/?a=1&amp;b=2</a>" in body
        )
        assert "Bad &lt;script&gt;alert(1)&lt;/script&gt; &amp; payload" in body

    def test_no_results_body(self) -> None:
        body = render_reddit_digest_html("2026-11-09", WINDOW, [], [], partial=False)
        assert REDDIT_NO_RESULTS in body
        assert "<h2>r/" not in body
        assert "<h2>Errors</h2>" not in body

    def test_all_failed_body(self) -> None:
        body = render_reddit_digest_html(
            "2026-11-09",
            WINDOW,
            items=[],
            fetch_errors=[ThreadFetchError(index=0, item="t3_abc", error="Connection error.")],
            partial=True,
        )
        assert "No posts matched" not in body
        assert "none could be summarized" in body
        assert "<h2>Errors</h2>" in body
        assert "<li>t3_abc: Connection error.</li>" in body

    def test_errors_section_escapes_entries(self) -> None:
        errors = [ThreadFetchError(index=0, item="<t3_x>", error="boom & gone")]
        body = render_reddit_digest_html("2026-11-09", WINDOW, [(THREAD, SUMMARY)], errors, partial=False)
        assert "<h2>Errors</h2>" in body
        assert "<li>&lt;t3_x&gt;: boom &amp; gone</li>" in body

    def test_partial_paragraph_is_final(self) -> None:
        body = render_reddit_digest_html("2026-11-09", WINDOW, [], [], partial=True)
        assert '<p style="font-weight: bold; color: #b45309;">' in body
        assert PARTIAL_LINE in body
        assert body.rstrip("\n").endswith("</html>")
        before_body_close = body.split("</body>")[0]
        assert before_body_close.rstrip().endswith(f'<p style="font-weight: bold; color: #b45309;">{PARTIAL_LINE}</p>')

    def test_differs_from_text_output(self) -> None:
        text = render_reddit_digest("2026-11-09", WINDOW, [(THREAD, SUMMARY)], [], partial=False)
        html_out = render_reddit_digest_html("2026-11-09", WINDOW, [(THREAD, SUMMARY)], [], partial=False)
        assert html_out != text

    def test_self_contained_no_external_resources(self) -> None:
        body = render_reddit_digest_html("2026-11-09", WINDOW, [(THREAD, SUMMARY)], [], partial=False)
        assert "http://image" not in body
        assert "<script" not in body
        assert "<img" not in body


class TestArxivDigestHtml:
    """Tests for render_arxiv_digest_html (mirrors the text branching)."""

    def test_happy_path_structure(self) -> None:
        body = render_arxiv_digest_html("2026-11-09", WINDOW, [(PAPER, ARXIV_SUMMARY)], 0, [], partial=False)
        assert body.startswith("<!DOCTYPE html>")
        assert "<h1>arXiv digest — 2026-11-09</h1>" in body
        assert "<h2>A Study of Y</h2>" in body
        assert "Authors: Alice, Bob" in body
        assert "Categories: cs.AI, cs.LG" in body
        assert '<a href="https://arxiv.org/abs/2601.00001">https://arxiv.org/abs/2601.00001</a>' in body
        assert "<h3>Summary:</h3>" in body
        assert "The paper studies Y." in body
        assert "<h3>Why relevant:</h3>" in body
        assert "Directly relevant to the profile." in body

    def test_escapes_untrusted_content(self) -> None:
        evil_paper = ArxivPaper(
            arxiv_id="2601.00002",
            abs_url="https://arxiv.org/abs/2601.00002?x=1&y=2",
            pdf_url="https://arxiv.org/pdf/2601.00002",
            title="Evil <script> & Paper",
            abstract="Abstract.",
            authors=["<b>Eve</b>"],
            categories=["cs.AI"],
            primary_category="cs.AI",
            published=datetime(2026, 11, 8, 12, 0, 0, tzinfo=timezone.utc),
            updated=datetime(2026, 11, 8, 12, 0, 0, tzinfo=timezone.utc),
        )
        evil_summary = ArxivPaperSummary(
            summary="Payload <script>alert(1)</script> & more",
            why_relevant="Why & <em>why</em>",
        )
        body = render_arxiv_digest_html("2026-11-09", WINDOW, [(evil_paper, evil_summary)], 0, [], partial=False)
        assert "<script>" not in body
        assert "&lt;script&gt;" in body
        assert "&lt;b&gt;Eve&lt;/b&gt;" in body
        assert 'href="https://arxiv.org/abs/2601.00002?x=1&amp;y=2"' in body

    def test_no_results_body(self) -> None:
        body = render_arxiv_digest_html("2026-11-09", WINDOW, [], 0, [], partial=False)
        assert ARXIV_NO_RESULTS in body
        assert "<h2>Errors</h2>" not in body

    def test_all_failed_body(self) -> None:
        body = render_arxiv_digest_html(
            "2026-11-09",
            WINDOW,
            items=[],
            screened_out=0,
            fetch_errors=["screening failed for 2601.00001: <boom> & gone"],
            partial=True,
        )
        assert "No relevant arXiv papers" not in body
        assert "none could be summarized" in body
        assert "<h2>Errors</h2>" in body
        assert "<li>screening failed for 2601.00001: &lt;boom&gt; &amp; gone</li>" in body

    def test_screened_out_line(self) -> None:
        body = render_arxiv_digest_html("2026-11-09", WINDOW, [], 7, [], partial=False)
        assert f"<p>{SCREENED_OUT_LINE.format(count=7)}</p>" in body

    def test_no_screened_out_line_when_zero(self) -> None:
        body = render_arxiv_digest_html("2026-11-09", WINDOW, [(PAPER, ARXIV_SUMMARY)], 0, [], partial=False)
        assert "screened out" not in body

    def test_partial_after_screened_out(self) -> None:
        body = render_arxiv_digest_html("2026-11-09", WINDOW, [], 3, [], partial=True)
        screened_idx = body.index(SCREENED_OUT_LINE.format(count=3))
        partial_idx = body.index(PARTIAL_LINE)
        assert screened_idx < partial_idx

    def test_differs_from_text_output(self) -> None:
        text = render_arxiv_digest("2026-11-09", WINDOW, [(PAPER, ARXIV_SUMMARY)], 0, [], partial=False)
        html_out = render_arxiv_digest_html("2026-11-09", WINDOW, [(PAPER, ARXIV_SUMMARY)], 0, [], partial=False)
        assert html_out != text


class TestFunnelLine:
    """Optional reranker funnel count header (rendered only when supplied)."""

    FUNNEL_LINE = "Ranked 110 threads, kept 22"

    def test_funnel_line_after_window_before_items(self) -> None:
        body = render_reddit_digest(
            "2026-11-09", WINDOW, [(THREAD, SUMMARY)], [], partial=False, funnel_line=self.FUNNEL_LINE
        )

        assert self.FUNNEL_LINE in body
        assert body.index("Window (UTC):") < body.index(self.FUNNEL_LINE) < body.index("## r/python")

    def test_funnel_line_none_is_byte_for_byte_v1(self) -> None:
        body_without = render_reddit_digest("2026-11-09", WINDOW, [(THREAD, SUMMARY)], [], partial=False)
        body_with = render_reddit_digest(
            "2026-11-09", WINDOW, [(THREAD, SUMMARY)], [], partial=False, funnel_line=self.FUNNEL_LINE
        )

        # Adding the funnel line inserts exactly one line and changes nothing else.
        assert body_with.replace(f"{self.FUNNEL_LINE}\n", "") == body_without
        assert body_without == render_reddit_digest(
            "2026-11-09", WINDOW, [(THREAD, SUMMARY)], [], partial=False, funnel_line=None
        )

    def test_funnel_line_before_no_results_body(self) -> None:
        body = render_reddit_digest("2026-11-09", WINDOW, [], [], partial=False, funnel_line=self.FUNNEL_LINE)

        assert self.FUNNEL_LINE in body
        assert body.index(self.FUNNEL_LINE) < body.index(REDDIT_NO_RESULTS)

    def test_html_funnel_line_after_window_paragraph(self) -> None:
        body = render_reddit_digest_html(
            "2026-11-09", WINDOW, [(THREAD, SUMMARY)], [], partial=False, funnel_line=self.FUNNEL_LINE
        )

        assert f"<p>{self.FUNNEL_LINE}</p>" in body
        assert body.index('<p class="window">') < body.index(f"<p>{self.FUNNEL_LINE}</p>")
        assert body.index(f"<p>{self.FUNNEL_LINE}</p>") < body.index("<h2>")

    def test_html_funnel_line_none_is_byte_for_byte_v1(self) -> None:
        body_without = render_reddit_digest_html("2026-11-09", WINDOW, [(THREAD, SUMMARY)], [], partial=False)
        body_with = render_reddit_digest_html(
            "2026-11-09", WINDOW, [(THREAD, SUMMARY)], [], partial=False, funnel_line=self.FUNNEL_LINE
        )

        assert body_with.replace(f"<p>{self.FUNNEL_LINE}</p>\n", "") == body_without

    def test_html_funnel_line_is_escaped(self) -> None:
        body = render_reddit_digest_html(
            "2026-11-09", WINDOW, [(THREAD, SUMMARY)], [], partial=False, funnel_line="Ranked <b>5</b> threads"
        )

        assert "<b>5</b>" not in body
        assert "Ranked &lt;b&gt;5&lt;/b&gt; threads" in body


class TestArxivFunnelLine:
    """Optional reranker funnel count header for the arXiv renderers."""

    FUNNEL_LINE = "Ranked 87 papers, kept 12"

    def test_funnel_line_after_window_before_items(self) -> None:
        body = render_arxiv_digest(
            "2026-11-09", WINDOW, [(PAPER, ARXIV_SUMMARY)], 0, [], partial=False, funnel_line=self.FUNNEL_LINE
        )

        assert self.FUNNEL_LINE in body
        assert body.index("Window (UTC):") < body.index(self.FUNNEL_LINE) < body.index("## A Study of Y")

    def test_funnel_line_none_is_byte_for_byte_v1(self) -> None:
        body_without = render_arxiv_digest("2026-11-09", WINDOW, [(PAPER, ARXIV_SUMMARY)], 0, [], partial=False)
        body_with = render_arxiv_digest(
            "2026-11-09", WINDOW, [(PAPER, ARXIV_SUMMARY)], 0, [], partial=False, funnel_line=self.FUNNEL_LINE
        )

        # Adding the funnel line inserts exactly one line and changes nothing else.
        assert body_with.replace(f"{self.FUNNEL_LINE}\n", "") == body_without
        assert body_without == render_arxiv_digest(
            "2026-11-09", WINDOW, [(PAPER, ARXIV_SUMMARY)], 0, [], partial=False, funnel_line=None
        )

    def test_funnel_line_before_no_results_body(self) -> None:
        body = render_arxiv_digest("2026-11-09", WINDOW, [], 0, [], partial=False, funnel_line=self.FUNNEL_LINE)

        assert self.FUNNEL_LINE in body
        assert body.index(self.FUNNEL_LINE) < body.index(ARXIV_NO_RESULTS)

    def test_html_funnel_line_after_window_paragraph(self) -> None:
        body = render_arxiv_digest_html(
            "2026-11-09", WINDOW, [(PAPER, ARXIV_SUMMARY)], 0, [], partial=False, funnel_line=self.FUNNEL_LINE
        )

        assert f"<p>{self.FUNNEL_LINE}</p>" in body
        assert body.index('<p class="window">') < body.index(f"<p>{self.FUNNEL_LINE}</p>")
        assert body.index(f"<p>{self.FUNNEL_LINE}</p>") < body.index("<h2>")

    def test_html_funnel_line_none_is_byte_for_byte_v1(self) -> None:
        body_without = render_arxiv_digest_html("2026-11-09", WINDOW, [(PAPER, ARXIV_SUMMARY)], 0, [], partial=False)
        body_with = render_arxiv_digest_html(
            "2026-11-09", WINDOW, [(PAPER, ARXIV_SUMMARY)], 0, [], partial=False, funnel_line=self.FUNNEL_LINE
        )

        assert body_with.replace(f"<p>{self.FUNNEL_LINE}</p>\n", "") == body_without

    def test_html_funnel_line_is_escaped(self) -> None:
        body = render_arxiv_digest_html(
            "2026-11-09", WINDOW, [(PAPER, ARXIV_SUMMARY)], 0, [], partial=False, funnel_line="Ranked <b>5</b> papers"
        )

        assert "<b>5</b>" not in body
        assert "Ranked &lt;b&gt;5&lt;/b&gt; papers" in body
