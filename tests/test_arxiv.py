"""Tests for arXiv announcement collection via RSS (no network; httpx.MockTransport)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx
import pytest

from matomeru.arxiv import _base_id, _normalize_category, collect_arxiv
from matomeru.config import ConfigError
from matomeru.window import RunWindow

UTC = timezone.utc

# Canned RSS feeds modeled EXACTLY on the live https://rss.arxiv.org/rss/<cat>
# shape: channel/item in NO namespace, arxiv:announce_type in the
# http://arxiv.org/schemas/atom namespace, dc:creator in
# http://purl.org/dc/elements/1.1/, RFC 2822 pubDate stamps.
#
# cs.AI feed contains:
#   item A: new announcement, multi-line abstract, two categories,
#           dc:creator with comma-separated names
#   item B: replacement announcement (announce_type=replace)
#   item C: cross-list announcement (announce_type=cross), single category
#   item D: older announcement stamp entirely outside the window
#           (what a run shortly after two announcements sees)
#   item E: malformed (no link, no pubDate) -> skipped with a WARNING
AI_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0">
  <channel>
    <title>cs.AI updates on arXiv.org</title>
    <pubDate>Mon, 09 Nov 2026 00:00:00 -0500</pubDate>
  </channel>
  <item>
    <title>Agents That Plan Ahead: A Survey</title>
    <link>https://arxiv.org/abs/2611.08572</link>
    <description>arXiv:2611.08572v1 Announce Type: new
Abstract: We survey agents that plan ahead.
Emergent behavior emerges when
multi-agent coordination is present.</description>
    <guid isPermaLink="false">oai:arXiv.org:2611.08572v1</guid>
    <category>cs.AI</category><category>cs.CY</category>
    <pubDate>Mon, 09 Nov 2026 00:00:00 -0500</pubDate>
    <arxiv:announce_type>new</arxiv:announce_type>
    <dc:creator>Tanat Tonguthaisri, Alice Author</dc:creator>
  </item>
  <item>
    <title>Revised Benchmark for Robotic Grasping</title>
    <link>https://arxiv.org/abs/2611.07000</link>
    <description>arXiv:2611.07000v2 Announce Type: replace
Abstract: A revised benchmark for robotic grasping with better metrics.</description>
    <guid isPermaLink="false">oai:arXiv.org:2611.07000v2</guid>
    <category>cs.AI</category>
    <pubDate>Mon, 09 Nov 2026 00:00:00 -0500</pubDate>
    <arxiv:announce_type>replace</arxiv:announce_type>
    <dc:creator>Carol Compiler</dc:creator>
  </item>
  <item>
    <title>Cross-Listed Study of Neural Nets</title>
    <link>https://arxiv.org/abs/2611.06000</link>
    <description>arXiv:2611.06000v1 Announce Type: cross
Abstract: A cross-listed study of neural network training dynamics.</description>
    <guid isPermaLink="false">oai:arXiv.org:2611.06000v1</guid>
    <category>cs.AI</category>
    <pubDate>Sun, 08 Nov 2026 20:00:00 -0500</pubDate>
    <arxiv:announce_type>cross</arxiv:announce_type>
    <dc:creator>Dave Digger</dc:creator>
  </item>
  <item>
    <title>An Older Announcement About Dusty Debris Disks</title>
    <link>https://arxiv.org/abs/2610.12345</link>
    <description>arXiv:2610.12345v3 Announce Type: replace
Abstract: This announcement stamp is outside the 24-hour window.</description>
    <guid isPermaLink="false">oai:arXiv.org:2610.12345v3</guid>
    <category>cs.AI</category>
    <pubDate>Fri, 06 Nov 2026 20:00:00 -0500</pubDate>
    <arxiv:announce_type>replace</arxiv:announce_type>
    <dc:creator>Eve Eavesdropper</dc:creator>
  </item>
  <item>
    <title>A Malformed Item Without Link or pubDate</title>
    <guid isPermaLink="false">oai:arXiv.org:2611.09999v1</guid>
    <description>arXiv:2611.09999v1 Announce Type: new
Abstract: This item should be skipped with a warning.</description>
    <category>cs.AI</category>
  </item>
</rss>
"""

# cs.LG feed: a duplicate of A's base id (different version suffix) plus one
# distinct versionless-guid paper. Used to exercise cross-category dedupe
# (first occurrence wins) and the versionless pdf URL fallback.
LG_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0">
  <channel>
    <title>cs.LG updates on arXiv.org</title>
    <pubDate>Mon, 09 Nov 2026 00:00:00 -0500</pubDate>
  </channel>
  <item>
    <title>Duplicate Across Categories</title>
    <link>https://arxiv.org/abs/2611.08572</link>
    <description>arXiv:2611.08572v9 Announce Type: cross
Abstract: Same base id as the cs.AI item, different version.</description>
    <guid isPermaLink="false">oai:arXiv.org:2611.08572v9</guid>
    <category>cs.LG</category>
    <pubDate>Mon, 09 Nov 2026 00:00:00 -0500</pubDate>
    <arxiv:announce_type>cross</arxiv:announce_type>
    <dc:creator>Franca Duplicate</dc:creator>
  </item>
  <item>
    <title>Scaling Laws for Tiny Models</title>
    <link>https://arxiv.org/abs/2611.08000</link>
    <description>arXiv:2611.08000 Announce Type: new
Abstract: Scaling laws for tiny language models on edge devices.</description>
    <guid isPermaLink="false">oai:arXiv.org:2611.08000</guid>
    <category>cs.LG</category>
    <pubDate>Mon, 09 Nov 2026 00:00:00 -0500</pubDate>
  </item>
</rss>
"""

# Boundary feed: items stamped exactly at the window start and window end
# (both must be dropped; strict inequality), plus one stamp in between.
BOUNDARY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0">
  <channel>
    <title>cs.AI updates on arXiv.org</title>
    <pubDate>Mon, 09 Nov 2026 00:00:00 -0500</pubDate>
  </channel>
  <item>
    <title>Exactly At Window Start</title>
    <link>https://arxiv.org/abs/2611.00001</link>
    <description>arXiv:2611.00001v1 Announce Type: new
Abstract: Announcement stamp equals window.start_utc.</description>
    <guid isPermaLink="false">oai:arXiv.org:2611.00001v1</guid>
    <category>cs.AI</category>
    <pubDate>Sun, 08 Nov 2026 07:00:00 -0500</pubDate>
    <arxiv:announce_type>new</arxiv:announce_type>
    <dc:creator>Start Stample</dc:creator>
  </item>
  <item>
    <title>Just After Window Start</title>
    <link>https://arxiv.org/abs/2611.00002</link>
    <description>arXiv:2611.00002v1 Announce Type: new
Abstract: Announcement stamp inside the window.</description>
    <guid isPermaLink="false">oai:arXiv.org:2611.00002v1</guid>
    <category>cs.AI</category>
    <pubDate>Sun, 08 Nov 2026 12:00:00 -0500</pubDate>
    <arxiv:announce_type>new</arxiv:announce_type>
    <dc:creator>In Inside</dc:creator>
  </item>
  <item>
    <title>Exactly At Window End</title>
    <link>https://arxiv.org/abs/2611.00003</link>
    <description>arXiv:2611.00003v1 Announce Type: new
Abstract: Announcement stamp equals window.end_utc.</description>
    <guid isPermaLink="false">oai:arXiv.org:2611.00003v1</guid>
    <category>cs.AI</category>
    <pubDate>Mon, 09 Nov 2026 07:00:00 -0500</pubDate>
    <arxiv:announce_type>new</arxiv:announce_type>
    <dc:creator>End Stample</dc:creator>
  </item>
</rss>
"""


# Per-category selector fixtures: the cs.AI feed holds a "quantum" paper (X),
# an unrelated paper (Z), and a cross-listed paper (Y) whose base id also
# appears in the cs.LG feed. Used to exercise per-feed selector filtering
# running BEFORE cross-category dedupe.
CT_AI_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0">
  <channel>
    <title>cs.AI updates on arXiv.org</title>
    <pubDate>Mon, 09 Nov 2026 00:00:00 -0500</pubDate>
  </channel>
  <item>
    <title>Quantum Widget Calibration</title>
    <link>https://arxiv.org/abs/2611.20001</link>
    <description>arXiv:2611.20001v1 Announce Type: new
Abstract: Quantum widget calibration protocols for noisy devices.</description>
    <guid isPermaLink="false">oai:arXiv.org:2611.20001v1</guid>
    <category>cs.AI</category>
    <pubDate>Mon, 09 Nov 2026 00:00:00 -0500</pubDate>
    <arxiv:announce_type>new</arxiv:announce_type>
    <dc:creator>Quinn Quantum</dc:creator>
  </item>
  <item>
    <title>Unrelated Notes About Pottery</title>
    <link>https://arxiv.org/abs/2611.20002</link>
    <description>arXiv:2611.20002v1 Announce Type: new
Abstract: Kiln temperature curves and glaze chemistry.</description>
    <guid isPermaLink="false">oai:arXiv.org:2611.20002v1</guid>
    <category>cs.AI</category>
    <pubDate>Mon, 09 Nov 2026 00:00:00 -0500</pubDate>
    <arxiv:announce_type>new</arxiv:announce_type>
    <dc:creator>Paula Pottery</dc:creator>
  </item>
  <item>
    <title>Cross-Listed Shared Paper</title>
    <link>https://arxiv.org/abs/2611.20003</link>
    <description>arXiv:2611.20003v1 Announce Type: cross
Abstract: A paper cross-listed into both category feeds.</description>
    <guid isPermaLink="false">oai:arXiv.org:2611.20003v1</guid>
    <category>cs.AI</category>
    <pubDate>Mon, 09 Nov 2026 00:00:00 -0500</pubDate>
    <arxiv:announce_type>cross</arxiv:announce_type>
    <dc:creator>Cross Poster</dc:creator>
  </item>
</rss>
"""

CT_LG_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0">
  <channel>
    <title>cs.LG updates on arXiv.org</title>
    <pubDate>Mon, 09 Nov 2026 00:00:00 -0500</pubDate>
  </channel>
  <item>
    <title>Cross-Listed Shared Paper</title>
    <link>https://arxiv.org/abs/2611.20003</link>
    <description>arXiv:2611.20003v2 Announce Type: cross
Abstract: A paper cross-listed into both category feeds.</description>
    <guid isPermaLink="false">oai:arXiv.org:2611.20003v2</guid>
    <category>cs.LG</category>
    <pubDate>Mon, 09 Nov 2026 00:00:00 -0500</pubDate>
    <arxiv:announce_type>cross</arxiv:announce_type>
    <dc:creator>Cross Poster</dc:creator>
  </item>
</rss>
"""


def make_window() -> RunWindow:
    """Fixed 24-hour window: 2026-11-08 12:00 UTC to 2026-11-09 12:00 UTC."""
    return RunWindow(
        start_utc=datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC),
        end_utc=datetime(2026, 11, 9, 12, 0, 0, tzinfo=UTC),
        report_date=datetime(2026, 11, 9).date(),
        tz_name="UTC",
    )


def make_client(
    captured: list[httpx.Request],
    feeds: dict[str, str | int],
) -> httpx.AsyncClient:
    """Async client backed by a MockTransport that records requests.

    feeds maps URL paths (e.g. "/rss/cs.AI") to either an int status code
    or an XML body; unlisted paths get a 404.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        spec = feeds.get(request.url.path)
        if spec is None:
            return httpx.Response(404)
        if isinstance(spec, int):
            return httpx.Response(spec)
        return httpx.Response(200, content=spec.encode("utf-8"), headers={"Content-Type": "application/rss+xml"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def standard_feeds() -> dict[str, str | int]:
    """The default two-category feed mapping used by most tests."""
    return {"/rss/cs.AI": AI_FEED, "/rss/cs.LG": LG_FEED}


class TestCategoryValidation:
    """Tests for category normalization and validation."""

    def test_strips_whitespace(self) -> None:
        assert _normalize_category("  cs.AI  ") == "cs.AI"

    def test_strips_cat_prefix(self) -> None:
        assert _normalize_category("cat:cs.AI") == "cs.AI"

    def test_rejects_spaces(self) -> None:
        with pytest.raises(ConfigError, match="spaces and quotes"):
            _normalize_category("cs LG")

    def test_rejects_quotes(self) -> None:
        with pytest.raises(ConfigError, match="spaces and quotes"):
            _normalize_category('cs.AI"')

    def test_rejects_empty(self) -> None:
        with pytest.raises(ConfigError, match="invalid arXiv category"):
            _normalize_category("   ")


class TestBaseId:
    """Tests for version-suffix stripping used in dedupe."""

    def test_strips_version_after_dot(self) -> None:
        assert _base_id("2611.08572v1") == "2611.08572"

    def test_strips_multidigit_version(self) -> None:
        assert _base_id("2611.08572v12") == "2611.08572"

    def test_keeps_id_without_version(self) -> None:
        assert _base_id("2611.08572") == "2611.08572"


class TestCollectArxiv:
    """Tests for collect_arxiv against canned RSS announcement feeds."""

    @pytest.mark.asyncio
    async def test_per_category_urls_and_ua_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, standard_feeds()) as http:
            await collect_arxiv(http, ["cs.AI", "cs.LG"], [], make_window())

        assert len(captured) == 2
        assert [r.url.path for r in captured] == ["/rss/cs.AI", "/rss/cs.LG"]
        for request in captured:
            assert request.url.host == "rss.arxiv.org"
            assert request.headers["User-Agent"].startswith("matomeru/1.1.0")
            assert "(self-hosted digest)" in request.headers["User-Agent"]

    @pytest.mark.asyncio
    async def test_category_normalization_in_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, {"/rss/cs.AI": AI_FEED}) as http:
            await collect_arxiv(http, ["  cs.AI  ", "cat:cs.LG"], [], make_window())

        assert [r.url.path for r in captured] == ["/rss/cs.AI", "/rss/cs.LG"]

    @pytest.mark.asyncio
    async def test_rss_base_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.setenv("ARXIV_RSS_BASE", "http://mirror.example.org")

        async with make_client(captured, {"/rss/cs.AI": AI_FEED}) as http:
            await collect_arxiv(http, ["cs.AI"], [], make_window())

        assert captured[0].url.host == "mirror.example.org"

    @pytest.mark.asyncio
    async def test_rss_base_empty_env_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.setenv("ARXIV_RSS_BASE", "")

        async with make_client(captured, {"/rss/cs.AI": AI_FEED}) as http:
            await collect_arxiv(http, ["cs.AI"], [], make_window())

        assert captured[0].url.host == "rss.arxiv.org"

    @pytest.mark.asyncio
    async def test_empty_categories_raises_config_error(self) -> None:
        captured: list[httpx.Request] = []
        async with make_client(captured, {}) as http:
            with pytest.raises(ConfigError, match="arXiv collection needs at least one category"):
                await collect_arxiv(http, [], [], make_window())
        assert captured == []  # no request made

    @pytest.mark.asyncio
    async def test_empty_categories_raises_even_with_search_terms(self) -> None:
        """RSS is per-category, so search terms alone cannot retrieve."""
        captured: list[httpx.Request] = []
        async with make_client(captured, {}) as http:
            with pytest.raises(ConfigError, match="arXiv collection needs at least one category"):
                await collect_arxiv(http, [], ["agents"], make_window())
        assert captured == []  # no request made

    @pytest.mark.asyncio
    async def test_invalid_category_rejected_before_any_request(self) -> None:
        captured: list[httpx.Request] = []
        async with make_client(captured, {}) as http:
            with pytest.raises(ConfigError, match="spaces and quotes"):
                await collect_arxiv(http, ["cs.AI", "cs LG"], [], make_window())
        assert captured == []  # validation fails fast, no partial fetch

    @pytest.mark.asyncio
    async def test_parsing_all_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, standard_feeds()) as http:
            papers, _ = await collect_arxiv(http, ["cs.AI", "cs.LG"], [], make_window())

        fresh = papers[0]
        assert fresh.arxiv_id == "2611.08572v1"
        assert fresh.abs_url == "https://arxiv.org/abs/2611.08572"
        # PDF URL uses the versioned id from the guid
        assert fresh.pdf_url == "https://arxiv.org/pdf/2611.08572v1"
        assert fresh.title == "Agents That Plan Ahead: A Survey"
        # Multi-line abstract extracted after the "Abstract:" marker, with the
        # "arXiv:... Announce Type: ..." prefix removed and whitespace collapsed
        assert fresh.abstract == (
            "We survey agents that plan ahead. Emergent behavior emerges when multi-agent coordination is present."
        )
        # Comma-separated dc:creator is split and trimmed
        assert fresh.authors == ["Tanat Tonguthaisri", "Alice Author"]
        assert fresh.categories == ["cs.AI", "cs.CY"]
        # primary_category is the requested category of the feed
        assert fresh.primary_category == "cs.AI"
        # RFC 2822 stamp "Mon, 09 Nov 2026 00:00:00 -0500" -> 05:00Z Nov 9
        assert fresh.published == datetime(2026, 11, 9, 5, 0, 0, tzinfo=UTC)
        assert fresh.updated == datetime(2026, 11, 9, 5, 0, 0, tzinfo=UTC)
        assert fresh.published.tzinfo is UTC
        assert fresh.announcement_hint == "announce_type=new"

        cross = papers[3]
        assert cross.arxiv_id == "2611.06000v1"
        assert cross.announcement_hint == "announce_type=cross"
        assert cross.primary_category == "cs.AI"
        assert cross.published == datetime(2026, 11, 9, 1, 0, 0, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_replace_announce_type_and_versionless_guid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, standard_feeds()) as http:
            papers, _ = await collect_arxiv(http, ["cs.AI", "cs.LG"], [], make_window())

        replaced = papers[1]
        assert replaced.arxiv_id == "2611.07000v2"
        assert replaced.announcement_hint == "announce_type=replace"
        assert replaced.pdf_url == "https://arxiv.org/pdf/2611.07000v2"

        # Versionless guid: PDF URL falls back to the versionless link id,
        # and a missing dc:creator yields no authors
        versionless = papers[2]
        assert versionless.arxiv_id == "2611.08000"
        assert versionless.pdf_url == "https://arxiv.org/pdf/2611.08000"
        assert versionless.authors == []
        assert versionless.announcement_hint is None
        assert versionless.primary_category == "cs.LG"

    @pytest.mark.asyncio
    async def test_dedupe_across_categories_first_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, standard_feeds()) as http:
            papers, fetched = await collect_arxiv(http, ["cs.AI", "cs.LG"], [], make_window())

        assert fetched == 7  # 5 items in cs.AI + 2 items in cs.LG
        ids = [p.arxiv_id for p in papers]
        # The cs.AI occurrence (2611.08572v1) wins over the cs.LG duplicate
        assert ids == ["2611.08572v1", "2611.07000v2", "2611.08000", "2611.06000v1"]

    @pytest.mark.asyncio
    async def test_search_terms_substring_prefilter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, standard_feeds()) as http:
            papers, _ = await collect_arxiv(http, ["cs.AI"], ["robotic grasping"], make_window())

        assert [p.arxiv_id for p in papers] == ["2611.07000v2"]

    @pytest.mark.asyncio
    async def test_search_terms_match_abstract_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, standard_feeds()) as http:
            papers, _ = await collect_arxiv(http, ["cs.AI"], ["EMERGES WHEN"], make_window())

        # Matches A's abstract across the original line break (case-insensitive)
        assert [p.arxiv_id for p in papers] == ["2611.08572v1"]

    @pytest.mark.asyncio
    async def test_search_terms_any_term_matches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, standard_feeds()) as http:
            papers, _ = await collect_arxiv(http, ["cs.AI"], ["neural nets", "totally-unrelated-term"], make_window())

        assert [p.arxiv_id for p in papers] == ["2611.06000v1"]

    @pytest.mark.asyncio
    async def test_search_terms_quotes_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, standard_feeds()) as http:
            papers, _ = await collect_arxiv(http, ["cs.AI"], ['grasp"ing'], make_window())

        assert [p.arxiv_id for p in papers] == ["2611.07000v2"]

    @pytest.mark.asyncio
    async def test_search_terms_no_match_yields_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, standard_feeds()) as http:
            papers, fetched = await collect_arxiv(http, ["cs.AI"], ["no-such-term-anywhere"], make_window())

        assert fetched == 5  # fetch unaffected
        assert papers == []

    @pytest.mark.asyncio
    async def test_category_terms_per_feed_filter_before_dedupe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A cross-listed paper is kept when ANY feed's selector matches it.

        cs.AI has explicit terms that only its own quantum paper matches;
        cs.LG has an explicitly empty list (keep all). The shared paper Y
        does not match cs.AI's terms, so it is dropped from that feed BEFORE
        dedupe and survives via the cs.LG keep-all feed exactly once.
        """
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, {"/rss/cs.AI": CT_AI_FEED, "/rss/cs.LG": CT_LG_FEED}) as http:
            papers, fetched = await collect_arxiv(
                http,
                ["cs.AI", "cs.LG"],
                [],
                make_window(),
                category_terms={"cs.AI": ["quantum"], "cs.LG": []},
            )

        assert fetched == 4  # 3 items in cs.AI + 1 item in cs.LG
        # X kept via cs.AI's selector; Z dropped; Y kept once via cs.LG's
        # keep-all (its cs.AI occurrence was filtered before dedupe)
        assert [p.arxiv_id for p in papers] == ["2611.20001v1", "2611.20003v2"]

    @pytest.mark.asyncio
    async def test_category_terms_absent_key_falls_back_to_global(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A category without a category_terms entry uses the global terms."""
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, standard_feeds()) as http:
            papers, _ = await collect_arxiv(
                http,
                ["cs.AI"],
                ["robotic grasping"],
                make_window(),
                category_terms={"cs.LG": ["neural"]},
            )

        # cs.AI is absent from category_terms -> global search_terms apply
        assert [p.arxiv_id for p in papers] == ["2611.07000v2"]

    @pytest.mark.asyncio
    async def test_category_terms_exact_raw_key_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An entry whose key exactly matches the raw category is the selector."""
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, {"/rss/cs.AI": AI_FEED}) as http:
            papers, _ = await collect_arxiv(
                http,
                ["cs.AI"],
                [],
                make_window(),
                category_terms={"cs.AI": ["robotic grasping"]},
            )

        assert [p.arxiv_id for p in papers] == ["2611.07000v2"]

    @pytest.mark.asyncio
    async def test_category_terms_empty_list_keeps_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicitly empty list means "keep all papers in this category"."""
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, {"/rss/cs.AI": AI_FEED}) as http:
            papers, _ = await collect_arxiv(
                http,
                ["cs.AI"],
                ["robotic grasping"],
                make_window(),
                category_terms={"cs.AI": []},
            )

        # The explicit empty list replaces the global terms entirely
        assert len(papers) == 3
        assert {p.arxiv_id for p in papers} == {"2611.08572v1", "2611.07000v2", "2611.06000v1"}

    @pytest.mark.asyncio
    async def test_window_filter_on_announcement_stamps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, standard_feeds()) as http:
            papers, fetched = await collect_arxiv(http, ["cs.AI"], [], make_window())

        assert fetched == 5
        ids = [p.arxiv_id for p in papers]
        # D has an older announcement stamp (missed-day scenario) -> dropped
        assert "2610.12345v3" not in ids
        # Malformed item E never appears
        assert "2611.09999v1" not in ids

    @pytest.mark.asyncio
    async def test_window_boundaries_are_exclusive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stamp exactly at start_utc or end_utc is not kept."""
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, {"/rss/cs.AI": BOUNDARY_FEED}) as http:
            papers, fetched = await collect_arxiv(http, ["cs.AI"], [], make_window())

        assert fetched == 3
        # Only the strictly in-between stamp survives
        assert [p.arxiv_id for p in papers] == ["2611.00002v1"]

    @pytest.mark.asyncio
    async def test_max_results_truncation_after_sorting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, standard_feeds()) as http:
            papers, _ = await collect_arxiv(http, ["cs.AI", "cs.LG"], [], make_window(), max_results=2)

        # Sorted by announcement stamp desc; A/B/F share 05:00Z (first-seen
        # order kept) and C is 01:00Z, so truncation keeps A and B
        assert [p.arxiv_id for p in papers] == ["2611.08572v1", "2611.07000v2"]

    @pytest.mark.asyncio
    async def test_malformed_item_skipped_with_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, standard_feeds()) as http:
            with caplog.at_level(logging.WARNING, logger="matomeru.arxiv"):
                papers, fetched = await collect_arxiv(http, ["cs.AI"], [], make_window())

        assert fetched == 5  # fetch not aborted
        assert len(papers) == 3
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("2611.09999v1" in r.getMessage() for r in warnings)

    @pytest.mark.asyncio
    async def test_one_feed_failure_is_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, {"/rss/cs.AI": AI_FEED, "/rss/cs.LG": 500}) as http:
            papers, fetched = await collect_arxiv(http, ["cs.AI", "cs.LG"], [], make_window())

        assert fetched == 5  # only the healthy feed's items counted
        assert {p.primary_category for p in papers} == {"cs.AI"}

    @pytest.mark.asyncio
    async def test_one_feed_xml_parse_error_is_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, {"/rss/cs.AI": AI_FEED, "/rss/cs.LG": "<not-closed"}) as http:
            papers, fetched = await collect_arxiv(http, ["cs.AI", "cs.LG"], [], make_window())

        assert fetched == 5
        assert {p.primary_category for p in papers} == {"cs.AI"}

    @pytest.mark.asyncio
    async def test_all_feeds_failure_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, {"/rss/cs.AI": 500, "/rss/cs.LG": 500}) as http:
            with pytest.raises(RuntimeError, match="all 2 category feed\\(s\\) failed"):
                await collect_arxiv(http, ["cs.AI", "cs.LG"], [], make_window())

    @pytest.mark.asyncio
    async def test_single_feed_xml_error_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_RSS_BASE", raising=False)

        async with make_client(captured, {"/rss/cs.AI": "this is not xml at all"}) as http:
            with pytest.raises(RuntimeError, match="all 1 category feed\\(s\\) failed"):
                await collect_arxiv(http, ["cs.AI"], [], make_window())
