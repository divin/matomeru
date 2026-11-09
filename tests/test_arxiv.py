"""Tests for arXiv candidate retrieval (no network; httpx.MockTransport)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx
import pytest

from matomeru.arxiv import _build_search_query, collect_arxiv
from matomeru.config import ConfigError
from matomeru.window import RunWindow

# Canned Atom feed modeled on the real export.arxiv.org/api/query output for
# astro-ph.EP entries. Contains:
#   entry A: fresh paper with -05:00 offset timestamps and an arxiv:comment
#   entry B: revised paper (published out of window, updated in window) with
#            an arxiv:announced element
#   entry C: published in window but updated later than the window end
#   entry D: entirely out of window
#   entry E: malformed (unparseable published date), id parseable
FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <link href="http://arxiv.org/api/query" rel="self" type="application/atom+xml"/>
  <title type="html">ArXiv Query: search_query=...</title>
  <id>http://arxiv.org/api/abcdefgh</id>
  <updated>2026-11-09T00:00:00-05:00</updated>
  <opensearch:totalResults>5</opensearch:totalResults>
  <opensearch:startIndex>0</opensearch:startIndex>
  <opensearch:itemsPerPage>100</opensearch:itemsPerPage>

  <entry>
    <id>http://arxiv.org/abs/2611.08700v1</id>
    <updated>2026-11-08T20:00:00-05:00</updated>
    <published>2026-11-08T20:00:00-05:00</published>
    <title>A Hot Jupiter Around a Nearby M Dwarf: TWISTED-1</title>
    <summary>  We report the discovery of a transiting hot Jupiter
      around an M dwarf.  Multi-band photometry yields a radius of
      1.2 R_Jup. </summary>
    <author>
      <name>Alice Author</name>
    </author>
    <author>
      <name>Bob Builder</name>
    </author>
    <arxiv:comment>12 pages, 5 figures, accepted to AAS Journals</arxiv:comment>
    <link href="http://arxiv.org/abs/2611.08700v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2611.08700v1" type="application/pdf"/>
    <arxiv:primary_category term="astro-ph.EP"/>
    <category term="astro-ph.EP"/>
    <category term="astro-ph.IM"/>
  </entry>

  <entry>
    <id>http://arxiv.org/abs/2611.08123v2</id>
    <updated>2026-11-08T18:00:00-05:00</updated>
    <published>2026-11-05T20:00:00-05:00</published>
    <title>Revised: Atmospheric Retrieval of Sub-Neptunes</title>
    <summary>A revised analysis of sub-Neptune atmospheres with JWST.</summary>
    <author>
      <name>Carol Compiler</name>
    </author>
    <arxiv:announced>Wed, 4 Nov 2026 20:00:00 EST</arxiv:announced>
    <arxiv:comment>v2: added appendix</arxiv:comment>
    <link href="http://arxiv.org/abs/2611.08123v2" rel="alternate" type="text/html"/>
    <arxiv:primary_category term="astro-ph.EP"/>
    <category term="astro-ph.EP"/>
  </entry>

  <entry>
    <id>http://arxiv.org/abs/2611.09000v1</id>
    <updated>2026-11-09T13:00:00+00:00</updated>
    <published>2026-11-08T15:00:00+00:00</published>
    <title>Tidally Heated Exoplanets: A Population Study</title>
    <summary>We study a population of tidally heated exoplanets.</summary>
    <author>
      <name>Dave Digger</name>
    </author>
    <link href="http://arxiv.org/abs/2611.09000v1" rel="alternate" type="text/html"/>
    <arxiv:primary_category term="astro-ph.EP"/>
    <category term="astro-ph.EP"/>
  </entry>

  <entry>
    <id>http://arxiv.org/abs/2610.12345v3</id>
    <updated>2026-11-01T12:00:00+00:00</updated>
    <published>2026-11-01T12:00:00+00:00</published>
    <title>An Old Paper About Dusty Debris Disks</title>
    <summary>This paper is far outside the 24-hour window.</summary>
    <author>
      <name>Eve Eavesdropper</name>
    </author>
    <link href="http://arxiv.org/abs/2610.12345v3" rel="alternate" type="text/html"/>
    <arxiv:primary_category term="astro-ph.EP"/>
    <category term="astro-ph.EP"/>
  </entry>

  <entry>
    <id>http://arxiv.org/abs/2611.09999v1</id>
    <updated>2026-11-08T21:00:00-05:00</updated>
    <published>not-a-valid-timestamp</published>
    <title>A Broken Entry With an Unparseable Date</title>
    <summary>This entry should be skipped with a warning.</summary>
    <author>
      <name>Frank Faulty</name>
    </author>
    <link href="http://arxiv.org/abs/2611.09999v1" rel="alternate" type="text/html"/>
    <arxiv:primary_category term="astro-ph.EP"/>
    <category term="astro-ph.EP"/>
  </entry>
</feed>
"""

UTC = timezone.utc


def make_window() -> RunWindow:
    """Fixed 24-hour window: 2026-11-08 12:00 UTC to 2026-11-09 12:00 UTC."""
    return RunWindow(
        start_utc=datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC),
        end_utc=datetime(2026, 11, 9, 12, 0, 0, tzinfo=UTC),
        report_date=datetime(2026, 11, 9).date(),
        tz_name="UTC",
    )


def make_client(captured: list[httpx.Request], feed: str = FEED) -> httpx.AsyncClient:
    """Async client backed by a MockTransport that records requests."""

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, text=feed, headers={"Content-Type": "application/atom+xml"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestSearchQueryConstruction:
    """Tests for the search_query string construction."""

    def test_both_categories_and_terms(self) -> None:
        query = _build_search_query(["astro-ph.EP", "cs.LG"], ["exoplanet", "transit"])
        assert query == '(cat:astro-ph.EP OR cat:cs.LG) AND (abs:"exoplanet" OR abs:"transit")'

    def test_categories_only(self) -> None:
        query = _build_search_query(["astro-ph.EP"], [])
        assert query == "cat:astro-ph.EP"

    def test_terms_only(self) -> None:
        query = _build_search_query([], ["hot jupiter"])
        assert query == 'abs:"hot jupiter"'

    def test_embedded_quotes_stripped(self) -> None:
        query = _build_search_query([], ['we"ird" term'])
        assert query == 'abs:"weird term"'

    def test_both_empty_raises(self) -> None:
        with pytest.raises(ConfigError, match="at least one category or search term"):
            _build_search_query([], [])


class TestCollectArxiv:
    """Tests for collect_arxiv against a canned Atom feed."""

    @pytest.mark.asyncio
    async def test_query_params_categories_and_terms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_API_BASE", raising=False)

        async with make_client(captured) as http:
            papers, fetched = await collect_arxiv(
                http,
                ["astro-ph.EP", "cs.LG"],
                ["exoplanet", "transit"],
                make_window(),
                max_results=100,
            )

        assert len(captured) == 1  # politeness: single request per run
        request = captured[0]
        assert request.url.host == "export.arxiv.org"
        assert request.url.path == "/api/query"
        params = request.url.params
        assert params["search_query"] == ('(cat:astro-ph.EP OR cat:cs.LG) AND (abs:"exoplanet" OR abs:"transit")')
        assert params["sortBy"] == "submittedDate"
        assert params["sortOrder"] == "descending"
        assert params["start"] == "0"
        assert params["max_results"] == "100"
        assert request.headers["User-Agent"].startswith("matomeru/0.1.0")
        assert "(self-hosted digest)" in request.headers["User-Agent"]
        assert fetched == 5
        assert [p.arxiv_id for p in papers] == ["2611.08700v1", "2611.08123v2"]

    @pytest.mark.asyncio
    async def test_query_params_categories_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_API_BASE", raising=False)

        async with make_client(captured) as http:
            await collect_arxiv(http, ["astro-ph.EP"], [], make_window())

        assert captured[0].url.params["search_query"] == "cat:astro-ph.EP"

    @pytest.mark.asyncio
    async def test_query_params_terms_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_API_BASE", raising=False)

        async with make_client(captured) as http:
            await collect_arxiv(http, [], ["hot jupiter"], make_window())

        assert captured[0].url.params["search_query"] == 'abs:"hot jupiter"'

    @pytest.mark.asyncio
    async def test_api_base_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.setenv("ARXIV_API_BASE", "http://mirror.example.org")

        async with make_client(captured) as http:
            await collect_arxiv(http, ["astro-ph.EP"], [], make_window())

        assert captured[0].url.host == "mirror.example.org"

    @pytest.mark.asyncio
    async def test_both_empty_raises_config_error(self) -> None:
        captured: list[httpx.Request] = []
        async with make_client(captured) as http:
            with pytest.raises(ConfigError, match="at least one category or search term"):
                await collect_arxiv(http, [], [], make_window())
        assert captured == []  # no request made

    @pytest.mark.asyncio
    async def test_parsing_all_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_API_BASE", raising=False)

        async with make_client(captured) as http:
            papers, _ = await collect_arxiv(http, ["astro-ph.EP"], [], make_window())

        fresh = papers[0]
        assert fresh.arxiv_id == "2611.08700v1"
        assert fresh.abs_url == "http://arxiv.org/abs/2611.08700v1"
        assert fresh.pdf_url == "http://arxiv.org/pdf/2611.08700v1"
        assert fresh.title == "A Hot Jupiter Around a Nearby M Dwarf: TWISTED-1"
        assert fresh.abstract == (
            "We report the discovery of a transiting hot Jupiter around an M dwarf. "
            "Multi-band photometry yields a radius of 1.2 R_Jup."
        )
        assert fresh.authors == ["Alice Author", "Bob Builder"]
        assert fresh.categories == ["astro-ph.EP", "astro-ph.IM"]
        assert fresh.primary_category == "astro-ph.EP"
        # -05:00 offset converted to UTC: 2026-11-08T20:00:00-05:00 -> 01:00Z Nov 9
        assert fresh.published == datetime(2026, 11, 9, 1, 0, 0, tzinfo=UTC)
        assert fresh.updated == datetime(2026, 11, 9, 1, 0, 0, tzinfo=UTC)
        assert fresh.published.tzinfo is UTC
        assert fresh.announcement_hint == "12 pages, 5 figures, accepted to AAS Journals"

        revised = papers[1]
        assert revised.arxiv_id == "2611.08123v2"
        assert revised.pdf_url == "http://arxiv.org/pdf/2611.08123v2"
        assert revised.published == datetime(2026, 11, 6, 1, 0, 0, tzinfo=UTC)
        assert revised.updated == datetime(2026, 11, 8, 23, 0, 0, tzinfo=UTC)
        # arxiv:announced takes precedence over arxiv:comment
        assert revised.announcement_hint == "Wed, 4 Nov 2026 20:00:00 EST"
        assert revised.announcement_hint is not None and "arxiv" not in revised.announcement_hint

    @pytest.mark.asyncio
    async def test_window_filtering_on_newest_stamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_API_BASE", raising=False)

        async with make_client(captured) as http:
            papers, fetched = await collect_arxiv(http, ["astro-ph.EP"], [], make_window())

        assert fetched == 5
        ids = [p.arxiv_id for p in papers]
        # B: updated in window while published is out -> kept
        assert "2611.08123v2" in ids
        # C: published in window while updated is out -> excluded
        assert "2611.09000v1" not in ids
        # D: entirely out of window -> excluded
        assert "2610.12345v3" not in ids
        # Malformed entry E never appears
        assert "2611.09999v1" not in ids

    @pytest.mark.asyncio
    async def test_malformed_entry_skipped_with_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_API_BASE", raising=False)

        async with make_client(captured) as http:
            with caplog.at_level(logging.WARNING, logger="matomeru.arxiv"):
                papers, fetched = await collect_arxiv(http, ["astro-ph.EP"], [], make_window())

        assert fetched == 5  # fetch not aborted
        assert len(papers) == 2
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("2611.09999v1" in r.getMessage() for r in warnings)

    @pytest.mark.asyncio
    async def test_window_boundaries_are_exclusive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A stamp exactly at start_utc or end_utc is not kept."""

        # Window that exactly matches entry A's newest stamp (01:00Z Nov 9)
        window = RunWindow(
            start_utc=datetime(2026, 11, 9, 1, 0, 0, tzinfo=UTC),
            end_utc=datetime(2026, 11, 9, 1, 0, 0, tzinfo=UTC),
            report_date=datetime(2026, 11, 9).date(),
            tz_name="UTC",
        )
        captured: list[httpx.Request] = []
        monkeypatch.delenv("ARXIV_API_BASE", raising=False)

        async with make_client(captured) as http:
            papers, _ = await collect_arxiv(http, ["astro-ph.EP"], [], window, max_results=100)

        assert papers == []
