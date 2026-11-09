"""arXiv candidate retrieval via the official arXiv Atom API.

Fetches recently submitted papers matching configured categories and/or
abstract search terms, filters them against the 24-hour run window, and
returns ArxivPaper models for downstream relevance screening.

Note on arXiv's announcement cycle: announcements happen Sun-Thu around
20:00 ET, and the /list/<cat>/new pages reflect that cycle. A freshly
announced paper can have a ``published`` date days earlier, so the window
filter uses the newest stamp (``updated`` if it is after ``published``,
otherwise ``published``) to decide whether a paper is in-window.
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx

from matomeru import __version__
from matomeru.config import ConfigError
from matomeru.models import ArxivPaper
from matomeru.window import RunWindow

logger = logging.getLogger(__name__)

DEFAULT_ARXIV_API_BASE = "https://export.arxiv.org"

_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"


def _build_search_query(categories: list[str], search_terms: list[str]) -> str:
    """Build the arXiv search_query string from categories and search terms.

    - Both present: "(cat:X OR cat:Y) AND (abs:\\"t1\\" OR abs:\\"t2\\")"
    - Categories only: "cat:X OR cat:Y"
    - Terms only: "abs:\\"t1\\" OR abs:\\"t2\\""
    - Neither: ConfigError.

    Embedded double quotes in search terms are stripped before quoting.
    """
    if not categories and not search_terms:
        raise ConfigError("arXiv retrieval needs at least one category or search term")

    cat_part = " OR ".join(f"cat:{c}" for c in categories) if categories else ""
    terms = [t.replace('"', "").strip() for t in search_terms]
    terms = [t for t in terms if t]
    term_part = " OR ".join(f'abs:"{t}"' for t in terms) if terms else ""

    if cat_part and term_part:
        return f"({cat_part}) AND ({term_part})"
    return cat_part or term_part


def _api_base(api_base: str | None = None) -> str:
    """Resolve the arXiv API base URL (explicit override, else env, else default)."""
    base = api_base or os.environ.get("ARXIV_API_BASE") or DEFAULT_ARXIV_API_BASE
    return base.rstrip("/")


def _parse_utc(text: str) -> datetime:
    """Parse an Atom timestamp like 2026-11-08T20:00:00-05:00 into UTC."""
    dt = datetime.fromisoformat(text.strip())
    return dt.astimezone(timezone.utc)


def _pdf_url(abs_url: str) -> str:
    """Derive the PDF URL from an abstract URL (no version change)."""
    marker = "/abs/"
    idx = abs_url.rfind(marker)
    if idx == -1:
        return abs_url
    return abs_url[:idx] + "/pdf/" + abs_url[idx + len(marker) :]


def _parse_entry(entry: ET.Element) -> ArxivPaper:
    """Parse a single Atom <entry> into an ArxivPaper.

    Raises:
        ValueError: If the entry is missing required fields or has
            unparseable timestamps.
    """
    abs_url = (entry.findtext(f"{{{_ATOM_NS}}}id") or "").strip()
    title = " ".join((entry.findtext(f"{{{_ATOM_NS}}}title") or "").split())
    abstract = " ".join((entry.findtext(f"{{{_ATOM_NS}}}summary") or "").split())
    authors = [
        (author.findtext(f"{{{_ATOM_NS}}}name") or "").strip() for author in entry.findall(f"{{{_ATOM_NS}}}author")
    ]
    authors = [a for a in authors if a]
    categories = [cat.get("term", "") for cat in entry.findall(f"{{{_ATOM_NS}}}category")]
    categories = [c for c in categories if c]
    primary_el = entry.find(f"{{{_ARXIV_NS}}}primary_category")
    primary_category = primary_el.get("term", "") if primary_el is not None else ""
    published_text = entry.findtext(f"{{{_ATOM_NS}}}published")
    updated_text = entry.findtext(f"{{{_ATOM_NS}}}updated")
    if not abs_url or not title or not published_text or not updated_text:
        raise ValueError("entry missing required fields (id/title/published/updated)")
    published = _parse_utc(published_text)
    updated = _parse_utc(updated_text)

    announcement_hint: str | None = None
    announced = entry.findtext(f"{{{_ARXIV_NS}}}announced")
    comment = entry.findtext(f"{{{_ARXIV_NS}}}comment")
    if announced and announced.strip():
        announcement_hint = announced.strip()
    elif comment and comment.strip():
        announcement_hint = comment.strip()

    return ArxivPaper(
        arxiv_id=abs_url.rstrip("/").rsplit("/", 1)[-1],
        abs_url=abs_url,
        pdf_url=_pdf_url(abs_url),
        title=title,
        abstract=abstract,
        authors=authors,
        categories=categories,
        primary_category=primary_category or (categories[0] if categories else ""),
        published=published,
        updated=updated,
        announcement_hint=announcement_hint,
    )


def _newest_stamp(paper: ArxivPaper) -> datetime:
    """The window-relevant timestamp: newest of published/updated."""
    return paper.updated if paper.updated > paper.published else paper.published


async def collect_arxiv(
    http: httpx.AsyncClient,
    categories: list[str],
    search_terms: list[str],
    window: RunWindow,
    *,
    max_results: int = 100,
    api_base: str | None = None,
) -> tuple[list[ArxivPaper], int]:
    """Collect arXiv candidate papers within the run window.

    Makes a single polite request to the arXiv Atom API, sorted by
    submittedDate descending, and keeps papers whose newest stamp
    (updated if newer than published, else published) falls strictly
    inside (window.start_utc, window.end_utc).

    Args:
        http: Async HTTP client (httpx.AsyncClient).
        categories: arXiv categories such as "astro-ph.EP".
        search_terms: Abstract search terms.
        window: The 24-hour run window.
        max_results: Maximum number of results to request (default 100).
        api_base: Explicit API base URL override; when None, the
            ARXIV_API_BASE env var is consulted, defaulting to
            https://export.arxiv.org. Callers that have a RuntimeConfig
            should pass runtime.arxiv_api_base here (config is the
            source of truth).

    Returns:
        Tuple of (papers_in_window, fetched_total).

    Raises:
        ConfigError: If both categories and search_terms are empty.
        httpx.HTTPError: On network/HTTP failure.
    """
    search_query = _build_search_query(categories, search_terms)
    url = f"{_api_base(api_base)}/api/query"
    headers = {"User-Agent": f"matomeru/{__version__} (self-hosted digest)"}

    logger.debug("Requesting arXiv API: %s (query=%s)", url, search_query)
    response = await http.get(
        url,
        params={
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": "0",
            "max_results": str(max_results),
        },
        headers=headers,
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)
    entries = root.findall(f"{{{_ATOM_NS}}}entry")
    fetched_total = len(entries)

    papers: list[ArxivPaper] = []
    for entry in entries:
        try:
            paper = _parse_entry(entry)
        except (ValueError, AttributeError, TypeError) as e:
            entry_id = (entry.findtext(f"{{{_ATOM_NS}}}id") or "").strip()
            logger.warning("Skipping malformed arXiv entry %s: %s", entry_id or "<unknown>", e)
            continue
        stamp = _newest_stamp(paper)
        if window.start_utc < stamp < window.end_utc:
            papers.append(paper)

    logger.debug("arXiv: fetched=%d, in_window=%d", fetched_total, len(papers))
    return papers, fetched_total
