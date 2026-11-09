"""arXiv candidate retrieval via the per-category RSS announcement feeds.

Fetches the current announcement batch for each configured category from
``https://rss.arxiv.org/rss/<category>``. The RSS feed contains exactly the
papers announced in the current cycle (new submissions, cross-lists, and
replacements), so no query building or server-side filtering is needed.

Note on arXiv's announcement cycle: the RSS ``pubDate`` on every item is the
announcement stamp of the current batch (not the paper's original submission
date). Weekend submissions announce on Monday while keeping Friday stamps in
the Atom API, which is why collection is based on announcements: the feed's
``pubDate`` IS the announcement time. For RSS-collected papers, both
``published`` and ``updated`` therefore mean announcement time, and the
window filter keeps papers whose announcement stamp falls strictly inside
the run window. A normal daily run finds the whole batch in-window; a run
made shortly after two announcements (a missed day) may contain older
announcement stamps, which are correctly dropped.
"""

from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import httpx

from matomeru import __version__
from matomeru.config import ConfigError
from matomeru.models import ArxivPaper
from matomeru.window import RunWindow

logger = logging.getLogger(__name__)

DEFAULT_ARXIV_RSS_BASE = "https://rss.arxiv.org"

_ARXIV_NS = "http://arxiv.org/schemas/atom"
_DC_NS = "http://purl.org/dc/elements/1.1/"

# Version suffix such as the "v1" in "2611.08572v1"; only matched when it
# directly follows the dot-separated numeric part of an arXiv id.
_VERSION_SUFFIX_RE = re.compile(r"(?P<base>\.\d+)v\d+$")


def _rss_base() -> str:
    """Resolve the arXiv RSS base URL (env override, else default).

    Uses ``or`` so an explicitly empty ARXIV_RSS_BASE falls back to the
    default instead of producing an invalid URL.
    """
    base = os.environ.get("ARXIV_RSS_BASE") or DEFAULT_ARXIV_RSS_BASE
    return base.rstrip("/")


def _normalize_category(category: str) -> str:
    """Validate and normalize a single arXiv category name.

    Strips surrounding whitespace and an optional leading "cat:" prefix.
    Rejects values containing spaces or quotes.

    Raises:
        ConfigError: If the category is empty or contains spaces/quotes.
    """
    value = category.strip()
    if value.lower().startswith("cat:"):
        value = value[4:].strip()
    if not value:
        raise ConfigError(f"invalid arXiv category: {category!r}")
    if " " in value or '"' in value or "'" in value:
        raise ConfigError(f"invalid arXiv category {category!r}: spaces and quotes are not allowed")
    return value


def _normalize_terms(search_terms: list[str]) -> list[str]:
    """Prepare search terms for the client-side prefilter.

    Strips embedded quotes and surrounding whitespace; drops empty terms.
    """
    terms = [t.replace('"', "").strip() for t in search_terms]
    return [t for t in terms if t]


def _base_id(arxiv_id: str) -> str:
    """Strip the version suffix from an arXiv id (2611.08572v1 -> 2611.08572).

    The suffix is removed only when it follows the dot-separated numeric
    part, so ids without a version are left untouched.
    """
    return _VERSION_SUFFIX_RE.sub(r"\g<base>", arxiv_id)


def _id_from_guid(guid: str) -> str | None:
    """Extract the arXiv id from an RSS guid.

    Handles the "oai:arXiv.org:2611.08572v1" form (take the part after the
    last colon) as well as bare ids and permalink URLs.
    """
    candidate = guid.strip().rsplit(":", 1)[-1].strip()
    return candidate or None


def _id_from_link(link: str) -> str | None:
    """Extract the arXiv id from an abstract URL (last path segment)."""
    path = urlparse(link).path.rstrip("/")
    candidate = path.rsplit("/", 1)[-1].strip()
    return candidate or None


def _pdf_url(abs_url: str, versioned_id: str) -> str:
    """Derive the PDF URL from an abstract URL.

    Uses the versioned id from the guid when available (e.g.
    https://arxiv.org/pdf/2611.08572v1), otherwise keeps the versionless
    link id (https://arxiv.org/pdf/2611.08572).
    """
    marker = "/abs/"
    idx = abs_url.rfind(marker)
    if idx == -1:
        return abs_url
    pdf_base = abs_url[:idx] + "/pdf/"
    if _VERSION_SUFFIX_RE.search(versioned_id):
        return pdf_base + versioned_id
    return pdf_base + abs_url[idx + len(marker) :]


def _extract_abstract(description: str) -> str:
    """Extract the abstract text from an RSS item description.

    The description starts with an "arXiv:<id> Announce Type: <type>" prefix
    followed by an "Abstract:" marker; everything before (and including) the
    first "Abstract:" marker is dropped and remaining whitespace (including
    line breaks) is collapsed.
    """
    text = description.strip()
    marker = "Abstract:"
    idx = text.find(marker)
    if idx != -1:
        text = text[idx + len(marker) :]
    return " ".join(text.split())


def _parse_item(item: ET.Element, primary_category: str) -> ArxivPaper:
    """Parse a single RSS <item> into an ArxivPaper.

    The feed's channel is per-category, so the requested category is the
    primary category of every item in it.

    Raises:
        ValueError: If the item is missing required fields, has no parsable
            arXiv id, or has an unparseable pubDate.
    """
    title = " ".join((item.findtext("title") or "").split())
    link = (item.findtext("link") or "").strip()
    guid = (item.findtext("guid") or "").strip()
    pub_date_text = (item.findtext("pubDate") or "").strip()
    if not title or not link or not pub_date_text:
        raise ValueError("item missing required fields (title/link/pubDate)")

    arxiv_id = _id_from_guid(guid) or _id_from_link(link)
    if not arxiv_id:
        raise ValueError("item has no parsable arXiv id (guid/link)")

    try:
        stamp = parsedate_to_datetime(pub_date_text)
    except (TypeError, ValueError, OverflowError) as e:
        raise ValueError(f"unparseable pubDate {pub_date_text!r}") from e
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    stamp = stamp.astimezone(timezone.utc)

    # The announcement stamp is the window-relevant time; published and
    # updated both carry it (documented on the model).
    authors = [author.strip() for author in (item.findtext(f"{{{_DC_NS}}}creator") or "").split(",") if author.strip()]
    categories = [
        category.strip()
        for category in (element.text or "" for element in item.findall("category"))
        if category.strip()
    ]

    announcement_hint: str | None = None
    announce_el = item.find(f"{{{_ARXIV_NS}}}announce_type")
    if announce_el is not None and announce_el.text and announce_el.text.strip():
        announcement_hint = f"announce_type={announce_el.text.strip()}"

    return ArxivPaper(
        arxiv_id=arxiv_id,
        abs_url=link,
        pdf_url=_pdf_url(link, arxiv_id),
        title=title,
        abstract=_extract_abstract(item.findtext("description") or ""),
        authors=authors,
        categories=categories,
        primary_category=primary_category,
        published=stamp,
        updated=stamp,
        announcement_hint=announcement_hint,
    )


def _item_id_guess(item: ET.Element) -> str:
    """Best-effort id extraction for malformed-item warning logs."""
    guid = (item.findtext("guid") or "").strip()
    if guid:
        return guid
    return (item.findtext("link") or "").strip()


def _matches_terms(paper: ArxivPaper, terms: list[str]) -> bool:
    """Case-insensitive substring match of ANY term against title + abstract."""
    haystack = f"{paper.title} {paper.abstract}".lower()
    return any(term.lower() in haystack for term in terms)


async def collect_arxiv(
    http: httpx.AsyncClient,
    categories: list[str],
    search_terms: list[str],
    window: RunWindow,
    *,
    max_results: int = 100,
    category_terms: dict[str, list[str]] | None = None,
) -> tuple[list[ArxivPaper], int]:
    """Collect arXiv candidate papers from the RSS announcement feeds.

    Fetches one feed per category (``<rss_base>/rss/<category>``), parses the
    current announcement batch (new submissions, cross-lists, replacements),
    applies that feed's selector terms, deduplicates across categories by
    base arXiv id, applies the announcement-stamp window filter, sorts by
    announcement time descending, and truncates to ``max_results``.

    Processing order: collect per feed -> per-feed selector filter ->
    dedupe -> window filter -> sort by published desc -> truncate to
    max_results.

    Args:
        http: Async HTTP client (httpx.AsyncClient).
        categories: arXiv categories such as "cs.AI" (a "cat:" prefix is
            tolerated and stripped). Required: RSS is per-category, so
            search_terms alone cannot retrieve anything.
        search_terms: Client-side prefilter terms; a paper is kept when ANY
            term case-insensitively matches its title or abstract. Embedded
            quotes in terms are stripped. Empty list -> no filtering. Serves
            as the selector for every category without a category_terms
            entry.
        window: The 24-hour run window. Papers whose announcement stamp
            (the feed pubDate) is strictly inside
            (window.start_utc, window.end_utc) are kept.
        max_results: Maximum number of papers to return (default 100).
        category_terms: Optional per-category selector override mapping
            category names (keys are raw, exactly as configured) to
            search-term lists. Per feed, the selector is resolved as: if the
            normalized category (whitespace and optional "cat:" prefix
            stripped, lowercased) has an entry in this mapping, that list is
            THE selector for the feed — an explicitly empty list means "keep
            all papers in this category"; otherwise, if the raw category
            string has an entry, that list is the selector; otherwise the
            global ``search_terms`` list is the selector. Filtering is
            applied per feed BEFORE cross-category dedupe, so a paper enters
            the digest when it matches the selector of ANY feed it appears
            in (first-wins dedupe must not let one category's terms veto a
            paper another category's selector wants). A category_terms key
            without a matching configured category is inert. Terms are
            normalized like ``search_terms`` (quotes stripped, empties
            dropped). None or empty mapping -> every feed uses the global
            ``search_terms``.

    Returns:
        Tuple of (papers_in_window, fetched_total) where fetched_total is
        the total number of items seen across all successfully fetched feeds.

    Raises:
        ConfigError: If categories is empty (even when search_terms are
            present, since RSS retrieval is per-category), or if a category
            contains spaces or quotes.
        RuntimeError: If every category feed fails (matching the Reddit
            total-failure semantics). Individual feed failures are isolated:
            a WARNING is logged and the remaining categories still collect.
    """
    if not categories:
        raise ConfigError("arXiv collection needs at least one category")
    normalized = [_normalize_category(category) for category in categories]
    category_terms = category_terms or {}

    rss_base = _rss_base()
    headers = {"User-Agent": f"matomeru/{__version__} (self-hosted digest)"}

    fetched_total = 0
    papers: list[ArxivPaper] = []
    seen_base_ids: set[str] = set()
    failed = 0

    for category, normalized_cat in zip(categories, normalized):
        # Per-feed selector: explicit entry for the normalized category
        # (lowercased) wins, then an entry for the raw category string, then
        # the global search_terms fallback. An explicitly empty list means
        # "keep all papers in this category".
        terms_for_feed = category_terms.get(normalized_cat.lower(), category_terms.get(category, search_terms))
        feed_terms = _normalize_terms(terms_for_feed)
        url = f"{rss_base}/rss/{normalized_cat}"
        logger.debug("Requesting arXiv RSS feed: %s", url)
        try:
            response = await http.get(url, headers=headers)
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except (httpx.HTTPError, ET.ParseError) as e:
            failed += 1
            logger.warning("arXiv RSS feed for %s failed: %s", category, e)
            continue
        # The live feed places <item> elements as siblings of <channel>
        # (classic RSS puts them inside); iter("item") covers both shapes.
        items = list(root.iter("item"))
        fetched_total += len(items)
        for item in items:
            try:
                paper = _parse_item(item, primary_category=normalized_cat)
            except (ValueError, AttributeError, TypeError) as e:
                logger.warning("Skipping malformed arXiv item %s: %s", _item_id_guess(item) or "<unknown>", e)
                continue
            # Per-feed selector filter runs BEFORE the shared dedupe set so
            # that a paper kept by ANY feed's selector survives (first wins).
            if feed_terms and not _matches_terms(paper, feed_terms):
                continue
            base_id = _base_id(paper.arxiv_id)
            if base_id in seen_base_ids:
                continue
            seen_base_ids.add(base_id)
            papers.append(paper)

    if failed == len(normalized):
        raise RuntimeError(f"all {len(normalized)} category feed(s) failed")

    papers = [paper for paper in papers if window.start_utc < paper.published < window.end_utc]
    papers.sort(key=lambda paper: paper.published, reverse=True)
    papers = papers[:max_results]

    logger.debug("arXiv: fetched=%d, kept=%d", fetched_total, len(papers))
    return papers, fetched_total
