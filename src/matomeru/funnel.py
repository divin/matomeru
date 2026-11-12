"""Reranker funnel for Reddit threads and arXiv papers (v1.1.0, milestones 7b/7c).

Selection is deterministic code: the reranker scores every collected item
against each profile section (per-section queries — see
:mod:`matomeru.reranker_client`).

- Reddit (:func:`funnel_reddit_threads`): keep the **top X threads per
  subreddit** by score, with an optional global minimum-score floor applied
  *before* top-X selection.
- arXiv (:func:`funnel_arxiv_papers`): **threshold-based** — every paper whose
  score is at least ``min_score`` (raw logit space, see the plan) is kept; no
  top-N because daily arXiv inflow varies wildly.

The funnel only selects; it never reorders — the kept items stay in collection
order, so the digest ordering is unchanged from v1.

Scoring design (plan §Version 1.1.0): documents are ``title + selftext`` for
Reddit threads (comments excluded) and ``title + abstract`` for arXiv papers;
queries are the profile section texts as built by the caller
(``f"{name}\\n\\n{text}"``). Each item's score is its max across sections; the
winning section name is recorded for calibration.

Failure semantics: any exception from the reranker client propagates to the
caller, which fails the run (fail-early, consistent with collection failures).
A missing score for a collected item is a response integrity failure in the
client, never a silent exclusion.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Generic, TypeVar

from pydantic import BaseModel

from matomeru.models import ArxivPaper, RedditThread
from matomeru.reranker_client import RerankerClient

logger = logging.getLogger(__name__)

_ItemT = TypeVar("_ItemT")


class ScoredItem(BaseModel):
    """One scored item: its max score across sections plus the winning section.

    ``subreddit`` is the Reddit-specific name carried over from the Reddit
    funnel; for arXiv papers it holds the paper's primary category (the row
    schema is source-agnostic and the DB has no separate category column).
    """

    item_id: str
    subreddit: str
    score: float
    section: str
    kept: bool


class FunnelOutcome(BaseModel, Generic[_ItemT]):
    """Result of one funnel pass over the collected items.

    ``kept``/``dropped`` partition the input items (both in collection
    order); ``scored`` holds one row per input item in input order.
    """

    kept: list[_ItemT]
    dropped: list[_ItemT]
    scored: list[ScoredItem]


def _section_name(section_query: str) -> str:
    """Recover the section name from its query text.

    Callers build section queries as ``f"{name}\\n\\n{text}"`` (the plan's
    "each profile section's name + text"), so the name is the first
    ``\\n\\n``-separated component.
    """
    return section_query.split("\n\n", 1)[0]


def _select_kept_indices(
    threads: list[RedditThread], scored: list[ScoredItem], *, top_x: int, min_score: float | None
) -> set[int]:
    """Choose which thread indices survive the funnel.

    The optional floor drops threads scoring below ``min_score`` first
    (``None`` disables it); the surviving threads then compete for the top-X
    slots per subreddit. Tie-break: score descending, then subreddit name,
    then thread id.
    """
    eligible = [i for i, item in enumerate(scored) if min_score is None or item.score >= min_score]
    ranked = sorted(
        eligible,
        key=lambda i: (-scored[i].score, threads[i].subreddit, threads[i].id),
    )
    kept: set[int] = set()
    per_subreddit: dict[str, int] = defaultdict(int)
    for i in ranked:
        subreddit = threads[i].subreddit
        if per_subreddit[subreddit] < top_x:
            per_subreddit[subreddit] += 1
            kept.add(i)
    return kept


async def funnel_reddit_threads(
    threads: list[RedditThread],
    sections: list[str],
    client: RerankerClient,
    *,
    top_x: int,
    min_score: float | None,
    report_date: str,
    source: str = "reddit",
) -> FunnelOutcome[RedditThread]:
    """Score the collected threads and keep the top X per subreddit.

    Args:
        threads: Collected threads in collection order.
        sections: Reranker query texts, one per profile section, built as
            ``f"{name}\\n\\n{text}"`` by the caller.
        client: The reranker client (any exception it raises propagates).
        top_x: How many threads to keep per subreddit.
        min_score: Optional global score floor applied before top-X selection
            (``None`` disables the floor).
        report_date: Report date string, logged for run correlation.
        source: Source name, logged for run correlation.

    Returns:
        A :class:`FunnelOutcome[RedditThread]` with kept/dropped threads
        (both in collection order) and one :class:`ScoredItem` row per input
        thread.
    """
    if not threads:
        # Nothing collected: skip scoring entirely (an empty rerank request is
        # a client error, and a quiet day must stay a quiet v1 day).
        return FunnelOutcome(kept=[], dropped=[], scored=[])

    logger.debug("funnel source=%s report_date=%s scoring %d document(s)", source, report_date, len(threads))
    documents = [f"{thread.title}\n\n{thread.selftext}" for thread in threads]
    pairs = await client.rerank_sections_with_argmax(sections, documents)
    assert len(pairs) == len(threads), "reranker returned a score count mismatch"

    scored = [
        ScoredItem(
            item_id=thread.id,
            subreddit=thread.subreddit,
            score=score,
            section=_section_name(sections[section_idx]),
            kept=False,
        )
        for thread, (score, section_idx) in zip(threads, pairs, strict=True)
    ]

    kept_indices = _select_kept_indices(threads, scored, top_x=top_x, min_score=min_score)
    for i in kept_indices:
        scored[i].kept = True

    kept = [thread for i, thread in enumerate(threads) if i in kept_indices]
    dropped = [thread for i, thread in enumerate(threads) if i not in kept_indices]
    return FunnelOutcome(kept=kept, dropped=dropped, scored=scored)


async def funnel_arxiv_papers(
    papers: list[ArxivPaper],
    sections: list[str],
    client: RerankerClient,
    *,
    min_score: float,
    report_date: str,
    source: str = "arxiv",
) -> FunnelOutcome[ArxivPaper]:
    """Score the collected papers and keep every paper at or above the threshold.

    Threshold-based selection (plan §Version 1.1.0): every paper whose score
    (raw logit space) is ``>= min_score`` is kept; there is no top-N because
    daily arXiv inflow varies wildly. The funnel only selects; it never
    reorders — kept and dropped papers stay in collection order.

    Args:
        papers: Collected papers in collection order.
        sections: Reranker query texts, one per profile section, built as
            ``f"{name}\\n\\n{text}"`` by the caller.
        client: The reranker client (any exception it raises propagates).
        min_score: Required score threshold in raw logit space; papers scoring
            at or above it are kept.
        report_date: Report date string, logged for run correlation.
        source: Source name, logged for run correlation.

    Returns:
        A :class:`FunnelOutcome[ArxivPaper]` with kept/dropped papers (both
        in collection order) and one :class:`ScoredItem` row per input paper.
        The row's ``item_id`` is the paper's ``arxiv_id`` and ``subreddit``
        carries the paper's primary category (source-agnostic schema, see
        :class:`ScoredItem`).
    """
    if not papers:
        # Nothing collected: skip scoring entirely (an empty rerank request is
        # a client error, and a quiet day must stay a quiet v1 day).
        return FunnelOutcome(kept=[], dropped=[], scored=[])

    logger.debug("funnel source=%s report_date=%s scoring %d document(s)", source, report_date, len(papers))
    documents = [f"{paper.title}\n\n{paper.abstract}" for paper in papers]
    pairs = await client.rerank_sections_with_argmax(sections, documents)
    assert len(pairs) == len(papers), "reranker returned a score count mismatch"

    scored = [
        ScoredItem(
            item_id=paper.arxiv_id,
            subreddit=paper.primary_category,
            score=score,
            section=_section_name(sections[section_idx]),
            kept=False,
        )
        for paper, (score, section_idx) in zip(papers, pairs, strict=True)
    ]

    kept_indices = {i for i, item in enumerate(scored) if item.score >= min_score}
    for i in kept_indices:
        scored[i].kept = True

    kept = [paper for i, paper in enumerate(papers) if i in kept_indices]
    dropped = [paper for i, paper in enumerate(papers) if i not in kept_indices]
    return FunnelOutcome(kept=kept, dropped=dropped, scored=scored)
