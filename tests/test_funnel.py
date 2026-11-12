"""Tests for the reranker funnels: Reddit (7b) and arXiv (7c), network-free."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from matomeru.funnel import FunnelOutcome, ScoredItem, funnel_arxiv_papers, funnel_reddit_threads
from matomeru.models import ArxivPaper, RedditThread
from matomeru.reranker_client import RerankerClient

SECTIONS = ["Hardware\n\nInterested in ZFS and NAS builds.", "Cooking\n\nInterested in carbonara."]

SECTION_NAMES = [section.split("\n\n", 1)[0] for section in SECTIONS]


class StubRerankerClient(RerankerClient):
    """In-memory client: per-document scores are one float per section."""

    def __init__(self, scores: list[list[float]]) -> None:
        super().__init__("http://reranker.invalid/v1/rerank")
        self._scores = scores
        self.calls: list[tuple[list[str], list[str]]] = []

    async def rerank_sections_with_argmax(self, sections: list[str], documents: list[str]) -> list[tuple[float, int]]:
        self.calls.append((list(sections), list(documents)))
        assert len(documents) == len(self._scores), "stub received unexpected document count"
        pairs: list[tuple[float, int]] = []
        for doc_scores in self._scores:
            # A shorter doc_scores means later sections score -inf (ignored).
            best_idx = max(range(len(doc_scores)), key=lambda s: doc_scores[s])
            pairs.append((doc_scores[best_idx], best_idx))
        return pairs


class ExplodingRerankerClient(RerankerClient):
    """Client whose scoring always fails."""

    def __init__(self) -> None:
        super().__init__("http://reranker.invalid/v1/rerank")

    async def rerank_sections_with_argmax(self, sections: list[str], documents: list[str]) -> list[tuple[float, int]]:
        raise RuntimeError("reranker exploded")


def make_thread(i: int, subreddit: str = "python", score: int = 0) -> RedditThread:
    return RedditThread(id=f"t3_{i}", subreddit=subreddit, title=f"Post {i}", score=score)


def make_arxiv_paper(i: int, primary_category: str = "cs.LG") -> ArxivPaper:
    stamp = datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC)
    return ArxivPaper(
        arxiv_id=f"2611.{i:05d}",
        abs_url=f"https://arxiv.org/abs/2611.{i:05d}",
        pdf_url=f"https://arxiv.org/pdf/2611.{i:05d}",
        title=f"Paper {i}",
        abstract=f"Abstract {i}",
        authors=["Alice", "Bob"],
        categories=[primary_category],
        primary_category=primary_category,
        published=stamp,
        updated=stamp,
    )


async def run_funnel(
    threads: list[RedditThread],
    scores: list[list[float]],
    *,
    top_x: int = 3,
    min_score: float | None = None,
    client: RerankerClient | None = None,
) -> tuple[FunnelOutcome, RerankerClient]:
    stub: RerankerClient = StubRerankerClient(scores) if client is None else client
    outcome = await funnel_reddit_threads(
        threads,
        SECTIONS,
        stub,
        top_x=top_x,
        min_score=min_score,
        report_date="2026-11-12",
    )
    return outcome, stub


class TestFunnelSelection:
    """Top-X per subreddit, tie-breaks, and floor behavior."""

    @pytest.mark.asyncio
    async def test_top_x_per_subreddit(self) -> None:
        threads = [make_thread(1), make_thread(2), make_thread(3), make_thread(4), make_thread(5)]
        # One section only; scores out of order relative to collection order.
        scores = [[-5.0], [-1.0], [-9.0], [-2.0], [-7.0]]
        outcome, _ = await run_funnel(threads, scores, top_x=2)

        assert [t.id for t in outcome.kept] == ["t3_2", "t3_4"]
        assert [t.id for t in outcome.dropped] == ["t3_1", "t3_3", "t3_5"]

    @pytest.mark.asyncio
    async def test_top_x_is_per_subreddit(self) -> None:
        threads = [
            make_thread(1, subreddit="python"),
            make_thread(2, subreddit="linux"),
            make_thread(3, subreddit="python"),
            make_thread(4, subreddit="linux"),
        ]
        scores = [[-1.0], [-2.0], [-3.0], [-4.0]]
        outcome, _ = await run_funnel(threads, scores, top_x=1)

        assert [t.id for t in outcome.kept] == ["t3_1", "t3_2"]

    @pytest.mark.asyncio
    async def test_tie_break_by_id_when_scores_equal(self) -> None:
        threads = [make_thread(3), make_thread(1), make_thread(2)]
        scores = [[-5.0], [-5.0], [-5.0]]
        outcome, _ = await run_funnel(threads, scores, top_x=2)

        # Equal scores: lowest thread ids win, regardless of collection order.
        assert sorted(t.id for t in outcome.kept) == ["t3_1", "t3_2"]
        assert [t.id for t in outcome.dropped] == ["t3_3"]

    @pytest.mark.asyncio
    async def test_tie_break_by_subreddit_is_deterministic(self) -> None:
        # Cross-subreddit ties cannot change per-subreddit selection, but the
        # global ranking must still be deterministic (subreddit then id).
        threads = [make_thread(1, subreddit="zeta"), make_thread(2, subreddit="alpha")]
        scores = [[-5.0], [-5.0]]
        outcome, _ = await run_funnel(threads, scores, top_x=1)
        assert len(outcome.kept) == 2  # one slot per subreddit

    @pytest.mark.asyncio
    async def test_floor_none_keeps_low_scores(self) -> None:
        threads = [make_thread(1), make_thread(2)]
        scores = [[-900.0], [-800.0]]
        outcome, _ = await run_funnel(threads, scores, top_x=3, min_score=None)
        assert len(outcome.kept) == 2

    @pytest.mark.asyncio
    async def test_floor_drops_below_threshold(self) -> None:
        threads = [make_thread(1), make_thread(2), make_thread(3)]
        # Logit space: -9 is below the floor of -6, the rest pass.
        scores = [[-5.0], [-9.0], [-4.0]]
        outcome, _ = await run_funnel(threads, scores, top_x=3, min_score=-6.0)

        assert [t.id for t in outcome.kept] == ["t3_1", "t3_3"]
        assert [t.id for t in outcome.dropped] == ["t3_2"]

    @pytest.mark.asyncio
    async def test_floor_applied_before_top_x_frees_a_slot(self) -> None:
        # Floor removes the -9 thread first; the slot goes to the next
        # candidate above the floor instead of staying empty.
        threads = [make_thread(1), make_thread(2), make_thread(3)]
        scores = [[-1.0], [-9.0], [-2.0]]
        outcome, _ = await run_funnel(threads, scores, top_x=2, min_score=-6.0)

        assert sorted(t.id for t in outcome.kept) == ["t3_1", "t3_3"]


class TestFunnelScoring:
    """Scored rows, section attribution, and client interaction."""

    @pytest.mark.asyncio
    async def test_one_scored_row_per_item_with_winning_section(self) -> None:
        threads = [make_thread(1), make_thread(2)]
        # Thread 1 wins on Hardware (section 0), thread 2 on Cooking (section 1).
        scores = [[-2.0, -5.0], [-8.0, -3.0]]
        outcome, _ = await run_funnel(threads, scores, top_x=3)

        assert outcome.scored == [
            ScoredItem(item_id="t3_1", subreddit="python", score=-2.0, section="Hardware", kept=True),
            ScoredItem(item_id="t3_2", subreddit="python", score=-3.0, section="Cooking", kept=True),
        ]

    @pytest.mark.asyncio
    async def test_scored_kept_flags_match_selection(self) -> None:
        threads = [make_thread(1), make_thread(2), make_thread(3)]
        scores = [[-1.0], [-2.0], [-3.0]]
        outcome, _ = await run_funnel(threads, scores, top_x=1)

        assert [row.kept for row in outcome.scored] == [True, False, False]
        assert all(row.section == "Hardware" for row in outcome.scored)

    @pytest.mark.asyncio
    async def test_documents_are_title_plus_selftext(self) -> None:
        thread = RedditThread(id="t3_1", subreddit="python", title="Title here", selftext="Body text")
        outcome, stub = await run_funnel([thread], [[-1.0]], top_x=1)
        assert isinstance(stub, StubRerankerClient)

        assert stub.calls[0][1] == ["Title here\n\nBody text"]
        assert stub.calls[0][0] == SECTIONS
        assert [t.id for t in outcome.kept] == ["t3_1"]

    @pytest.mark.asyncio
    async def test_kept_ordering_is_collection_order(self) -> None:
        # Collection order interleaves subreddits and scores; the funnel only
        # selects, it never reorders.
        threads = [
            make_thread(1, subreddit="python"),
            make_thread(2, subreddit="linux"),
            make_thread(3, subreddit="python"),
            make_thread(4, subreddit="linux"),
        ]
        scores = [[-4.0], [-1.0], [-2.0], [-3.0]]
        outcome, _ = await run_funnel(threads, scores, top_x=1)

        # Collection order would be [python/1, linux/2, python/3, linux/4];
        # per-subreddit winners are t3_2 (linux) and t3_3 (python), kept in
        # collection order.
        assert [t.id for t in outcome.kept] == ["t3_2", "t3_3"]
        assert [t.id for t in outcome.dropped] == ["t3_1", "t3_4"]

    @pytest.mark.asyncio
    async def test_empty_threads_skips_client(self) -> None:
        stub = ExplodingRerankerClient()
        outcome = await funnel_reddit_threads([], SECTIONS, stub, top_x=3, min_score=None, report_date="2026-11-12")
        assert outcome.kept == []
        assert outcome.dropped == []
        assert outcome.scored == []

    @pytest.mark.asyncio
    async def test_client_exception_propagates(self) -> None:
        with pytest.raises(RuntimeError, match="reranker exploded"):
            await run_funnel([make_thread(1)], client=ExplodingRerankerClient(), scores=[])


async def run_arxiv_funnel(
    papers: list[ArxivPaper],
    scores: list[list[float]],
    *,
    min_score: float = -6.0,
    client: RerankerClient | None = None,
) -> tuple[FunnelOutcome[ArxivPaper], RerankerClient]:
    stub: RerankerClient = StubRerankerClient(scores) if client is None else client
    outcome = await funnel_arxiv_papers(
        papers,
        SECTIONS,
        stub,
        min_score=min_score,
        report_date="2026-11-12",
    )
    return outcome, stub


class TestArxivFunnelSelection:
    """Threshold-based selection (score >= min_score, logit space)."""

    @pytest.mark.asyncio
    async def test_threshold_keeps_at_and_above_min_score(self) -> None:
        papers = [make_arxiv_paper(1), make_arxiv_paper(2), make_arxiv_paper(3), make_arxiv_paper(4)]
        # Logit space: -6.0 (exactly at the threshold) and above are kept.
        scores = [[-5.0], [-6.0], [-6.5], [-11.0]]
        outcome, _ = await run_arxiv_funnel(papers, scores, min_score=-6.0)

        assert [p.arxiv_id for p in outcome.kept] == ["2611.00001", "2611.00002"]
        assert [p.arxiv_id for p in outcome.dropped] == ["2611.00003", "2611.00004"]

    @pytest.mark.asyncio
    async def test_kept_and_dropped_preserve_collection_order(self) -> None:
        # Scores out of order relative to collection order; the funnel only
        # selects, it never reorders.
        papers = [make_arxiv_paper(1), make_arxiv_paper(2), make_arxiv_paper(3), make_arxiv_paper(4)]
        scores = [[-9.0], [-2.0], [-12.0], [-4.0]]
        outcome, _ = await run_arxiv_funnel(papers, scores, min_score=-6.0)

        assert [p.arxiv_id for p in outcome.kept] == ["2611.00002", "2611.00004"]
        assert [p.arxiv_id for p in outcome.dropped] == ["2611.00001", "2611.00003"]

    @pytest.mark.asyncio
    async def test_no_top_x_cap_all_above_threshold_are_kept(self) -> None:
        papers = [make_arxiv_paper(i) for i in range(1, 6)]
        scores = [[-1.0], [-2.0], [-3.0], [-4.0], [-5.0]]
        outcome, _ = await run_arxiv_funnel(papers, scores, min_score=-6.0)

        assert [p.arxiv_id for p in outcome.kept] == ["2611.0000" + str(i) for i in range(1, 6)]
        assert outcome.dropped == []

    @pytest.mark.asyncio
    async def test_empty_papers_skips_client(self) -> None:
        stub = ExplodingRerankerClient()
        outcome = await funnel_arxiv_papers([], SECTIONS, stub, min_score=-6.0, report_date="2026-11-12")
        assert outcome.kept == []
        assert outcome.dropped == []
        assert outcome.scored == []

    @pytest.mark.asyncio
    async def test_client_exception_propagates(self) -> None:
        with pytest.raises(RuntimeError, match="reranker exploded"):
            await run_arxiv_funnel([make_arxiv_paper(1)], client=ExplodingRerankerClient(), scores=[])


class TestArxivFunnelScoring:
    """Scored rows, section attribution, and document format."""

    @pytest.mark.asyncio
    async def test_one_scored_row_per_paper_with_winning_section(self) -> None:
        papers = [make_arxiv_paper(1), make_arxiv_paper(2)]
        # Paper 1 wins on Hardware (section 0), paper 2 on Cooking (section 1).
        scores = [[-2.0, -5.0], [-8.0, -3.0]]
        outcome, _ = await run_arxiv_funnel(papers, scores, min_score=-6.0)

        assert outcome.scored == [
            ScoredItem(item_id="2611.00001", subreddit="cs.LG", score=-2.0, section="Hardware", kept=True),
            ScoredItem(item_id="2611.00002", subreddit="cs.LG", score=-3.0, section="Cooking", kept=True),
        ]

    @pytest.mark.asyncio
    async def test_scored_kept_flags_match_selection(self) -> None:
        papers = [make_arxiv_paper(1), make_arxiv_paper(2), make_arxiv_paper(3)]
        scores = [[-1.0], [-9.0], [-3.0]]
        outcome, _ = await run_arxiv_funnel(papers, scores, min_score=-6.0)

        assert [row.item_id for row in outcome.scored] == ["2611.00001", "2611.00002", "2611.00003"]
        assert [row.kept for row in outcome.scored] == [True, False, True]
        assert all(row.section == "Hardware" for row in outcome.scored)

    @pytest.mark.asyncio
    async def test_scored_subreddit_field_carries_primary_category(self) -> None:
        papers = [make_arxiv_paper(1, primary_category="cs.DB")]
        outcome, _ = await run_arxiv_funnel(papers, [[-1.0]], min_score=-6.0)
        assert outcome.scored[0].subreddit == "cs.DB"

    @pytest.mark.asyncio
    async def test_documents_are_title_plus_abstract(self) -> None:
        paper = ArxivPaper(
            arxiv_id="2611.00001",
            abs_url="https://arxiv.org/abs/2611.00001",
            pdf_url="https://arxiv.org/pdf/2611.00001",
            title="Title here",
            abstract="Abstract text",
            authors=["Alice"],
            categories=["cs.LG"],
            primary_category="cs.LG",
            published=datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC),
            updated=datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC),
        )
        outcome, stub = await run_arxiv_funnel([paper], [[-1.0]], min_score=-6.0)
        assert isinstance(stub, StubRerankerClient)

        assert stub.calls[0][1] == ["Title here\n\nAbstract text"]
        assert stub.calls[0][0] == SECTIONS
        assert [p.arxiv_id for p in outcome.kept] == ["2611.00001"]
