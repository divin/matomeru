"""Tests for the reddit collection logic using a duck-typed SagashiClient stub."""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, cast

import pytest

from matomeru.models import (
    RedditComment,
    RedditPost,
    RedditThread,
    SubredditListing,
    ThreadBatch,
    ThreadFetchError,
)
from matomeru.reddit import collect_reddit
from matomeru.sagashi_client import SagashiClient
from matomeru.window import RunWindow

START_TS = 1_700_000_000.0
END_TS = 1_700_008_640.0  # start + 24h

WINDOW = RunWindow(
    start_utc=datetime.fromtimestamp(START_TS, tz=timezone.utc),
    end_utc=datetime.fromtimestamp(END_TS, tz=timezone.utc),
    report_date=date(2023, 11, 14),
    tz_name="UTC",
)


def make_post(pid: str, score: int, created_utc: float, subreddit: str = "python") -> RedditPost:
    return RedditPost(
        id=pid,
        subreddit=subreddit,
        title=f"Title {pid}",
        score=score,
        author="alice",
        created_utc=created_utc,
        num_comments=1,
        url=f"https://example.com/{pid}",
        permalink=f"/r/{subreddit}/comments/{pid}/",
        selftext="",
    )


class StubSagashiClient:
    """Duck-typed stand-in for SagashiClient with scripted responses."""

    def __init__(
        self,
        listings: dict[str, SubredditListing | Exception] | None = None,
        batch: ThreadBatch | None = None,
    ) -> None:
        self.listings = listings or {}
        self.batch = batch or ThreadBatch()
        self.list_calls: list[dict[str, Any]] = []
        self.fetch_calls: list[dict[str, Any]] = []

    async def list_subreddit_posts(
        self,
        subreddit: str,
        sort: str = "top",
        time_filter: str = "all",
        limit: int = 100,
        before: str | None = None,
        after: str | None = None,
        cursor: str | None = None,
    ) -> SubredditListing:
        self.list_calls.append(
            {
                "subreddit": subreddit,
                "sort": sort,
                "time_filter": time_filter,
                "limit": limit,
                "before": before,
                "after": after,
                "cursor": cursor,
            }
        )
        outcome = self.listings[subreddit]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def fetch_reddit_threads(self, items: list[str], comment_count: int = 20) -> ThreadBatch:
        self.fetch_calls.append({"items": list(items), "comment_count": comment_count})
        return self.batch


def make_listing(posts: list[RedditPost]) -> SubredditListing:
    return SubredditListing(post_items=posts, cursor=None)


def as_client(stub: StubSagashiClient) -> SagashiClient:
    """Present the duck-typed stub as a SagashiClient for collect_reddit."""
    return cast(SagashiClient, stub)


@pytest.mark.asyncio
class TestCollectReddit:
    """Tests for collect_reddit."""

    async def test_filters_posts_outside_window(self) -> None:
        in_range = make_post("in", 10, START_TS + 100)
        too_old = make_post("old", 99, START_TS - 1)
        exactly_start = make_post("boundary_start", 99, START_TS)
        too_new = make_post("new", 99, END_TS + 1)
        exactly_end = make_post("boundary_end", 99, END_TS)
        stub = StubSagashiClient(
            listings={"python": make_listing([in_range, too_old, exactly_start, too_new, exactly_end])}
        )
        threads, errors = await collect_reddit(as_client(stub), ["python"], WINDOW)
        assert errors == []
        assert [c["items"] for c in stub.fetch_calls] == [["in"]]
        assert [t.id for t in threads] == []

    async def test_sorts_by_score_and_takes_top_n(self) -> None:
        posts = [
            make_post("low", 1, START_TS + 10),
            make_post("high", 100, START_TS + 20),
            make_post("mid", 50, START_TS + 30),
            make_post("extra", 75, START_TS + 40),
        ]
        stub = StubSagashiClient(listings={"python": make_listing(posts)})
        threads, _ = await collect_reddit(as_client(stub), ["python"], WINDOW, posts_per_subreddit=3)
        assert [c["items"] for c in stub.fetch_calls] == [["high", "extra", "mid"]]

    async def test_dedupes_across_subreddits_first_wins(self) -> None:
        dup = make_post("dup", 50, START_TS + 10, subreddit="r1")
        listing1 = make_listing([dup, make_post("only1", 10, START_TS + 20, subreddit="r1")])
        listing2 = make_listing([dup, make_post("only2", 99, START_TS + 30, subreddit="r2")])
        stub = StubSagashiClient(listings={"r1": listing1, "r2": listing2})
        await collect_reddit(as_client(stub), ["r1", "r2"], WINDOW, posts_per_subreddit=2)
        # First occurrence wins: 'dup' attributed to r1; fetch gets 3 deduped ids.
        assert len(stub.fetch_calls) == 1
        assert stub.fetch_calls[0]["items"] == ["dup", "only1", "only2"]

    async def test_listing_failure_logged_and_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        good = make_listing([make_post("ok", 10, START_TS + 10)])
        stub = StubSagashiClient(listings={"broken": RuntimeError("boom"), "good": good})
        with caplog.at_level(logging.WARNING, logger="matomeru.reddit"):
            threads, errors = await collect_reddit(as_client(stub), ["broken", "good"], WINDOW)
        assert any("broken" in record.message for record in caplog.records)
        assert [c["subreddit"] for c in stub.list_calls] == ["broken", "good"]
        assert [c["items"] for c in stub.fetch_calls] == [["ok"]]
        assert errors == []

    async def test_single_batch_fetch_with_deduped_ids_and_comment_count(self) -> None:
        listing1 = make_listing([make_post("a", 10, START_TS + 1), make_post("b", 9, START_TS + 2)])
        listing2 = make_listing([make_post("c", 8, START_TS + 3)])
        stub = StubSagashiClient(listings={"r1": listing1, "r2": listing2})
        threads, errors = await collect_reddit(as_client(stub), ["r1", "r2"], WINDOW, comments_per_post=7)
        assert len(stub.fetch_calls) == 1
        call = stub.fetch_calls[0]
        assert call["items"] == ["a", "b", "c"]
        assert call["comment_count"] == 7
        assert threads == []
        assert errors == []

    async def test_returns_threads_and_errors_from_batch(self) -> None:
        listing = make_listing([make_post("a", 10, START_TS + 1)])
        batch = ThreadBatch(
            threads=[
                RedditThread(
                    id="a",
                    subreddit="python",
                    title="Title a",
                    score=10,
                    created_utc=START_TS + 1,
                    permalink="/r/python/comments/a/",
                    num_comments=1,
                    comments=[RedditComment(author="bob", score=1, body="hey", created_utc=START_TS + 2)],
                )
            ],
            errors=[ThreadFetchError(index=0, item="ghost", error="missing")],
        )
        stub = StubSagashiClient(listings={"python": listing}, batch=batch)
        threads, errors = await collect_reddit(as_client(stub), ["python"], WINDOW)
        assert len(threads) == 1
        assert threads[0].comments[0].body == "hey"
        assert [(e.index, e.item, e.error) for e in errors] == [(0, "ghost", "missing")]

    async def test_no_posts_skips_fetch(self) -> None:
        stub = StubSagashiClient(listings={"python": make_listing([])})
        threads, errors = await collect_reddit(as_client(stub), ["python"], WINDOW)
        assert stub.fetch_calls == []
        assert (threads, errors) == ([], [])

    async def test_listing_call_uses_expected_arguments(self) -> None:
        stub = StubSagashiClient(listings={"python": make_listing([])})
        await collect_reddit(as_client(stub), ["python"], WINDOW, posts_per_subreddit=5)
        call = stub.list_calls[0]
        assert call["sort"] == "top"
        assert call["time_filter"] == "all"
        assert call["limit"] == 25  # min(100, posts_per_subreddit + 20)
        assert call["after"] == str(START_TS)
        assert call["before"] is None

    async def test_all_subreddits_fail_still_no_fetch(self, caplog: pytest.LogCaptureFixture) -> None:
        stub = StubSagashiClient(listings={"a": RuntimeError("x"), "b": RuntimeError("y")})
        with caplog.at_level(logging.WARNING, logger="matomeru.reddit"):
            threads, errors = await collect_reddit(as_client(stub), ["a", "b"], WINDOW)
        assert stub.fetch_calls == []
        assert (threads, errors) == ([], [])
        assert caplog.text.count("listing failed") == 2
