"""Tests for SagashiClient against an in-memory fake FastMCP server.

No network and no real MCP server: the fake server is served over fastmcp's
in-memory transport and exposes the same tools with canned payloads.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastmcp import FastMCP

from matomeru.sagashi_client import SagashiClient, SagashiClientError

POST_A = {
    "id": "t3_a",
    "subreddit": "python",
    "title": "Post A",
    "score": 100,
    "author": "alice",
    "created_utc": 1_700_000_000.0,
    "num_comments": 4,
    "url": "https://example.com/a",
    "permalink": "/r/python/comments/a/",
    "selftext": "alpha",
}

POST_B = {
    "id": "t3_b",
    "subreddit": "python",
    "title": "Post B",
    "score": 50,
    "created_utc": 1_700_000_100.0,
    "permalink": "/r/python/comments/b/",
    "selftext": "",
}

THREAD_A = {
    "id": "t3_a",
    "subreddit": "python",
    "title": "Post A",
    "score": 100,
    "created_utc": 1_700_000_000.0,
    "permalink": "/r/python/comments/a/",
    "selftext": "alpha",
    "num_comments": 1,
    "comments": [{"author": "bob", "score": 2, "body": "nice", "created_utc": 1_700_000_050.0}],
}


def make_fake_server(
    list_payload: dict[str, Any] | None = None,
    threads_payload: dict[str, Any] | None = None,
    list_impl: Any = None,
    threads_impl: Any = None,
) -> FastMCP:
    """Build a minimal fake Sagashi MCP server for in-memory testing."""
    server = FastMCP("sagashi-fake")

    list_fn = list_impl or (lambda subreddit, sort, time_filter, limit, before, after, cursor: list_payload or {})
    threads_fn = threads_impl or (lambda items, comment_count: threads_payload or {})

    @server.tool
    def list_subreddit_posts(
        subreddit: str,
        sort: str = "top",
        time_filter: str = "all",
        limit: int = 100,
        before: str | None = None,
        after: str | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        return list_fn(subreddit, sort, time_filter, limit, before, after, cursor)

    @server.tool
    def fetch_reddit_threads(items: list[str], comment_count: int = 20) -> dict[str, Any]:
        return threads_fn(items, comment_count)

    return server


def make_client(**kwargs: Any) -> SagashiClient:
    """Create a SagashiClient wired to an in-memory fake server."""
    return SagashiClient("http://sagashi.invalid", transport=make_fake_server(**kwargs))


@pytest.mark.asyncio
class TestListSubredditPosts:
    """Tests for list_subreddit_posts mapping."""

    async def test_maps_posts_and_cursor(self) -> None:
        client = make_client(list_payload={"posts": [POST_A, POST_B], "pagination": {"cursor": "page-2"}})
        listing = await client.list_subreddit_posts("python")
        assert [p.id for p in listing.post_items] == ["t3_a", "t3_b"]
        assert listing.post_items[0].score == 100
        assert listing.post_items[0].author == "alice"
        assert listing.cursor == "page-2"

    async def test_tolerates_missing_optional_fields(self) -> None:
        client = make_client(list_payload={"posts": [POST_B], "pagination": {}})
        listing = await client.list_subreddit_posts("python")
        post = listing.post_items[0]
        assert post.author is None
        assert post.num_comments == 0
        assert post.url == ""
        assert listing.cursor is None

    async def test_missing_id_raises(self) -> None:
        bad = {**POST_A, "id": None}
        client = make_client(list_payload={"posts": [bad], "pagination": {}})
        with pytest.raises(SagashiClientError, match="missing required field 'id'"):
            await client.list_subreddit_posts("python")

    async def test_missing_permalink_raises(self) -> None:
        bad = {**POST_A, "permalink": None}
        client = make_client(list_payload={"posts": [bad], "pagination": {}})
        with pytest.raises(SagashiClientError, match="missing required field 'permalink'"):
            await client.list_subreddit_posts("python")

    async def test_forwards_arguments(self) -> None:
        seen: dict[str, Any] = {}

        def list_impl(subreddit: str, sort: str, time_filter: str, limit: int, before: Any, after: Any, cursor: Any):
            seen.update(
                subreddit=subreddit,
                sort=sort,
                time_filter=time_filter,
                limit=limit,
                before=before,
                after=after,
                cursor=cursor,
            )
            return {"posts": [], "pagination": {}}

        client = make_client(list_impl=list_impl)
        await client.list_subreddit_posts("python", limit=10, after="1000.0", cursor="c1")
        assert seen == {
            "subreddit": "python",
            "sort": "top",
            "time_filter": "all",
            "limit": 10,
            "before": None,
            "after": "1000.0",
            "cursor": "c1",
        }

    async def test_missing_posts_key_raises(self) -> None:
        client = make_client(list_payload={"unexpected": True})
        with pytest.raises(SagashiClientError, match="missing 'posts'"):
            await client.list_subreddit_posts("python")


@pytest.mark.asyncio
class TestFetchRedditThreads:
    """Tests for fetch_reddit_threads mapping."""

    async def test_maps_threads_and_comments(self) -> None:
        payload = {
            "results": [THREAD_A],
            "errors": [],
            "metadata": {"count": 1},
        }
        client = make_client(threads_payload=payload)
        batch = await client.fetch_reddit_threads(["t3_a"])
        assert len(batch.threads) == 1
        thread = batch.threads[0]
        assert thread.id == "t3_a"
        assert thread.score == 100
        assert thread.permalink == "/r/python/comments/a/"
        assert len(thread.comments) == 1
        assert thread.comments[0].author == "bob"
        assert thread.comments[0].body == "nice"
        assert batch.errors == []

    async def test_maps_errors(self) -> None:
        payload = {
            "results": [],
            "errors": [
                {"item": "t3_x", "error": "not found"},
                {"item": "t3_y", "error": "rate limited", "index": 5},
            ],
        }
        client = make_client(threads_payload=payload)
        batch = await client.fetch_reddit_threads(["t3_x", "t3_y"])
        assert batch.threads == []
        assert [(e.index, e.item, e.error) for e in batch.errors] == [
            (0, "t3_x", "not found"),
            (5, "t3_y", "rate limited"),
        ]

    async def test_error_string_entry_falls_back(self) -> None:
        payload = {"results": [], "errors": ["kaput"]}
        client = make_client(threads_payload=payload)
        batch = await client.fetch_reddit_threads(["t3_z"])
        assert batch.errors[0].error == "kaput"
        assert batch.errors[0].item == ""

    async def test_missing_thread_id_raises(self) -> None:
        bad = {**THREAD_A, "id": None}
        client = make_client(threads_payload={"results": [bad], "errors": []})
        with pytest.raises(SagashiClientError, match="missing required field 'id'"):
            await client.fetch_reddit_threads(["t3_a"])

    async def test_forwards_items_and_comment_count(self) -> None:
        seen: dict[str, Any] = {}

        def threads_impl(items: list[str], comment_count: int) -> dict[str, Any]:
            seen.update(items=items, comment_count=comment_count)
            return {"results": [], "errors": []}

        client = make_client(threads_impl=threads_impl)
        await client.fetch_reddit_threads(["a", "b"], comment_count=7)
        assert seen == {"items": ["a", "b"], "comment_count": 7}


@pytest.mark.asyncio
class TestErrorWrapping:
    """Tests for SagashiClientError wrapping of arbitrary failures."""

    async def test_tool_exception_wrapped(self) -> None:
        def boom(*args: Any) -> dict[str, Any]:
            raise RuntimeError("kaboom")

        client = make_client(list_impl=boom)
        with pytest.raises(SagashiClientError, match="MCP call 'list_subreddit_posts' failed"):
            await client.list_subreddit_posts("python")

    async def test_non_json_response_wrapped(self) -> None:
        server = FastMCP("sagashi-fake")

        @server.tool
        def list_subreddit_posts(subreddit: str) -> str:
            """Return a non-JSON text payload."""
            return "not json"

        client = SagashiClient("http://sagashi.invalid", transport=server)
        with pytest.raises(SagashiClientError):
            await client.list_subreddit_posts("python")


class TestTransportAndHeaders:
    """Tests for transport construction and bearer-token headers."""

    def test_headers_without_token(self) -> None:
        client = SagashiClient("http://sagashi.invalid")
        assert client.headers is None

    def test_headers_with_token(self) -> None:
        client = SagashiClient("http://sagashi.invalid", bearer_token="sekrit")
        assert client.headers == {"Authorization": "Bearer sekrit"}

    def test_transport_override_is_used(self) -> None:
        server = FastMCP("sagashi-fake")
        client = SagashiClient("http://sagashi.invalid", transport=server)
        assert client._make_transport() is server

    def test_base_url_preserved(self) -> None:
        client = SagashiClient("http://example.invalid/mcp")
        assert client.base_url == "http://example.invalid/mcp"
