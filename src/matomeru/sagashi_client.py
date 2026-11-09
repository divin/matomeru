"""Thin async client for the Sagashi MCP server.

Connection lifecycle: each method opens its own short-lived connection with
``async with Client(...)``. This is the simplest correct option — the Sagashi
server is only hit a handful of times per run, so per-call connection setup
cost is negligible, and we avoid holding idle streams between calls.

Tool-call/response mapping assumptions (verified against Sagashi's contract):

- ``list_subreddit_posts`` accepts ``subreddit``, ``sort``, ``time_filter``,
  ``limit``, ``before``, ``after``, ``cursor`` and returns
  ``{"posts": [...], "pagination": {"cursor": ...}, "metadata": {...}}``.
  Each post dict maps into :class:`~matomeru.models.RedditPost`.
- ``fetch_reddit_threads`` accepts ``items`` (list of ids/urls) and
  ``comment_count`` and returns
  ``{"results": [...], "errors": [...], "metadata": {...}}``. Results map into
  :class:`~matomeru.models.RedditThread` (with ``comments`` mapped to
  :class:`~matomeru.models.RedditComment`); errors map into
  :class:`~matomeru.models.ThreadFetchError` (dicts are expected to carry
  ``item`` and ``error``/``message`` plus optional ``index``; bare strings are
  treated as the error message with an unknown item).
- Response payloads are read from structured content when the server provides
  it, otherwise parsed from the JSON text content blocks.

Missing *optional* post/thread fields (score, author, created_utc, ...) are
tolerated with defaults, but a post or thread without an ``id`` or
``permalink`` is never padded with invented values — the whole call raises
:class:`SagashiClientError` instead.

Tests may inject an in-memory transport via the ``transport`` argument to run
against a fake FastMCP server without any network access.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import TextContent

from matomeru.models import (
    RedditComment,
    RedditPost,
    RedditThread,
    SubredditListing,
    ThreadBatch,
    ThreadFetchError,
)

logger = logging.getLogger(__name__)

LIST_POSTS_TOOL = "list_subreddit_posts"
FETCH_THREADS_TOOL = "fetch_reddit_threads"


class SagashiClientError(Exception):
    """Raised when a Sagashi MCP call fails or returns an unexpected payload."""


def _unwrap_payload(result: Any) -> dict[str, Any]:
    """Extract a JSON object from an MCP tool result.

    Handles both fastmcp ``CallToolResult`` objects and raw content lists.
    Prefers structured content, falling back to parsing JSON text blocks.
    """
    content = getattr(result, "content", None)
    if content is None:
        content = result
    is_error = getattr(result, "is_error", False)
    if is_error:
        raise SagashiClientError(f"Sagashi tool call returned an error: {content!r:.200}")

    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured

    for block in content or []:
        if isinstance(block, TextContent):
            try:
                payload = json.loads(block.text)
            except json.JSONDecodeError as e:
                raise SagashiClientError(f"Sagashi returned non-JSON text: {block.text[:200]!r}") from e
            if not isinstance(payload, dict):
                raise SagashiClientError("Sagashi tool response is not a JSON object")
            return payload

    raise SagashiClientError("Sagashi tool response contained no usable payload")


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    """Fetch a required field, raising SagashiClientError if absent or None."""
    value = mapping.get(key)
    if value is None:
        raise SagashiClientError(f"{context} is missing required field {key!r}")
    return value


def _map_post(post: dict[str, Any]) -> RedditPost:
    """Map one Sagashi post dict to a RedditPost.

    id/permalink must be present; every other field tolerates absence.
    """
    return RedditPost(
        id=str(_required(post, "id", "post")),
        permalink=str(_required(post, "permalink", "post")),
        subreddit=str(post.get("subreddit") or ""),
        title=str(post.get("title") or ""),
        score=int(post.get("score") or 0),
        author=post.get("author"),
        created_utc=float(post.get("created_utc") or 0.0),
        num_comments=int(post.get("num_comments") or 0),
        url=str(post.get("url") or ""),
        selftext=str(post.get("selftext") or ""),
    )


def _map_comment(comment: dict[str, Any]) -> RedditComment:
    """Map one Sagashi comment dict to a RedditComment (all fields tolerated)."""
    return RedditComment(
        author=comment.get("author"),
        score=int(comment.get("score") or 0),
        body=str(comment.get("body") or ""),
        created_utc=float(comment.get("created_utc") or 0.0),
    )


def _map_thread(thread: dict[str, Any]) -> RedditThread:
    """Map one Sagashi thread dict to a RedditThread.

    id/permalink must be present; every other field tolerates absence.
    """
    comments_raw = thread.get("comments") or []
    return RedditThread(
        id=str(_required(thread, "id", "thread")),
        permalink=str(_required(thread, "permalink", "thread")),
        subreddit=str(thread.get("subreddit") or ""),
        title=str(thread.get("title") or ""),
        score=int(thread.get("score") or 0),
        created_utc=float(thread.get("created_utc") or 0.0),
        selftext=str(thread.get("selftext") or ""),
        num_comments=int(thread.get("num_comments") or 0),
        comments=[_map_comment(c) for c in comments_raw if isinstance(c, dict)],
    )


def _map_fetch_error(raw: Any, index: int) -> ThreadFetchError:
    """Map one Sagashi error entry to a ThreadFetchError."""
    if isinstance(raw, dict):
        return ThreadFetchError(
            index=int(raw.get("index", index)),
            item=str(raw.get("item") or raw.get("id") or ""),
            error=str(raw.get("error") or raw.get("message") or ""),
        )
    return ThreadFetchError(index=index, item="", error=str(raw))


class SagashiClient:
    """Async client for the Sagashi MCP server's Reddit tools."""

    def __init__(
        self,
        base_url: str,
        bearer_token: str | None = None,
        *,
        transport: Any | None = None,
    ) -> None:
        """Create a client.

        Args:
            base_url: Base URL of the Sagashi MCP server (streamable HTTP).
            bearer_token: Optional bearer token sent as an Authorization header.
            transport: Optional transport override (used by tests to plug in an
                in-memory FastMCP server instead of the HTTP transport).
        """
        self._base_url = base_url
        self._bearer_token = bearer_token
        self._transport_override = transport

    @property
    def base_url(self) -> str:
        """Base URL of the Sagashi MCP server."""
        return self._base_url

    @property
    def headers(self) -> dict[str, str] | None:
        """HTTP headers applied to the streamable HTTP transport."""
        if self._bearer_token is None:
            return None
        return {"Authorization": f"Bearer {self._bearer_token}"}

    def _make_transport(self) -> Any:
        """Build the transport for a call (honouring the test override)."""
        if self._transport_override is not None:
            return self._transport_override
        return StreamableHttpTransport(url=self._base_url, headers=self.headers)

    async def _call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke one Sagashi tool and return its JSON payload.

        Opens a short-lived connection per call (see module docstring).
        """
        try:
            async with Client(self._make_transport()) as client:
                result = await client.call_tool(tool, arguments)
        except SagashiClientError:
            raise
        except Exception as e:
            raise SagashiClientError(f"MCP call {tool!r} failed: {e}") from e
        return _unwrap_payload(result)

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
        """List posts from one subreddit.

        Sagashi's ``after``/``before`` are exclusive ``created_utc`` bounds in
        epoch seconds.

        Raises:
            SagashiClientError: On transport/protocol failure, malformed
                payloads, or posts lacking ``id``/``permalink``.
        """
        arguments: dict[str, Any] = {
            "subreddit": subreddit,
            "sort": sort,
            "time_filter": time_filter,
            "limit": limit,
        }
        if before is not None:
            arguments["before"] = before
        if after is not None:
            arguments["after"] = after
        if cursor is not None:
            arguments["cursor"] = cursor

        payload = await self._call(LIST_POSTS_TOOL, arguments)
        posts_raw = payload.get("posts")
        if posts_raw is None:
            raise SagashiClientError("list_subreddit_posts response missing 'posts'")
        if not isinstance(posts_raw, list):
            raise SagashiClientError("list_subreddit_posts 'posts' is not a list")

        pagination = payload.get("pagination")
        cursor_value = pagination.get("cursor") if isinstance(pagination, dict) else None
        try:
            posts = [_map_post(p) for p in posts_raw if isinstance(p, dict)]
        except (KeyError, TypeError, ValueError) as e:
            raise SagashiClientError(f"Malformed post in list_subreddit_posts response: {e}") from e
        return SubredditListing(post_items=posts, cursor=cursor_value if isinstance(cursor_value, str) else None)

    async def fetch_reddit_threads(self, items: list[str], comment_count: int = 20) -> ThreadBatch:
        """Fetch multiple threads (with comments) in one batch call.

        Raises:
            SagashiClientError: On transport/protocol failure or malformed
                payloads (threads lacking ``id``/``permalink`` included).
        """
        arguments: dict[str, Any] = {"items": list(items), "comment_count": comment_count}
        payload = await self._call(FETCH_THREADS_TOOL, arguments)

        results_raw = payload.get("results")
        errors_raw = payload.get("errors")
        if results_raw is None or errors_raw is None:
            raise SagashiClientError("fetch_reddit_threads response missing 'results'/'errors'")
        if not isinstance(results_raw, list) or not isinstance(errors_raw, list):
            raise SagashiClientError("fetch_reddit_threads 'results'/'errors' are not lists")

        try:
            threads = [_map_thread(t) for t in results_raw if isinstance(t, dict)]
        except (KeyError, TypeError, ValueError) as e:
            raise SagashiClientError(f"Malformed thread in fetch_reddit_threads response: {e}") from e
        errors = [_map_fetch_error(e, i) for i, e in enumerate(errors_raw)]
        return ThreadBatch(threads=threads, errors=errors)
