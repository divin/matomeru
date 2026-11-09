"""Tests for the pydantic models used to parse Sagashi payloads."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from matomeru.models import (
    RedditComment,
    RedditPost,
    RedditThread,
    SubredditListing,
    ThreadBatch,
    ThreadFetchError,
)


class TestRedditPost:
    """Tests for RedditPost parsing."""

    def test_full_post_parses(self) -> None:
        post = RedditPost(
            id="abc123",
            subreddit="python",
            title="Hello",
            score=42,
            author="alice",
            created_utc=1_700_000_000.5,
            num_comments=7,
            url="https://example.com/x",
            permalink="/r/python/comments/abc123/hello/",
            selftext="Body text",
        )
        assert post.id == "abc123"
        assert post.subreddit == "python"
        assert post.title == "Hello"
        assert post.score == 42
        assert post.author == "alice"
        assert post.created_utc == 1_700_000_000.5
        assert post.num_comments == 7
        assert post.url == "https://example.com/x"
        assert post.permalink == "/r/python/comments/abc123/hello/"
        assert post.selftext == "Body text"

    def test_missing_optional_fields_get_defaults(self) -> None:
        post = RedditPost(id="abc123")  # only id supplied; everything else defaults
        assert post.subreddit == ""
        assert post.title == ""
        assert post.score == 0
        assert post.author is None
        assert post.created_utc == 0.0
        assert post.num_comments == 0
        assert post.url == ""
        assert post.permalink == ""
        assert post.selftext == ""

    def test_id_is_required(self) -> None:
        with pytest.raises(ValidationError):
            RedditPost.model_validate({})

    def test_parses_from_sagashi_style_dict(self) -> None:
        raw = {
            "id": "t3_abc",
            "subreddit": "python",
            "title": "t",
            "score": 5,
            "created_utc": 1_700_000_000.0,
            "permalink": "/r/python/comments/abc/",
        }
        post = RedditPost.model_validate(raw)
        assert post.author is None
        assert post.num_comments == 0
        assert post.selftext == ""


class TestRedditComment:
    """Tests for RedditComment parsing."""

    def test_full_comment(self) -> None:
        comment = RedditComment(author="bob", score=3, body="Nice", created_utc=1_700_000_100.0)
        assert comment.author == "bob"
        assert comment.score == 3
        assert comment.body == "Nice"
        assert comment.created_utc == 1_700_000_100.0

    def test_defaults(self) -> None:
        comment = RedditComment()
        assert comment.author is None
        assert comment.score == 0
        assert comment.body == ""
        assert comment.created_utc == 0.0


class TestRedditThread:
    """Tests for RedditThread parsing."""

    def test_thread_with_comments(self) -> None:
        thread = RedditThread(
            id="t3_abc",
            subreddit="python",
            title="Hello",
            score=10,
            created_utc=1_700_000_000.0,
            permalink="/r/python/comments/abc/",
            selftext="text",
            num_comments=2,
            comments=[
                {"author": "a", "score": 1, "body": "one", "created_utc": 1.0},
                {"author": None, "score": 0, "body": "two", "created_utc": 2.0},
            ],
        )
        assert thread.num_comments == 2
        assert len(thread.comments) == 2
        assert thread.comments[0].author == "a"
        assert thread.comments[1].body == "two"

    def test_comments_default_empty(self) -> None:
        thread = RedditThread(id="t3_abc")
        assert thread.comments == []


class TestThreadFetchError:
    """Tests for ThreadFetchError parsing."""

    def test_fields(self) -> None:
        err = ThreadFetchError(index=2, item="t3_x", error="boom")
        assert (err.index, err.item, err.error) == (2, "t3_x", "boom")

    def test_defaults(self) -> None:
        err = ThreadFetchError()
        assert (err.index, err.item, err.error) == (0, "", "")


class TestContainers:
    """Tests for SubredditListing and ThreadBatch."""

    def test_subreddit_listing(self) -> None:
        listing = SubredditListing(
            post_items=[RedditPost(id="1"), RedditPost(id="2")],
            cursor="next-page",
        )
        assert [p.id for p in listing.post_items] == ["1", "2"]
        assert listing.cursor == "next-page"

    def test_subreddit_listing_defaults(self) -> None:
        listing = SubredditListing()
        assert listing.post_items == []
        assert listing.cursor is None

    def test_thread_batch(self) -> None:
        batch = ThreadBatch(
            threads=[RedditThread(id="t1")],
            errors=[ThreadFetchError(index=0, item="t2", error="gone")],
        )
        assert len(batch.threads) == 1
        assert len(batch.errors) == 1

    def test_thread_batch_defaults(self) -> None:
        batch = ThreadBatch()
        assert batch.threads == []
        assert batch.errors == []
