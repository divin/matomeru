"""Pydantic data models for Reddit and arXiv items."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class RedditPost(BaseModel):
    """A single Reddit post as returned by Sagashi's listing tools."""

    id: str
    subreddit: str = ""
    title: str = ""
    score: int = 0
    author: str | None = None
    created_utc: float = 0.0
    num_comments: int = 0
    url: str = ""
    permalink: str = ""
    selftext: str = ""


class RedditComment(BaseModel):
    """A single comment on a Reddit thread."""

    author: str | None = None
    score: int = 0
    body: str = ""
    created_utc: float = 0.0


class RedditThread(BaseModel):
    """A Reddit submission with its fetched comments."""

    id: str
    subreddit: str = ""
    title: str = ""
    score: int = 0
    author: str | None = None
    created_utc: float = 0.0
    permalink: str = ""
    selftext: str = ""
    num_comments: int = 0
    comments: list[RedditComment] = Field(default_factory=list)


class ThreadFetchError(BaseModel):
    """An error fetching one thread in a batch request."""

    index: int = 0
    item: str = ""
    error: str = ""


class SubredditListing(BaseModel):
    """A page of posts from one subreddit."""

    post_items: list[RedditPost] = Field(default_factory=list)
    cursor: str | None = None


class ThreadBatch(BaseModel):
    """Result of a batch thread fetch: fetched threads plus per-item errors."""

    threads: list[RedditThread] = Field(default_factory=list)
    errors: list[ThreadFetchError] = Field(default_factory=list)


class ArxivPaper(BaseModel):
    """A single arXiv paper as collected from the arXiv RSS announcement feed.

    For RSS-collected papers, ``published`` and ``updated`` both carry the
    feed's announcement stamp (the RFC 2822 pubDate converted to UTC), not
    the paper's original submission or revision timestamps.
    """

    arxiv_id: str
    abs_url: str
    pdf_url: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: datetime
    updated: datetime
    announcement_hint: str | None = None


class ArxivCandidateBatch(BaseModel):
    """A batch of arXiv candidate papers collected in one run."""

    papers: list[ArxivPaper] = Field(default_factory=list)
