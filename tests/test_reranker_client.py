"""Tests for the RerankerClient (all network access via a fake transport)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from matomeru.reranker_client import RerankerClient, RerankerClientError


def make_response(payload: dict[str, Any], status_code: int = 200) -> httpx.Response:
    """Build a httpx.Response with the given JSON payload."""
    return httpx.Response(status_code=status_code, json=payload)


class FakeTransport:
    """A minimal fake transport that records the last request and returns canned responses."""

    def __init__(self, payload: dict[str, Any] | None = None, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.last_request: dict[str, Any] = {}
        self.call_count = 0
        self.last_headers: dict[str, str] = {}

    async def post(
        self, url: str, *, json: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> httpx.Response:
        self.last_request = {"url": url, "json": json}
        self.last_headers = dict(headers or {})
        self.call_count += 1
        if self.status_code != 200:
            return httpx.Response(status_code=self.status_code, text="reranker unavailable")
        return make_response(self.payload or {})


def make_client(transport: FakeTransport) -> RerankerClient:
    return RerankerClient("http://reranker.invalid/v1/rerank", transport=transport)


def make_payload(scores: list[float]) -> dict[str, Any]:
    """A well-formed rerank response with the given scores."""
    return {"results": [{"index": i, "relevance_score": s} for i, s in enumerate(scores)]}


@pytest.mark.asyncio
async def test_rerank_happy_path() -> None:
    """Scores are mapped from relevance_score and order is preserved."""
    transport = FakeTransport(payload=make_payload([-5.6, -1.2, 0.0]))
    client = make_client(transport)

    results = await client.rerank("query", ["doc a", "doc b", "doc c"])

    assert [r.score for r in results] == [-5.6, -1.2, 0.0]
    assert [r.index for r in results] == [0, 1, 2]
    assert transport.last_request["json"] == {
        "model": "reranker",
        "query": "query",
        "documents": ["doc a", "doc b", "doc c"],
    }


@pytest.mark.asyncio
async def test_rerank_sends_model_reranker_and_query() -> None:
    """The request body is {model, query, documents}."""
    transport = FakeTransport(payload=make_payload([-1.0]))
    client = make_client(transport)

    await client.rerank("query", ["doc a"])

    assert transport.last_request["json"] == {"model": "reranker", "query": "query", "documents": ["doc a"]}


@pytest.mark.asyncio
async def test_rerank_no_token_no_authorization_header() -> None:
    """No token means no Authorization header is sent."""
    transport = FakeTransport(payload=make_payload([-1.0]))
    client = make_client(transport)

    await client.rerank("query", ["doc a"])

    assert "Authorization" not in transport.last_headers


@pytest.mark.asyncio
async def test_rerank_token_sends_authorization_header() -> None:
    """A token sends the Authorization header."""
    transport = FakeTransport(payload=make_payload([-1.0]))
    client = RerankerClient("http://reranker.invalid/v1/rerank", token="secret", transport=transport)

    await client.rerank("query", ["doc a"])

    assert transport.last_headers["Authorization"] == "Bearer secret"


@pytest.mark.asyncio
async def test_rerank_missing_index_raises() -> None:
    """A missing index raises RerankerClientError."""
    transport = FakeTransport(
        payload={"results": [{"index": 0, "relevance_score": -1.0}, {"index": 2, "relevance_score": -1.0}]}
    )
    client = make_client(transport)

    with pytest.raises(RerankerClientError, match="expected 3 results, got 2"):
        await client.rerank("query", ["doc a", "doc b", "doc c"])


@pytest.mark.asyncio
async def test_rerank_duplicate_index_raises() -> None:
    """A duplicate index raises RerankerClientError."""
    transport = FakeTransport(
        payload={"results": [{"index": 0, "relevance_score": -1.0}, {"index": 0, "relevance_score": -1.0}]}
    )
    client = make_client(transport)

    with pytest.raises(RerankerClientError, match="duplicate index"):
        await client.rerank("query", ["doc a", "doc b", "doc c"])


@pytest.mark.asyncio
async def test_rerank_wrong_count_raises() -> None:
    """Too few results raises RerankerClientError."""
    transport = FakeTransport(payload=make_payload([-1.0, -1.0]))
    client = make_client(transport)

    with pytest.raises(RerankerClientError, match="expected 3 results, got 2"):
        await client.rerank("query", ["doc a", "doc b", "doc c"])


@pytest.mark.asyncio
async def test_rerank_non_200_raises() -> None:
    """A non-200 response raises RerankerClientError including the response body."""
    transport = FakeTransport(payload={}, status_code=500)
    client = make_client(transport)

    with pytest.raises(RerankerClientError, match="reranker unavailable"):
        await client.rerank("query", ["doc a"])


@pytest.mark.asyncio
async def test_rerank_network_error_raises() -> None:
    """A network error raises RerankerClientError."""
    transport = FakeTransport(payload=make_payload([-1.0]))
    client = make_client(transport)

    async def failing_post(
        url: str, *, json: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport.post = failing_post  # type: ignore[method-assign]

    with pytest.raises(RerankerClientError, match="connection refused"):
        await client.rerank("query", ["doc a"])


@pytest.mark.asyncio
async def test_rerank_sections_max_per_document() -> None:
    """rerank_sections returns the max score per document across sections."""
    transport = FakeTransport(payload={})
    client = make_client(transport)

    section_payloads = [
        make_payload([-5.0, -1.0]),
        make_payload([-2.0, -3.0]),
    ]
    payloads = iter(section_payloads)

    async def post(
        url: str, *, json: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> httpx.Response:
        return make_response(next(payloads))

    transport.post = post  # type: ignore[method-assign]

    scores = await client.rerank_sections(["section one", "section two"], ["doc a", "doc b"])

    assert scores == [-2.0, -1.0]


@pytest.mark.asyncio
async def test_rerank_sections_uses_section_query() -> None:
    """rerank_sections sends the section text as the query."""
    transport = FakeTransport(payload=make_payload([-1.0]))
    client = make_client(transport)

    await client.rerank_sections(["section one"], ["doc a"])

    assert transport.last_request["json"] == {"model": "reranker", "query": "section one", "documents": ["doc a"]}


@pytest.mark.asyncio
async def test_rerank_sections_no_token_no_authorization_header() -> None:
    """No token means no Authorization header is sent."""
    transport = FakeTransport(payload=make_payload([-1.0]))
    client = make_client(transport)

    await client.rerank_sections(["section one"], ["doc a"])

    assert "Authorization" not in transport.last_headers


@pytest.mark.asyncio
async def test_rerank_sections_token_sends_authorization_header() -> None:
    """A token sends the Authorization header."""
    transport = FakeTransport(payload=make_payload([-1.0]))
    client = RerankerClient("http://reranker.invalid/v1/rerank", token="secret", transport=transport)

    await client.rerank_sections(["section one"], ["doc a"])

    assert transport.last_headers["Authorization"] == "Bearer secret"


class SequencedTransport:
    """A fake transport that returns a different canned payload per call."""

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self._payloads = payloads
        self.call_count = 0

    async def post(
        self, url: str, *, json: dict[str, Any] | None = None, headers: dict[str, str] | None = None
    ) -> httpx.Response:
        payload = self._payloads[self.call_count]
        self.call_count += 1
        return make_response(payload)


@pytest.mark.asyncio
async def test_rerank_sections_with_argmax_returns_winning_section() -> None:
    """Each document gets (max score, index of the winning section)."""
    transport = SequencedTransport([make_payload([-9.0, -5.0]), make_payload([-5.6, -1.2])])
    client = RerankerClient("http://reranker.invalid/v1/rerank", transport=transport)

    pairs = await client.rerank_sections_with_argmax(["section zero", "section one"], ["doc a", "doc b"])

    assert pairs == [(-5.6, 1), (-1.2, 1)]
    assert transport.call_count == 2


@pytest.mark.asyncio
async def test_rerank_sections_with_argmax_tie_prefers_first_section() -> None:
    """On an exact tie the first (lowest-index) section wins."""
    transport = SequencedTransport([make_payload([-3.0]), make_payload([-3.0])])
    client = RerankerClient("http://reranker.invalid/v1/rerank", transport=transport)

    pairs = await client.rerank_sections_with_argmax(["section zero", "section one"], ["doc a"])

    assert pairs == [(-3.0, 0)]


@pytest.mark.asyncio
async def test_rerank_sections_still_returns_flat_scores() -> None:
    """The 7a public behavior is unchanged: one plain score per document."""
    transport = SequencedTransport([make_payload([-9.0, -5.0]), make_payload([-5.6, -1.2])])
    client = RerankerClient("http://reranker.invalid/v1/rerank", transport=transport)

    scores = await client.rerank_sections(["section zero", "section one"], ["doc a", "doc b"])

    assert scores == [-5.6, -1.2]


@pytest.mark.asyncio
async def test_rerank_sections_with_argmax_requires_a_section() -> None:
    """Empty sections raise the same error as rerank_sections."""
    transport = SequencedTransport([])
    client = RerankerClient("http://reranker.invalid/v1/rerank", transport=transport)

    with pytest.raises(RerankerClientError, match="requires at least one section"):
        await client.rerank_sections_with_argmax([], ["doc a"])
