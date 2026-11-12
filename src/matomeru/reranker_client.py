"""Thin async client for a llama-server ``/v1/rerank`` endpoint.

Connection lifecycle: each method opens its own short-lived connection with
``async with httpx.AsyncClient(...)``. This is the simplest correct option —
the reranker is hit a handful of times per run, so per-call connection setup
cost is negligible, and we avoid holding idle streams between calls.

Request/response contract (llama-server ``/v1/rerank`` semantics):

- The configured URL is the **full endpoint** (e.g.
  ``http://192.168.178.64:9093/v1/rerank``), not a base URL.
- Body: ``{"model": "reranker", "query": ..., "documents": [...]}`` — the
  sidecar serves exactly one model, so the name is fixed.
- Response: ``{"results": [{"index": 0, "relevance_score": -5.6}, ...]}``.
  ``relevance_score`` is the raw cross-encoder logit (mostly negative,
  unbounded) and is stored **as-is** — no sigmoid transformation, so the
  threshold calibration in the funnel operates in logit space.

Integrity is checked fail-early: the response must contain exactly
``len(documents)`` results with indices ``0..len(documents)-1``, each exactly
once. Any deviation raises :class:`RerankerClientError` (a missing score for
a collected item is a response integrity failure, never a silent exclusion —
see the plan's failure semantics).

Tests may inject a fake transport via the ``transport`` argument to run
without any network access.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 120.0


class RerankerClientError(Exception):
    """Raised when a reranker request fails or returns an unexpected payload."""


class RerankerResult(BaseModel):
    """One reranked document: its index into the request document list plus the raw relevance score."""

    index: int
    score: float


class RerankerClient:
    """Async client for a llama-server ``/v1/rerank`` endpoint."""

    def __init__(
        self,
        url: str,
        token: str | None = None,
        *,
        transport: Any | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Create a client.

        Args:
            url: Full rerank endpoint URL, e.g. ``http://host:9093/v1/rerank``.
            token: Optional bearer token sent as an Authorization header.
            transport: Optional transport override (used by tests to plug in a
                fake transport instead of the real HTTP layer).
            timeout: Request timeout in seconds. Generous by default:
                reranking hundreds of documents can take a while.
        """
        self._url = url
        self._token = token
        self._transport_override = transport
        self._timeout = timeout

    @property
    def headers(self) -> dict[str, str]:
        """HTTP headers for the request (Authorization only when a token is set)."""
        if self._token is None:
            return {}
        return {"Authorization": f"Bearer {self._token}"}

    def _make_transport(self) -> httpx.AsyncClient:
        """Build the HTTP transport for one call (honouring the test override)."""
        if self._transport_override is not None:
            return self._transport_override
        return httpx.AsyncClient(timeout=self._timeout)

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST one JSON body and return the parsed JSON object response.

        Raises:
            RerankerClientError: On transport failure, non-200 status, or a
                body that is not a JSON object.
        """
        transport = self._make_transport()
        try:
            response = await transport.post(self._url, headers=self.headers, json=body)
        except httpx.HTTPError as e:
            raise RerankerClientError(f"rerank request failed: {e}") from e

        body_text = response.text
        if response.status_code != 200:
            raise RerankerClientError(f"rerank endpoint returned HTTP {response.status_code}: {body_text[:200]!r}")
        try:
            payload = json.loads(body_text)
        except json.JSONDecodeError as e:
            raise RerankerClientError(f"rerank endpoint returned non-JSON body: {body_text[:200]!r}") from e
        if not isinstance(payload, dict):
            raise RerankerClientError(f"rerank endpoint response is not a JSON object: {body_text[:200]!r}")
        return payload

    async def rerank(self, query: str, documents: list[str]) -> list[RerankerResult]:
        """Rank documents against one query.

        Results are returned in the order the server sent them (the integrity
        check guarantees the indices are a permutation of ``0..n-1``).

        Args:
            query: The query text.
            documents: The documents to rank, in request order.

        Returns:
            One result per document, in server order.

        Raises:
            RerankerClientError: On transport failure, non-200 status, or a
                malformed/incomplete response (missing/duplicate/out-of-range
                index, wrong result count, or missing fields).
        """
        if not documents:
            raise RerankerClientError("rerank requires at least one document")

        payload = await self._post({"model": "reranker", "query": query, "documents": documents})
        results_raw = payload.get("results")
        if not isinstance(results_raw, list):
            raise RerankerClientError(f"rerank response missing 'results' list: {str(payload)[:200]!r}")

        results: list[RerankerResult] = []
        for i, entry in enumerate(results_raw):
            if not isinstance(entry, dict):
                raise RerankerClientError(f"rerank result {i} is not an object: {entry!r}")
            index = entry.get("index")
            if not isinstance(index, int) or isinstance(index, bool):
                raise RerankerClientError(f"rerank result {i} is missing required integer field 'index'")
            if index not in range(len(documents)):
                raise RerankerClientError(f"rerank result {i} has out-of-range index {index!r}")
            if "relevance_score" not in entry:
                raise RerankerClientError(f"rerank result {i} is missing required field 'relevance_score'")
            score = entry["relevance_score"]
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise RerankerClientError(f"rerank result {i} has non-numeric relevance_score: {score!r}")
            results.append(RerankerResult(index=index, score=float(score)))

        seen: set[int] = set()
        for result in results:
            if result.index in seen:
                raise RerankerClientError(f"rerank response integrity: duplicate index {result.index}")
            seen.add(result.index)
        if len(results) != len(documents):
            raise RerankerClientError(
                f"rerank response integrity: expected {len(documents)} results, got {len(results)}"
            )
        return results

    async def rerank_sections_with_argmax(self, sections: list[str], documents: list[str]) -> list[tuple[float, int]]:
        """Score documents against every profile section, keeping the argmax per document.

        Companion to :meth:`rerank_sections` that additionally exposes *which*
        section produced each document's winning score (the funnel records it
        as the winning section name). One rerank call per section (sequential
        — the sidecar processes one batched request per call).

        Args:
            sections: The query texts (one per profile section).
            documents: The documents to rank, in request order.

        Returns:
            One ``(score, section_index)`` pair per document, in document
            order, where ``score`` is the max across sections and
            ``section_index`` is the index of the section that produced it.
            On ties the **first** section (lowest index) wins.

        Raises:
            RerankerClientError: On transport failure, non-200 status, or a
                malformed/incomplete response in any section call.
        """
        if not sections:
            raise RerankerClientError("rerank_sections requires at least one section")
        best: list[float] = [float("-inf")] * len(documents)
        best_section: list[int] = [0] * len(documents)
        for section_idx, section in enumerate(sections):
            results = await self.rerank(section, documents)
            for result in results:
                if result.score > best[result.index]:
                    best[result.index] = result.score
                    best_section[result.index] = section_idx
        return list(zip(best, best_section, strict=True))

    async def rerank_sections(self, sections: list[str], documents: list[str]) -> list[float]:
        """Score documents against every profile section, taking the max per document.

        One rerank call per section (sequential — the sidecar processes one
        batched request per call), so the returned score for document ``d`` is
        ``max(score(section, documents[d]))`` across all sections. This is the
        per-section scoring design from the plan: a single whole-profile query
        produced flat separation, while per-section queries separate cleanly.

        Args:
            sections: The query texts (one per profile section).
            documents: The documents to rank, in request order.

        Returns:
            One score per document, in document order.

        Raises:
            RerankerClientError: On transport failure, non-200 status, or a
                malformed/incomplete response in any section call.
        """
        pairs = await self.rerank_sections_with_argmax(sections, documents)
        return [score for score, _section_idx in pairs]
