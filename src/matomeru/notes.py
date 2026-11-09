"""Async client for the Notes MCP server's note-writing tools.

Mirrors :mod:`matomeru.sagashi_client` conventions: each method opens its
own short-lived connection with ``async with Client(...)``, a streamable
HTTP transport carries an Authorization header when a bearer token is set,
and tests may inject an in-memory transport via the ``transport`` argument.

Only writing is exposed here (``create_folder`` / ``create_note``); reading
is out of scope for Matomeru. Errors are wrapped in
:class:`NotesClientError`, except that "already exists"-style failures from
``create_folder`` are treated as success (folder creation is idempotent).
"""

from __future__ import annotations

import logging
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import TextContent

logger = logging.getLogger(__name__)

CREATE_FOLDER_TOOL = "create_folder"
CREATE_NOTE_TOOL = "create_note"


class NotesClientError(Exception):
    """Raised when a Notes MCP call fails or returns an unexpected payload."""


class _FolderAlreadyExists(Exception):
    """Internal sentinel: the folder already exists (treated as success)."""


def _is_exists_error(text: str) -> bool:
    """Whether an error message reads like an 'already exists' failure."""
    return "exist" in text.lower()


def _result_text(result: Any) -> str:
    """Flatten an MCP tool result's text blocks into one string."""
    content = getattr(result, "content", None)
    if content is None:
        content = result
    return "\n".join(block.text for block in (content or []) if isinstance(block, TextContent))


class NotesMcpClient:
    """Async client for the Notes MCP server's writing tools."""

    def __init__(
        self,
        notes_url: str,
        bearer_token: str | None = None,
        *,
        transport: Any | None = None,
    ) -> None:
        """Create a client.

        Args:
            notes_url: Base URL of the Notes MCP server (streamable HTTP).
            bearer_token: Optional bearer token sent as an Authorization header.
            transport: Optional transport override (used by tests to plug in
                an in-memory FastMCP server instead of the HTTP transport).
        """
        self._notes_url = notes_url
        self._bearer_token = bearer_token
        self._transport_override = transport

    @property
    def notes_url(self) -> str:
        """Base URL of the Notes MCP server."""
        return self._notes_url

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
        return StreamableHttpTransport(url=self._notes_url, headers=self.headers)

    async def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Invoke one Notes tool and return its raw result.

        Errors whose text indicates an 'already exists' condition are
        surfaced as :class:`_FolderAlreadyExists` so idempotent folder
        creation can treat them as success; everything else is wrapped in
        :class:`NotesClientError`.
        """
        try:
            async with Client(self._make_transport()) as client:
                result = await client.call_tool(tool, arguments)
        except NotesClientError:
            raise
        except Exception as e:
            if _is_exists_error(str(e)):
                raise _FolderAlreadyExists(str(e)) from e
            raise NotesClientError(f"Notes MCP call {tool!r} failed: {e}") from e

        is_error = getattr(result, "is_error", False)
        if is_error:
            text = _result_text(result)
            if _is_exists_error(text):
                raise _FolderAlreadyExists(text)
            raise NotesClientError(f"Notes tool call {tool!r} returned an error: {text[:200]!r}")
        return result

    async def ensure_folder(self, folder_path: str) -> None:
        """Ensure a folder exists in the notes vault.

        Calls the Notes ``create_folder`` tool; "already exists"-style
        errors are treated as success.

        Raises:
            NotesClientError: On transport/protocol failure or any other
                tool error.
        """
        try:
            await self._call(CREATE_FOLDER_TOOL, {"path": folder_path})
        except _FolderAlreadyExists:
            logger.debug("Folder %r already exists; treating ensure_folder as success", folder_path)

    async def save_note(self, note_path: str, content: str) -> None:
        """Create (upsert) a note at the given path with the given content.

        Raises:
            NotesClientError: On transport/protocol failure or tool error.
        """
        await self._call(CREATE_NOTE_TOOL, {"path": note_path, "content": content})


def build_note_path(folder: str, report_date: str) -> str:
    """Build the deterministic note path for one report date.

    The caller supplies the source-specific folder from configuration; the
    digest's first line already identifies the source.
    """
    return f"{folder.rstrip('/')}/{report_date}.md"
