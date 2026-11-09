"""Tests for the Notes MCP client against an in-memory fake FastMCP server.

No network and no real MCP server: the fake server is served over fastmcp's
in-memory transport and exposes create_folder/create_note with recorded calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from fastmcp import FastMCP

from matomeru.notes import NotesClientError, NotesMcpClient, build_note_path


@dataclass
class CallLog:
    """Recorded tool calls: (tool, arguments) tuples."""

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


def make_fake_server(
    log: CallLog,
    *,
    folder_exists_error: bool = False,
    note_error: str | None = None,
    folder_error: str | None = None,
) -> FastMCP:
    """Build a fake Notes MCP server recording calls.

    folder_exists_error: create_folder raises an 'already exists' error.
    folder_error / note_error: raise an unrelated error message instead.
    """
    server = FastMCP("notes-fake")

    @server.tool
    def create_folder(path: str) -> str:
        log.calls.append(("create_folder", {"path": path}))
        if folder_error:
            raise RuntimeError(folder_error)
        if folder_exists_error:
            raise ValueError(f"Folder '{path}' already exists")
        return f"created {path}"

    @server.tool
    def create_note(path: str, content: str) -> str:
        log.calls.append(("create_note", {"path": path, "content": content}))
        if note_error:
            raise RuntimeError(note_error)
        return f"saved {path}"

    return server


def make_client(log: CallLog, **kwargs: Any) -> NotesMcpClient:
    return NotesMcpClient("http://notes.invalid", transport=make_fake_server(log, **kwargs))


@pytest.mark.asyncio
async def test_ensure_folder_calls_create_folder() -> None:
    log = CallLog()
    client = make_client(log)
    await client.ensure_folder("notes/reddit")
    assert log.calls == [("create_folder", {"path": "notes/reddit"})]


@pytest.mark.asyncio
async def test_ensure_folder_tolerates_exists_error() -> None:
    log = CallLog()
    client = make_client(log, folder_exists_error=True)
    await client.ensure_folder("notes/arxiv")  # must not raise
    assert log.calls == [("create_folder", {"path": "notes/arxiv"})]


@pytest.mark.asyncio
async def test_ensure_folder_wraps_other_errors() -> None:
    log = CallLog()
    client = make_client(log, folder_error="disk full")
    with pytest.raises(NotesClientError, match="create_folder"):
        await client.ensure_folder("notes/reddit")


@pytest.mark.asyncio
async def test_save_note_passes_path_and_content() -> None:
    log = CallLog()
    client = make_client(log)
    await client.save_note("notes/reddit/2026-11-09.md", "# Reddit digest — 2026-11-09\n\nbody")
    assert log.calls == [
        ("create_note", {"path": "notes/reddit/2026-11-09.md", "content": "# Reddit digest — 2026-11-09\n\nbody"})
    ]


@pytest.mark.asyncio
async def test_save_note_wraps_errors() -> None:
    log = CallLog()
    client = make_client(log, note_error="permission denied")
    with pytest.raises(NotesClientError, match="create_note"):
        await client.save_note("notes/reddit/2026-11-09.md", "content")


@pytest.mark.asyncio
async def test_client_with_bearer_token_still_works_in_memory() -> None:
    log = CallLog()
    client = NotesMcpClient("http://notes.invalid", "tok", transport=make_fake_server(log))
    assert client.headers == {"Authorization": "Bearer tok"}
    await client.save_note("p.md", "c")
    assert log.calls == [("create_note", {"path": "p.md", "content": "c"})]


def test_build_note_path_strips_trailing_slashes() -> None:
    assert build_note_path("notes/reddit/", "2026-11-09") == "notes/reddit/2026-11-09.md"
    assert build_note_path("notes/reddit", "2026-11-09") == "notes/reddit/2026-11-09.md"


def test_build_note_path_is_deterministic() -> None:
    assert build_note_path("a/b/", "2026-11-09") == build_note_path("a/b", "2026-11-09")
