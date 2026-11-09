"""State store for Matomeru pipeline runs using SQLite."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from matomeru.window import RunWindow


@contextmanager
def _connect(state_db_path: str) -> Generator[sqlite3.Connection]:
    """Yield an sqlite3.Connection with the schema ensured, closing it afterwards."""
    path = Path(state_db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        _ensure_schema(conn)
        yield conn
    finally:
        conn.close()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure the runs table exists in the database."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            source TEXT NOT NULL,
            report_date TEXT NOT NULL,
            window_start_utc TEXT NOT NULL,
            window_end_utc TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('running','completed','failed')),
            email_sent INTEGER NOT NULL DEFAULT 0,
            note_saved INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (source, report_date)
        )
        """
    )
    conn.commit()


def begin_run(state_db_path: str, source: str, window: RunWindow, *, force: bool = False) -> tuple[str, str]:
    """Begin a run for the given source and window.

    Args:
        state_db_path: Path to the SQLite state database.
        source: Source name (e.g. 'reddit', 'arxiv').
        window: The RunWindow to record.
        force: If True, restart runs even if already completed.

    Returns:
        Tuple of (report_date, action) where action is 'created', 'skipped', or 'restart'.
    """
    with _connect(state_db_path) as conn:
        report_date_str = window.report_date.isoformat()
        window_start_str = window.start_utc.isoformat()
        window_end_str = window.end_utc.isoformat()

        cursor = conn.execute(
            "SELECT status, window_start_utc, window_end_utc FROM runs WHERE source = ? AND report_date = ?",
            (source, report_date_str),
        )
        row = cursor.fetchone()

        if row is None:
            # No existing row: create new
            conn.execute(
                """
                INSERT INTO runs (source, report_date, window_start_utc, window_end_utc, status, updated_at)
                VALUES (?, ?, ?, ?, 'running', ?)
                """,
                (
                    source,
                    report_date_str,
                    window_start_str,
                    window_end_str,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            return report_date_str, "created"

        status = row[0]

        if status == "completed":
            if force:
                # Restart: update status to running, keep original window values
                conn.execute(
                    """
                    UPDATE runs SET status = 'running', updated_at = ?
                    WHERE source = ? AND report_date = ?
                    """,
                    (datetime.now(timezone.utc).isoformat(), source, report_date_str),
                )
                conn.commit()
                return report_date_str, "restart"
            else:
                return report_date_str, "skipped"

        # status is 'running' or 'failed': treat as restartable
        conn.execute(
            """
            UPDATE runs SET status = 'running', window_start_utc = ?, window_end_utc = ?, updated_at = ?
            WHERE source = ? AND report_date = ?
            """,
            (
                window_start_str,
                window_end_str,
                datetime.now(timezone.utc).isoformat(),
                source,
                report_date_str,
            ),
        )
        conn.commit()
        return report_date_str, "restart"


def mark_delivery(
    state_db_path: str,
    source: str,
    report_date: str,
    *,
    email_sent: bool | None = None,
    note_saved: bool | None = None,
) -> None:
    """Update delivery flags for a run.

    Args:
        state_db_path: Path to the SQLite state database.
        source: Source name.
        report_date: Report date string (YYYY-MM-DD).
        email_sent: If not None, update email_sent flag.
        note_saved: If not None, update note_saved flag.
    """
    with _connect(state_db_path) as conn:
        updates = []
        params = []

        if email_sent is not None:
            updates.append("email_sent = ?")
            params.append(1 if email_sent else 0)

        if note_saved is not None:
            updates.append("note_saved = ?")
            params.append(1 if note_saved else 0)

        if updates:
            now_iso = datetime.now(timezone.utc).isoformat()
            params.extend([now_iso, source, report_date])
            conn.execute(
                f"UPDATE runs SET {', '.join(updates)}, updated_at = ? WHERE source = ? AND report_date = ?",
                params,
            )
            conn.commit()


def complete_run(state_db_path: str, source: str, report_date: str) -> None:
    """Mark a run as completed.

    Args:
        state_db_path: Path to the SQLite state database.
        source: Source name.
        report_date: Report date string (YYYY-MM-DD).
    """
    with _connect(state_db_path) as conn:
        conn.execute(
            """
            UPDATE runs SET status = 'completed', updated_at = ?
            WHERE source = ? AND report_date = ?
            """,
            (datetime.now(timezone.utc).isoformat(), source, report_date),
        )
        conn.commit()


def fail_run(state_db_path: str, source: str, report_date: str) -> None:
    """Mark a run as failed.

    Args:
        state_db_path: Path to the SQLite state database.
        source: Source name.
        report_date: Report date string (YYYY-MM-DD).
    """
    with _connect(state_db_path) as conn:
        conn.execute(
            """
            UPDATE runs SET status = 'failed', updated_at = ?
            WHERE source = ? AND report_date = ?
            """,
            (datetime.now(timezone.utc).isoformat(), source, report_date),
        )
        conn.commit()


def get_run(state_db_path: str, source: str, report_date: str) -> dict | None:
    """Get a run record by source and report_date.

    Args:
        state_db_path: Path to the SQLite state database.
        source: Source name.
        report_date: Report date string (YYYY-MM-DD).

    Returns:
        Dict with keys: source, report_date, window_start_utc, window_end_utc, status, email_sent, note_saved, updated_at  # noqa: E501
        or None if not found.
    """
    with _connect(state_db_path) as conn:
        cursor = conn.execute(
            """
            SELECT source, report_date, window_start_utc, window_end_utc, status, email_sent, note_saved, updated_at
            FROM runs WHERE source = ? AND report_date = ?
            """,
            (source, report_date),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return {
            "source": row[0],
            "report_date": row[1],
            "window_start_utc": row[2],
            "window_end_utc": row[3],
            "status": row[4],
            "email_sent": bool(row[5]),
            "note_saved": bool(row[6]),
            "updated_at": row[7],
        }
