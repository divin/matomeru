"""Tests for state store."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from matomeru.funnel import ScoredItem
from matomeru.state import (
    begin_run,
    complete_run,
    fail_run,
    get_run,
    mark_delivery,
    record_rerank_scores,
)
from matomeru.window import compute_run_window


class TestBeginRun:
    """Tests for begin_run."""

    def test_created_new_run(self, tmp_path: Path) -> None:
        """First run for (source, report_date) creates a row."""
        db_path = str(tmp_path / "test.db")
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = compute_run_window("UTC", now=now)

        report_date, action = begin_run(db_path, "reddit", window)

        assert action == "created"
        assert report_date == window.report_date.isoformat()

        run = get_run(db_path, "reddit", window.report_date.isoformat())
        assert run is not None
        assert run["source"] == "reddit"
        assert run["status"] == "running"
        assert run["email_sent"] is False
        assert run["note_saved"] is False
        assert run["window_start_utc"] == window.start_utc.isoformat()
        assert run["window_end_utc"] == window.end_utc.isoformat()

    def test_skipped_when_completed_without_force(self, tmp_path: Path) -> None:
        """Completed run without force returns 'skipped'."""
        db_path = str(tmp_path / "test.db")
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = compute_run_window("UTC", now=now)

        # First run: create
        report_date, action = begin_run(db_path, "reddit", window)
        assert action == "created"

        # Complete it
        complete_run(db_path, "reddit", report_date)

        # Second run: should be skipped
        report_date2, action2 = begin_run(db_path, "reddit", window)
        assert action2 == "skipped"
        assert report_date2 == report_date

    def test_restart_with_force_on_completed(self, tmp_path: Path) -> None:
        """Completed run with force returns 'restart' and updates status."""
        db_path = str(tmp_path / "test.db")
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = compute_run_window("UTC", now=now)

        # First run: create
        report_date, action = begin_run(db_path, "reddit", window)
        assert action == "created"

        # Complete it
        complete_run(db_path, "reddit", report_date)

        # Second run with force: restart
        report_date2, action2 = begin_run(db_path, "reddit", window, force=True)
        assert action2 == "restart"

        run = get_run(db_path, "reddit", report_date)
        assert run is not None
        assert run["status"] == "running"

    def test_restart_on_running(self, tmp_path: Path) -> None:
        """Running run returns 'restart'."""
        db_path = str(tmp_path / "test.db")
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = compute_run_window("UTC", now=now)

        # First run: create
        report_date, action = begin_run(db_path, "reddit", window)
        assert action == "created"

        # Second run without completing: restart
        report_date2, action2 = begin_run(db_path, "reddit", window)
        assert action2 == "restart"

        run = get_run(db_path, "reddit", report_date)
        assert run is not None
        assert run["status"] == "running"

    def test_restart_on_failed(self, tmp_path: Path) -> None:
        """Failed run returns 'restart'."""
        db_path = str(tmp_path / "test.db")
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = compute_run_window("UTC", now=now)

        # First run: create
        report_date, action = begin_run(db_path, "reddit", window)
        assert action == "created"

        # Mark as failed
        fail_run(db_path, "reddit", report_date)

        # Second run: restart
        report_date2, action2 = begin_run(db_path, "reddit", window)
        assert action2 == "restart"

        run = get_run(db_path, "reddit", report_date)
        assert run is not None
        assert run["status"] == "running"

    def test_different_sources_independent(self, tmp_path: Path) -> None:
        """Different sources maintain separate state."""
        db_path = str(tmp_path / "test.db")
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = compute_run_window("UTC", now=now)

        report_date1, action1 = begin_run(db_path, "reddit", window)
        report_date2, action2 = begin_run(db_path, "arxiv", window)

        assert action1 == "created"
        assert action2 == "created"
        assert report_date1 == report_date2

        run_reddit = get_run(db_path, "reddit", window.report_date.isoformat())
        run_arxiv = get_run(db_path, "arxiv", window.report_date.isoformat())

        assert run_reddit is not None
        assert run_arxiv is not None
        assert run_reddit["source"] == "reddit"
        assert run_arxiv["source"] == "arxiv"


class TestMarkDelivery:
    """Tests for mark_delivery."""

    def test_email_sent_flag(self, tmp_path: Path) -> None:
        """email_sent flag can be set."""
        db_path = str(tmp_path / "test.db")
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = compute_run_window("UTC", now=now)

        report_date, _ = begin_run(db_path, "reddit", window)
        mark_delivery(db_path, "reddit", report_date, email_sent=True)

        run = get_run(db_path, "reddit", report_date)
        assert run is not None
        assert run["email_sent"] is True

    def test_note_saved_flag(self, tmp_path: Path) -> None:
        """note_saved flag can be set."""
        db_path = str(tmp_path / "test.db")
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = compute_run_window("UTC", now=now)

        report_date, _ = begin_run(db_path, "reddit", window)
        mark_delivery(db_path, "reddit", report_date, note_saved=True)

        run = get_run(db_path, "reddit", report_date)
        assert run is not None
        assert run["note_saved"] is True

    def test_both_flags(self, tmp_path: Path) -> None:
        """Both flags can be set together."""
        db_path = str(tmp_path / "test.db")
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = compute_run_window("UTC", now=now)

        report_date, _ = begin_run(db_path, "reddit", window)
        mark_delivery(db_path, "reddit", report_date, email_sent=True, note_saved=True)

        run = get_run(db_path, "reddit", report_date)
        assert run is not None
        assert run["email_sent"] is True
        assert run["note_saved"] is True

    def test_unset_flags_false(self, tmp_path: Path) -> None:
        """Unset flags default to False."""
        db_path = str(tmp_path / "test.db")
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = compute_run_window("UTC", now=now)

        report_date, _ = begin_run(db_path, "reddit", window)
        mark_delivery(db_path, "reddit", report_date, email_sent=False)

        run = get_run(db_path, "reddit", report_date)
        assert run is not None
        assert run["email_sent"] is False
        assert run["note_saved"] is False

    def test_noop_when_no_flags(self, tmp_path: Path) -> None:
        """No-op when no flags are specified."""
        db_path = str(tmp_path / "test.db")
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = compute_run_window("UTC", now=now)

        report_date, _ = begin_run(db_path, "reddit", window)

        # Call with no flags - should not error
        mark_delivery(db_path, "reddit", report_date)

        run = get_run(db_path, "reddit", report_date)
        assert run is not None
        assert run["email_sent"] is False
        assert run["note_saved"] is False


class TestCompleteRun:
    """Tests for complete_run."""

    def test_complete_run(self, tmp_path: Path) -> None:
        """Mark run as completed."""
        db_path = str(tmp_path / "test.db")
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = compute_run_window("UTC", now=now)

        report_date, _ = begin_run(db_path, "reddit", window)
        complete_run(db_path, "reddit", report_date)

        run = get_run(db_path, "reddit", report_date)
        assert run is not None
        assert run["status"] == "completed"


class TestFailRun:
    """Tests for fail_run."""

    def test_fail_run(self, tmp_path: Path) -> None:
        """Mark run as failed."""
        db_path = str(tmp_path / "test.db")
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = compute_run_window("UTC", now=now)

        report_date, _ = begin_run(db_path, "reddit", window)
        fail_run(db_path, "reddit", report_date)

        run = get_run(db_path, "reddit", report_date)
        assert run is not None
        assert run["status"] == "failed"


class TestGetRun:
    """Tests for get_run."""

    def test_get_existing_run(self, tmp_path: Path) -> None:
        """Retrieve existing run."""
        db_path = str(tmp_path / "test.db")
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = compute_run_window("UTC", now=now)

        report_date, _ = begin_run(db_path, "reddit", window)

        run = get_run(db_path, "reddit", report_date)
        assert run is not None
        assert run is not None
        assert run["source"] == "reddit"
        assert run["report_date"] == report_date
        assert run["status"] == "running"

    def test_get_nonexistent_run(self, tmp_path: Path) -> None:
        """Get nonexistent run returns None."""
        db_path = str(tmp_path / "test.db")

        run = get_run(db_path, "reddit", "2025-06-15")
        assert run is None

    def test_persists_across_connections(self, tmp_path: Path) -> None:
        """Data persists across new connections."""
        db_path = str(tmp_path / "test.db")
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = compute_run_window("UTC", now=now)

        # First connection
        begin_run(db_path, "reddit", window)

        # Second connection should see the same data
        run = get_run(db_path, "reddit", window.report_date.isoformat())
        assert run is not None
        assert run["source"] == "reddit"
        assert run["status"] == "running"

    def test_roundtrip_datetime_format(self, tmp_path: Path) -> None:
        """Datetime strings are stored in ISO-8601 format."""
        db_path = str(tmp_path / "test.db")
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = compute_run_window("UTC", now=now)

        report_date, _ = begin_run(db_path, "reddit", window)

        run = get_run(db_path, "reddit", report_date)
        assert run is not None
        # Check that stored format is ISO-8601 (has 'T' separator)
        assert "T" in run["window_start_utc"]
        assert "T" in run["window_end_utc"]
        assert run["window_end_utc"].endswith("+00:00") or run["window_end_utc"].endswith("Z")


class TestSchema:
    """Tests for schema behavior."""

    def test_table_created_implicitly(self, tmp_path: Path) -> None:
        """Table is created on first use."""
        db_path = str(tmp_path / "test.db")

        # Should not raise
        begin_run(db_path, "reddit", compute_run_window("UTC"))

        # Verify table exists
        conn = pytest.importorskip("sqlite3").connect(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='runs'")
        assert cursor.fetchone() is not None

    def test_status_constraint_enforced(self, tmp_path: Path) -> None:
        """Status constraint prevents invalid values."""
        db_path = str(tmp_path / "test.db")
        conn = pytest.importorskip("sqlite3").connect(db_path)

        # Create table manually
        conn.execute(
            """
            CREATE TABLE runs (
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

        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        window = compute_run_window("UTC", now=now)
        begin_run(db_path, "reddit", window)

        # Cannot insert invalid status directly
        with pytest.raises(Exception):
            conn.execute(
                "INSERT INTO runs (source, report_date, window_start_utc, window_end_utc, status, updated_at) VALUES (?, ?, ?, ?, ?, ?)",  # noqa: E501
                (
                    "test",
                    "2025-06-15",
                    window.start_utc.isoformat(),
                    window.end_utc.isoformat(),
                    "invalid",
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        conn.close()


class TestRecordRerankScores:
    """Tests for record_rerank_scores and the rerank_scores table."""

    def _fetch_all(self, db_path: str) -> list[tuple]:
        conn = sqlite3.connect(db_path)
        try:
            return conn.execute(
                "SELECT source, report_date, item_id, score, kept, section FROM rerank_scores ORDER BY item_id"
            ).fetchall()
        finally:
            conn.close()

    def test_table_created_implicitly(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        record_rerank_scores(db_path, "reddit", "2026-11-12", [])
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rerank_scores'")
            assert cursor.fetchone() is not None
        finally:
            conn.close()

    def test_rows_written_and_retrievable(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        rows = [
            ScoredItem(item_id="t3_1", subreddit="python", score=-5.1, section="Hardware", kept=True),
            ScoredItem(item_id="t3_2", subreddit="python", score=-8.4, section="Cooking", kept=False),
        ]
        record_rerank_scores(db_path, "reddit", "2026-11-12", rows)

        assert self._fetch_all(db_path) == [
            ("reddit", "2026-11-12", "t3_1", -5.1, 1, "Hardware"),
            ("reddit", "2026-11-12", "t3_2", -8.4, 0, "Cooking"),
        ]

    def test_kept_stored_as_zero_or_one(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        record_rerank_scores(
            db_path,
            "reddit",
            "2026-11-12",
            [ScoredItem(item_id="t3_1", subreddit="python", score=-1.0, section="S", kept=True)],
        )
        conn = sqlite3.connect(db_path)
        try:
            kept_values = [row[0] for row in conn.execute("SELECT kept FROM rerank_scores").fetchall()]
        finally:
            conn.close()
        assert kept_values == [1]
        assert all(isinstance(v, int) for v in kept_values)

    def test_idempotent_reinsert_replaces_row(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        first = ScoredItem(item_id="t3_1", subreddit="python", score=-5.0, section="Old", kept=False)
        second = ScoredItem(item_id="t3_1", subreddit="python", score=-2.0, section="New", kept=True)
        record_rerank_scores(db_path, "reddit", "2026-11-12", [first])
        record_rerank_scores(db_path, "reddit", "2026-11-12", [second])

        rows = self._fetch_all(db_path)
        assert len(rows) == 1
        assert rows[0] == ("reddit", "2026-11-12", "t3_1", -2.0, 1, "New")

    def test_rows_are_scoped_by_source_and_date(self, tmp_path: Path) -> None:
        db_path = str(tmp_path / "test.db")
        row = ScoredItem(item_id="t3_1", subreddit="python", score=-1.0, section="S", kept=True)
        record_rerank_scores(db_path, "reddit", "2026-11-12", [row])
        record_rerank_scores(db_path, "reddit", "2026-11-13", [row])
        record_rerank_scores(db_path, "arxiv", "2026-11-12", [row])

        assert len(self._fetch_all(db_path)) == 3
