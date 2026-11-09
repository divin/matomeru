"""Tests for CLI entry points (matomeru-reddit / matomeru-arxiv)."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest

from matomeru.cli import main_arxiv, main_reddit, run_source_stub
from matomeru.window import compute_run_window

REQUIRED_ENV_VARS = [
    "SAGASHI_MCP_URL",
    "NOTES_MCP_URL",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_FROM",
    "SMTP_TO",
    "TZ",
]

MINIMAL_TOML = """[profile]
path = "{profile_path}"

[reddit]
subreddits = ["python"]

[arxiv]
categories = []

[notes]
reddit_folder = ""
arxiv_folder = ""
"""


def make_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    """Build a complete valid environment for the CLI."""
    env = {
        "SAGASHI_MCP_URL": "http://localhost:8000/mcp",
        "NOTES_MCP_URL": "http://localhost:8001/mcp",
        "LLM_BASE_URL": "http://localhost:8002/v1",
        "LLM_MODEL": "test-model",
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "587",
        "SMTP_FROM": "from@example.com",
        "SMTP_TO": "to@example.com",
        "TZ": "UTC",
        "MATOMERU_STATE_DB": str(tmp_path / "state" / "state.sqlite3"),
    }
    env.update(overrides)
    return env


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Set a complete valid environment and a minimal config; return config path."""
    toml_path = tmp_path / "matomeru.toml"
    toml_path.write_text(MINIMAL_TOML.format(profile_path=str(tmp_path / "profile.md")))

    # Ensure required vars are absent before setting, so dotenv/state don't leak in
    for var in REQUIRED_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("MATOMERU_STATE_DB", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    for key, value in make_env(tmp_path).items():
        monkeypatch.setenv(key, value)
    return toml_path


class TestDryRun:
    """--dry-run prints the plan and never touches state."""

    def test_dry_run_exit_zero_no_db(self, cli_env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main_reddit(["--dry-run", "--config", str(cli_env)])

        assert exit_code == 0
        state_db = Path(make_env(tmp_path)["MATOMERU_STATE_DB"])
        assert not state_db.exists()
        assert not state_db.parent.exists()

    def test_dry_run_prints_window_plan(self, cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main_reddit(["--dry-run", "--config", str(cli_env)])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "dry" in captured.err.lower()
        # Window bounds in ISO-8601 UTC appear in the logged plan
        assert "+00:00" in captured.err

    def test_dry_run_arxiv_exit_zero_no_db(self, cli_env: Path, tmp_path: Path) -> None:
        exit_code = main_arxiv(["--dry-run", "--config", str(cli_env)])

        assert exit_code == 0
        assert not Path(make_env(tmp_path)["MATOMERU_STATE_DB"]).exists()


class TestMissingEnv:
    """Missing required env vars exit 1 with a clean error, no DB created."""

    def test_missing_env_exit_one_no_db(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(MINIMAL_TOML.format(profile_path=str(tmp_path / "profile.md")))
        state_db = tmp_path / "state" / "state.sqlite3"

        for var in REQUIRED_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv("MATOMERU_STATE_DB", raising=False)

        exit_code = main_reddit(["--config", str(toml_path)])

        assert exit_code == 1
        captured = capsys.readouterr()
        # Clean error: message on stderr, no traceback
        assert "Missing required environment variables" in captured.err
        assert "Traceback" not in captured.err
        assert not state_db.exists()

    def test_missing_env_arxiv_exit_one(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(MINIMAL_TOML.format(profile_path=str(tmp_path / "profile.md")))

        for var in REQUIRED_ENV_VARS:
            monkeypatch.delenv(var, raising=False)

        assert main_arxiv(["--config", str(toml_path)]) == 1


class TestForceFlag:
    """--force is accepted and drives the restart path."""

    def test_force_accepted_creates_run(self, cli_env: Path, tmp_path: Path) -> None:
        exit_code = main_reddit(["--force", "--config", str(cli_env)])

        assert exit_code == 0
        conn = sqlite3.connect(make_env(tmp_path)["MATOMERU_STATE_DB"])
        try:
            row = conn.execute("SELECT source, status FROM runs").fetchone()
            assert row is not None
            assert row[0] == "reddit"
            assert row[1] == "running"
        finally:
            conn.close()

    def test_force_restarts_completed_run(self, cli_env: Path, tmp_path: Path) -> None:
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]

        assert main_reddit(["--config", str(cli_env)]) == 0
        # Mark the run completed, then re-run with --force
        conn = sqlite3.connect(state_db)
        try:
            conn.execute("UPDATE runs SET status = 'completed'")
            conn.commit()
        finally:
            conn.close()

        assert main_reddit(["--force", "--config", str(cli_env)]) == 0
        conn = sqlite3.connect(state_db)
        try:
            assert conn.execute("SELECT status FROM runs").fetchone()[0] == "running"
        finally:
            conn.close()

    def test_without_force_skips_completed_run(
        self, cli_env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]

        assert main_reddit(["--config", str(cli_env)]) == 0
        conn = sqlite3.connect(state_db)
        try:
            conn.execute("UPDATE runs SET status = 'completed'")
            conn.commit()
        finally:
            conn.close()

        exit_code = main_reddit(["--config", str(cli_env)])

        assert exit_code == 0
        assert "skipping" in capsys.readouterr().err.lower()


class TestStubRun:
    """A real (non-dry-run) invocation leaves the run in 'running' state."""

    def test_run_logs_not_implemented_and_stays_running(
        self, cli_env: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main_reddit(["--config", str(cli_env)])

        assert exit_code == 0
        assert "pipeline not yet implemented" in capsys.readouterr().err

        conn = sqlite3.connect(make_env(tmp_path)["MATOMERU_STATE_DB"])
        try:
            row = conn.execute("SELECT status, email_sent, note_saved FROM runs").fetchone()
            assert row is not None
            assert row[0] == "running"
            assert row[1] == 0
            assert row[2] == 0
        finally:
            conn.close()

    def test_arxiv_run_creates_arxiv_row(self, cli_env: Path, tmp_path: Path) -> None:
        assert main_arxiv(["--config", str(cli_env)]) == 0

        conn = sqlite3.connect(make_env(tmp_path)["MATOMERU_STATE_DB"])
        try:
            rows = conn.execute("SELECT source, status FROM runs").fetchall()
            assert rows == [("arxiv", "running")]
        finally:
            conn.close()


class TestRunSourceStub:
    """The stub pipeline function logs and returns without side effects."""

    def test_stub_logs_not_implemented(self, caplog: pytest.LogCaptureFixture) -> None:
        window = compute_run_window("UTC")

        with caplog.at_level(logging.INFO):
            run_source_stub("reddit", window)

        assert "pipeline not yet implemented" in caplog.text

    def test_stub_accepts_arxiv(self, caplog: pytest.LogCaptureFixture) -> None:
        window = compute_run_window("Europe/Berlin")

        with caplog.at_level(logging.INFO):
            run_source_stub("arxiv", window)

        assert "arxiv" in caplog.text
