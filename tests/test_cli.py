"""Tests for CLI entry points (matomeru-reddit / matomeru-arxiv).

All external collaborators (Sagashi, LLM, SMTP, Notes MCP) are stubbed at
the ``matomeru.cli`` module boundary; no test touches the network. State is
asserted through a real SQLite database in ``tmp_path`` via ``get_run``.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path

import httpx
import pytest

import matomeru.cli as cli
from matomeru.llm import ArxivPaperSummary, RedditItemSummary
from matomeru.mail import MailError
from matomeru.models import ArxivPaper, RedditThread, ThreadFetchError
from matomeru.notes import NotesClientError
from matomeru.sagashi_client import SagashiClientError
from matomeru.state import get_run
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

DELIVERY_TOML = """[profile]
path = "{profile_path}"

[reddit]
subreddits = ["python"]

[arxiv]
categories = []

[notes]
reddit_folder = "digests/reddit"
arxiv_folder = "digests/arxiv"
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


def _write_profile(tmp_path: Path) -> Path:
    profile_path = tmp_path / "profile.md"
    profile_path.write_text("Interested in: exotic particles and slow loris conservation.")
    return profile_path


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Set a complete valid environment and a minimal config; return config path."""
    _write_profile(tmp_path)
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


def _write_delivery_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Environment + config with notes folders configured for both sources."""
    _write_profile(tmp_path)
    toml_path = tmp_path / "matomeru.toml"
    toml_path.write_text(DELIVERY_TOML.format(profile_path=str(tmp_path / "profile.md")))

    for var in REQUIRED_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("MATOMERU_STATE_DB", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    for key, value in make_env(tmp_path).items():
        monkeypatch.setenv(key, value)
    return toml_path


# --- External collaborator stubs ---------------------------------------------


def make_thread(i: int) -> RedditThread:
    return RedditThread(
        id=f"t3_{i}",
        subreddit="python",
        title=f"Post {i}",
        score=i,
        author="alice",
        permalink=f"/r/python/comments/{i}",
        num_comments=2,
    )


def make_paper(i: int) -> ArxivPaper:
    stamp = datetime(2026, 11, 8, 12, 0, 0, tzinfo=UTC)
    return ArxivPaper(
        arxiv_id=f"2611.{i:05d}",
        abs_url=f"https://arxiv.org/abs/2611.{i:05d}",
        pdf_url=f"https://arxiv.org/pdf/2611.{i:05d}",
        title=f"Paper {i}",
        abstract=f"Abstract {i}",
        authors=["Alice", "Bob"],
        categories=["cs.LG"],
        primary_category="cs.LG",
        published=stamp,
        updated=stamp,
    )


def stub_collect_reddit(
    monkeypatch: pytest.MonkeyPatch,
    threads: list[RedditThread],
    errors: list[ThreadFetchError] | None = None,
    *,
    fail: bool = False,
) -> None:
    """Stub matomeru.cli.collect_reddit at the module boundary."""

    async def fake_collect_reddit(
        client: object,
        subreddits: list[str],
        window: object,
        *,
        posts_per_subreddit: int = 3,
        comments_per_post: int = 20,
    ) -> tuple[list[RedditThread], list[ThreadFetchError]]:
        if fail:
            raise SagashiClientError("sagashi is down")
        return threads, list(errors or [])

    monkeypatch.setattr(cli, "collect_reddit", fake_collect_reddit)


def stub_summarize_reddit(monkeypatch: pytest.MonkeyPatch, fail_ids: set[str] | None = None) -> None:
    """Stub matomeru.cli.summarize_reddit_item; fail for the given thread ids."""
    fail_ids = fail_ids or set()

    async def fake_summarize_reddit_item(agent: object, profile_text: str, thread: RedditThread) -> RedditItemSummary:
        if thread.id in fail_ids:
            raise RuntimeError("LLM exploded")
        return RedditItemSummary(summary=f"Summary of {thread.title}", why_relevant="matches profile")

    monkeypatch.setattr(cli, "summarize_reddit_item", fake_summarize_reddit_item)


def stub_collect_arxiv(
    monkeypatch: pytest.MonkeyPatch,
    papers: list[ArxivPaper],
    *,
    fail: bool = False,
) -> None:
    """Stub matomeru.cli.collect_arxiv at the module boundary."""

    async def fake_collect_arxiv(
        http: object,
        categories: list[str],
        search_terms: list[str],
        window: object,
        *,
        max_results: int = 100,
    ) -> tuple[list[ArxivPaper], int]:
        if fail:
            raise httpx.ConnectError("arxiv unreachable")
        return papers, len(papers)

    monkeypatch.setattr(cli, "collect_arxiv", fake_collect_arxiv)


def stub_screen_arxiv(
    monkeypatch: pytest.MonkeyPatch,
    decisions: dict[str, bool],
    fail_ids: set[str] | None = None,
) -> None:
    fail_ids = fail_ids or set()

    """Stub matomeru.cli.screen_arxiv_paper with include flags per arxiv_id."""

    class _Decision:
        def __init__(self, include: bool) -> None:
            self.include = include
            self.reason = "because"

    async def fake_screen(agent: object, profile_text: str, paper: ArxivPaper) -> _Decision:
        if paper.arxiv_id in fail_ids:
            raise RuntimeError("screening exploded")
        return _Decision(decisions[paper.arxiv_id])

    monkeypatch.setattr(cli, "screen_arxiv_paper", fake_screen)


def stub_summarize_arxiv(monkeypatch: pytest.MonkeyPatch, fail_ids: set[str] | None = None) -> None:
    """Stub matomeru.cli.summarize_arxiv_paper; fail for the given arxiv_ids."""
    fail_ids = fail_ids or set()

    async def fake_summarize(agent: object, profile_text: str, paper: ArxivPaper) -> ArxivPaperSummary:
        if paper.arxiv_id in fail_ids:
            raise RuntimeError("LLM exploded")
        return ArxivPaperSummary(summary=f"Summary of {paper.title}", why_relevant="matches profile")

    monkeypatch.setattr(cli, "summarize_arxiv_paper", fake_summarize)


def stub_send_mail(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail: bool = False,
) -> list[dict[str, object]]:
    """Stub matomeru.cli.send_mail; records host/port/user/pass/from/recipients/body."""
    calls: list[dict[str, object]] = []

    async def fake_send_mail(
        host: str,
        port: int,
        username: str | None,
        password: str | None,
        sender: str,
        recipients: list[str],
        message: EmailMessage,
    ) -> None:
        calls.append(
            {
                "host": host,
                "port": port,
                "username": username,
                "password": password,
                "sender": sender,
                "recipients": list(recipients),
                "subject": message.get("Subject", ""),
                "body": str(message.get_content()),
            }
        )
        if fail:
            raise MailError("SMTP send failed: connection refused")

    monkeypatch.setattr(cli, "send_mail", fake_send_mail)
    return calls


def stub_notes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_on: str | None = None,
) -> list[tuple[str, str]]:
    """Stub matomeru.cli.NotesMcpClient; records (note_path, content) pairs."""
    saved: list[tuple[str, str]] = []

    class FakeNotesClient:
        def __init__(self, notes_url: str, bearer_token: str | None = None) -> None:
            self.notes_url = notes_url

        async def ensure_folder(self, folder_path: str) -> None:
            if fail_on == "ensure_folder":
                raise NotesClientError("notes MCP unavailable")

        async def save_note(self, note_path: str, content: str) -> None:
            saved.append((note_path, content))
            if fail_on == "save_note":
                raise NotesClientError("notes MCP unavailable")

    monkeypatch.setattr(cli, "NotesMcpClient", FakeNotesClient)
    return saved


def _insert_completed_run(state_db: str, source: str, report_date: str = "2026-11-09") -> None:
    Path(state_db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(state_db)
    try:
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
        conn.execute(
            """
            INSERT INTO runs (source, report_date, window_start_utc, window_end_utc, status, updated_at)
            VALUES (?, ?, '2026-11-08T00:00:00+00:00', '2026-11-09T00:00:00+00:00', 'completed', '2026-11-09T00:00:00+00:00')
            """,
            (source, report_date),
        )
        conn.commit()
    finally:
        conn.close()


# --- Dry run -----------------------------------------------------------------


class TestDryRun:
    """--dry-run prints the plan and never touches state."""

    def test_dry_run_exit_zero_no_db(self, cli_env: Path, tmp_path: Path) -> None:
        exit_code = cli.main_reddit(["--dry-run", "--config", str(cli_env)])

        assert exit_code == 0
        state_db = Path(make_env(tmp_path)["MATOMERU_STATE_DB"])
        assert not state_db.exists()
        assert not state_db.parent.exists()

    def test_dry_run_prints_window_plan(self, cli_env: Path, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = cli.main_reddit(["--dry-run", "--config", str(cli_env)])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "dry" in captured.err.lower()
        # Window bounds in ISO-8601 UTC appear in the logged plan
        assert "+00:00" in captured.err

    def test_dry_run_arxiv_exit_zero_no_db(self, cli_env: Path, tmp_path: Path) -> None:
        exit_code = cli.main_arxiv(["--dry-run", "--config", str(cli_env)])

        assert exit_code == 0
        assert not Path(make_env(tmp_path)["MATOMERU_STATE_DB"]).exists()

    def test_dry_run_never_calls_pipeline(self, monkeypatch: pytest.MonkeyPatch, cli_env: Path) -> None:
        """Stubs raise if called: dry-run must not reach collection or delivery."""

        def boom(*args: object, **kwargs: object) -> None:
            raise AssertionError("pipeline must not run on dry-run")

        monkeypatch.setattr(cli, "collect_reddit", boom)
        monkeypatch.setattr(cli, "collect_arxiv", boom)
        monkeypatch.setattr(cli, "send_mail", boom)
        monkeypatch.setattr(cli, "NotesMcpClient", boom)

        assert cli.main_reddit(["--dry-run", "--config", str(cli_env)]) == 0
        assert cli.main_arxiv(["--dry-run", "--config", str(cli_env)]) == 0


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

        exit_code = cli.main_reddit(["--config", str(toml_path)])

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

        assert cli.main_arxiv(["--config", str(toml_path)]) == 1


class TestMissingProfile:
    """A missing profile file is a ConfigError: exit 1, no run row created."""

    def test_missing_profile_exit_one_no_db(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(MINIMAL_TOML.format(profile_path=str(tmp_path / "does-not-exist.md")))

        for var in REQUIRED_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv("MATOMERU_STATE_DB", raising=False)
        for key, value in make_env(tmp_path).items():
            monkeypatch.setenv(key, value)

        exit_code = cli.main_reddit(["--config", str(toml_path)])

        assert exit_code == 1
        assert not Path(make_env(tmp_path)["MATOMERU_STATE_DB"]).exists()


# --- Skip / force ------------------------------------------------------------


class TestSkipAndForce:
    """Skipped runs touch nothing; --force reprocesses."""

    def test_skipped_path_exits_zero_without_running_pipeline(
        self, monkeypatch: pytest.MonkeyPatch, cli_env: Path, tmp_path: Path
    ) -> None:
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]
        _insert_completed_run(state_db, "reddit")

        def boom(*args: object, **kwargs: object) -> None:
            raise AssertionError("pipeline must not run for a skipped run")

        monkeypatch.setattr(cli, "collect_reddit", boom)
        monkeypatch.setattr(cli, "send_mail", boom)
        monkeypatch.setattr(cli, "NotesMcpClient", boom)

        exit_code = cli.main_reddit(["--config", str(cli_env)])

        assert exit_code == 0
        row = get_run(state_db, "reddit", "2026-11-09")
        assert row is not None
        assert row["status"] == "completed"

    def test_force_proceeds_and_reprocesses(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_path = _write_delivery_config(monkeypatch, tmp_path)
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]
        _insert_completed_run(state_db, "reddit")

        stub_collect_reddit(monkeypatch, [make_thread(1)])
        stub_summarize_reddit(monkeypatch)
        stub_send_mail(monkeypatch)
        saved = stub_notes(monkeypatch)

        exit_code = cli.main_reddit(["--force", "--config", str(config_path)])

        assert exit_code == 0
        row = get_run(state_db, "reddit", "2026-11-09")
        assert row is not None
        assert row["status"] == "completed"
        assert len(saved) == 1
        assert saved[0][0].startswith("digests/reddit/")


# --- Reddit pipeline ---------------------------------------------------------


class TestRedditSuccess:
    """Full success path: complete_run, both deliveries, exit 0."""

    def test_full_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_path = _write_delivery_config(monkeypatch, tmp_path)
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]

        stub_collect_reddit(monkeypatch, [make_thread(1), make_thread(2)])
        stub_summarize_reddit(monkeypatch)
        mail_calls = stub_send_mail(monkeypatch)
        saved = stub_notes(monkeypatch)

        exit_code = cli.main_reddit(["--config", str(config_path)])

        assert exit_code == 0
        row = get_run(state_db, "reddit", row_report_date(state_db, "reddit"))
        assert row is not None
        assert row["status"] == "completed"
        assert row["email_sent"] is True
        assert row["note_saved"] is True

        # Email: subject, sender, recipients
        assert len(mail_calls) == 1
        assert mail_calls[0]["sender"] == "from@example.com"
        assert mail_calls[0]["recipients"] == ["to@example.com"]
        assert str(mail_calls[0]["subject"]).startswith("Matomeru reddit digest ")

        # Note: deterministic path and digest title in body
        assert len(saved) == 1
        note_path, note_body = saved[0]
        assert note_path.startswith("digests/reddit/")
        assert note_path.endswith(".md")
        assert "# Reddit digest" in note_body
        assert "Post 1" in note_body
        assert "Post 2" in note_body
        assert "Partial run" not in note_body


def row_report_date(state_db: str, source: str) -> str:
    """The single report_date recorded for a source (test helper)."""
    conn = sqlite3.connect(state_db)
    try:
        rows = conn.execute("SELECT report_date FROM runs WHERE source = ?", (source,)).fetchall()
        assert len(rows) == 1
        return str(rows[0][0])
    finally:
        conn.close()


class TestRedditPartialFailures:
    """Item failures mark the run partial (fail_run + exit 1), delivery continues."""

    def test_llm_failure_marks_partial_and_exits_one(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_path = _write_delivery_config(monkeypatch, tmp_path)
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]

        stub_collect_reddit(monkeypatch, [make_thread(1), make_thread(2)])
        stub_summarize_reddit(monkeypatch, fail_ids={"t3_1"})
        mail_calls = stub_send_mail(monkeypatch)
        saved = stub_notes(monkeypatch)

        exit_code = cli.main_reddit(["--config", str(config_path)])

        assert exit_code == 1
        run_date = row_report_date(state_db, "reddit")
        row = get_run(state_db, "reddit", run_date)
        assert row is not None
        assert row["status"] == "failed"
        # Remaining items still delivered
        assert row["email_sent"] is True
        assert row["note_saved"] is True
        assert len(mail_calls) == 1
        note_body = saved[0][1]
        assert "Post 2" in note_body
        assert "Post 1" not in note_body
        assert "Partial run: some items failed and are omitted." in note_body
        assert "t3_1" in note_body  # error entry names the failed thread

    def test_fetch_error_marks_partial_and_exits_one(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_path = _write_delivery_config(monkeypatch, tmp_path)
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]

        stub_collect_reddit(
            monkeypatch,
            [make_thread(1)],
            errors=[ThreadFetchError(index=0, item="t3_9", error="fetch failed")],
        )
        stub_summarize_reddit(monkeypatch)
        stub_send_mail(monkeypatch)
        saved = stub_notes(monkeypatch)

        exit_code = cli.main_reddit(["--config", str(config_path)])

        assert exit_code == 1
        row = get_run(state_db, "reddit", row_report_date(state_db, "reddit"))
        assert row is not None
        assert row["status"] == "failed"
        assert "t3_9: fetch failed" in saved[0][1]

    def test_collection_failure_fails_run_without_delivery(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_path = _write_delivery_config(monkeypatch, tmp_path)
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]

        stub_collect_reddit(monkeypatch, [], fail=True)
        stub_send_mail(monkeypatch)
        stub_notes(monkeypatch)

        exit_code = cli.main_reddit(["--config", str(config_path)])

        assert exit_code == 1
        row = get_run(state_db, "reddit", row_report_date(state_db, "reddit"))
        assert row is not None
        assert row["status"] == "failed"


class TestRedditDestinationFailures:
    """Destination failures are independent; the other destination still runs."""

    def test_mail_failure_exits_one_note_still_saved(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_path = _write_delivery_config(monkeypatch, tmp_path)
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]

        stub_collect_reddit(monkeypatch, [make_thread(1)])
        stub_summarize_reddit(monkeypatch)
        mail_calls = stub_send_mail(monkeypatch, fail=True)
        saved = stub_notes(monkeypatch)

        exit_code = cli.main_reddit(["--config", str(config_path)])

        assert exit_code == 1
        row = get_run(state_db, "reddit", row_report_date(state_db, "reddit"))
        assert row is not None
        assert row["status"] == "failed"
        assert row["email_sent"] is False
        assert row["note_saved"] is True
        assert len(mail_calls) == 1  # attempted
        assert "# Reddit digest" in saved[0][1]  # note still delivered

    def test_notes_failure_exits_one_mail_still_sent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_path = _write_delivery_config(monkeypatch, tmp_path)
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]

        stub_collect_reddit(monkeypatch, [make_thread(1)])
        stub_summarize_reddit(monkeypatch)
        mail_calls = stub_send_mail(monkeypatch)
        saved = stub_notes(monkeypatch, fail_on="save_note")

        exit_code = cli.main_reddit(["--config", str(config_path)])

        assert exit_code == 1
        row = get_run(state_db, "reddit", row_report_date(state_db, "reddit"))
        assert row is not None
        assert row["status"] == "failed"
        assert row["email_sent"] is True
        assert row["note_saved"] is False
        assert len(mail_calls) == 1
        assert len(saved) == 1  # save was attempted before failing

    def test_notes_folder_unconfigured_is_skip_not_failure(
        self, monkeypatch: pytest.MonkeyPatch, cli_env: Path, tmp_path: Path
    ) -> None:
        """Empty notes folder: warning + skip; email still completes the run."""
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]

        stub_collect_reddit(monkeypatch, [make_thread(1)])
        stub_summarize_reddit(monkeypatch)
        mail_calls = stub_send_mail(monkeypatch)
        saved = stub_notes(monkeypatch)

        exit_code = cli.main_reddit(["--config", str(cli_env)])

        assert exit_code == 0
        row = get_run(state_db, "reddit", row_report_date(state_db, "reddit"))
        assert row is not None
        assert row["status"] == "completed"
        assert row["email_sent"] is True
        assert saved == []  # notes never attempted
        assert len(mail_calls) == 1


class TestRedditEmptyResults:
    """An empty day still renders a no-results digest and both deliveries."""

    def test_empty_items_complete_normally(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_path = _write_delivery_config(monkeypatch, tmp_path)
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]

        stub_collect_reddit(monkeypatch, [])
        stub_summarize_reddit(monkeypatch)
        mail_calls = stub_send_mail(monkeypatch)
        saved = stub_notes(monkeypatch)

        exit_code = cli.main_reddit(["--config", str(config_path)])

        assert exit_code == 0
        row = get_run(state_db, "reddit", row_report_date(state_db, "reddit"))
        assert row is not None
        assert row["status"] == "completed"
        assert row["email_sent"] is True
        assert row["note_saved"] is True
        assert len(mail_calls) == 1
        note_body = saved[0][1]
        assert "No posts matched the configured subreddits in this window." in note_body
        assert "Partial run" not in note_body


# --- arXiv pipeline ----------------------------------------------------------


class TestArxivSuccess:
    """Screening filters non-relevant papers; relevant ones are summarized."""

    def test_screening_excludes_non_relevant_and_completes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_path = _write_delivery_config(monkeypatch, tmp_path)
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]

        papers = [make_paper(1), make_paper(2)]
        stub_collect_arxiv(monkeypatch, papers)
        stub_screen_arxiv(monkeypatch, {"2611.00001": True, "2611.00002": False})
        summarize_calls: list[str] = []

        async def counting_summarize(agent: object, profile_text: str, paper: ArxivPaper) -> ArxivPaperSummary:
            summarize_calls.append(paper.arxiv_id)
            return ArxivPaperSummary(summary=f"Summary of {paper.title}", why_relevant="matches profile")

        monkeypatch.setattr(cli, "summarize_arxiv_paper", counting_summarize)
        stub_send_mail(monkeypatch)
        saved = stub_notes(monkeypatch)

        exit_code = cli.main_arxiv(["--config", str(config_path)])

        assert exit_code == 0
        row = get_run(state_db, "arxiv", row_report_date(state_db, "arxiv"))
        assert row is not None
        assert row["status"] == "completed"
        assert row["email_sent"] is True
        assert row["note_saved"] is True

        # Only the relevant paper was summarized and rendered
        assert summarize_calls == ["2611.00001"]
        note_body = saved[0][1]
        assert "Paper 1" in note_body
        assert "Paper 2" not in note_body
        assert "1 further paper(s) were screened out" in note_body
        assert "Partial run" not in note_body


class TestArxivPartialFailures:
    """Screening failure excludes silently; summarization failure is partial."""

    def test_screening_failure_treated_as_excluded_exit_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_path = _write_delivery_config(monkeypatch, tmp_path)
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]

        papers = [make_paper(1), make_paper(2)]
        stub_collect_arxiv(monkeypatch, papers)
        stub_screen_arxiv(monkeypatch, {"2611.00002": True}, fail_ids={"2611.00001"})
        stub_summarize_arxiv(monkeypatch)
        stub_send_mail(monkeypatch)
        saved = stub_notes(monkeypatch)

        exit_code = cli.main_arxiv(["--config", str(config_path)])

        # Conservative default already excludes; not a partial run.
        assert exit_code == 0
        row = get_run(state_db, "arxiv", row_report_date(state_db, "arxiv"))
        assert row is not None
        assert row["status"] == "completed"
        note_body = saved[0][1]
        assert "Paper 2" in note_body
        assert "screening failed for 2611.00001" in note_body  # errors section
        assert "Partial run" not in note_body

    def test_summary_failure_marks_partial_and_exits_one(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_path = _write_delivery_config(monkeypatch, tmp_path)
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]

        papers = [make_paper(1)]
        stub_collect_arxiv(monkeypatch, papers)
        stub_screen_arxiv(monkeypatch, {"2611.00001": True})
        stub_summarize_arxiv(monkeypatch, fail_ids={"2611.00001"})
        mail_calls = stub_send_mail(monkeypatch)
        saved = stub_notes(monkeypatch)

        exit_code = cli.main_arxiv(["--config", str(config_path)])

        assert exit_code == 1
        row = get_run(state_db, "arxiv", row_report_date(state_db, "arxiv"))
        assert row is not None
        assert row["status"] == "failed"
        # Deliveries still attempted with a no-results + partial body
        assert row["email_sent"] is True
        assert row["note_saved"] is True
        note_body = saved[0][1]
        assert "No relevant arXiv papers matched" in note_body
        assert "Partial run: some items failed and are omitted." in note_body
        assert len(mail_calls) == 1

    def test_collection_failure_fails_run_without_delivery(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_path = _write_delivery_config(monkeypatch, tmp_path)
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]

        stub_collect_arxiv(monkeypatch, [], fail=True)

        def boom(*args: object, **kwargs: object) -> None:
            raise AssertionError("delivery must not run after collection failure")

        monkeypatch.setattr(cli, "send_mail", boom)
        monkeypatch.setattr(cli, "NotesMcpClient", boom)

        exit_code = cli.main_arxiv(["--config", str(config_path)])

        assert exit_code == 1
        row = get_run(state_db, "arxiv", row_report_date(state_db, "arxiv"))
        assert row is not None
        assert row["status"] == "failed"
        assert row["email_sent"] is False
        assert row["note_saved"] is False


class TestArxivEmptyResults:
    """An empty arXiv day is a successful run with a no-results digest."""

    def test_empty_papers_complete_normally(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_path = _write_delivery_config(monkeypatch, tmp_path)
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]

        stub_collect_arxiv(monkeypatch, [])
        stub_screen_arxiv(monkeypatch, {})
        stub_summarize_arxiv(monkeypatch)
        stub_send_mail(monkeypatch)
        saved = stub_notes(monkeypatch)

        exit_code = cli.main_arxiv(["--config", str(config_path)])

        assert exit_code == 0
        row = get_run(state_db, "arxiv", row_report_date(state_db, "arxiv"))
        assert row is not None
        assert row["status"] == "completed"
        assert "No relevant arXiv papers matched" in saved[0][1]


# --- Window is shared but source-scoped --------------------------------------


class TestSourceIndependence:
    """Each CLI only runs its own source pipeline."""

    def test_reddit_run_does_not_create_arxiv_row(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_path = _write_delivery_config(monkeypatch, tmp_path)
        state_db = make_env(tmp_path)["MATOMERU_STATE_DB"]

        stub_collect_reddit(monkeypatch, [make_thread(1)])
        stub_summarize_reddit(monkeypatch)
        stub_send_mail(monkeypatch)
        stub_notes(monkeypatch)

        assert cli.main_reddit(["--config", str(config_path)]) == 0

        conn = sqlite3.connect(state_db)
        try:
            sources = [r[0] for r in conn.execute("SELECT source FROM runs").fetchall()]
        finally:
            conn.close()
        assert sources == ["reddit"]

    def test_invalid_timezone_is_config_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Bad TZ surfaces as ConfigError before state access."""
        config_path = _write_delivery_config(monkeypatch, tmp_path)
        monkeypatch.setenv("TZ", "Not/AZone")
        monkeypatch.setenv("MATOMERU_STATE_DB", str(tmp_path / "state" / "state.sqlite3"))

        exit_code = cli.main_reddit(["--config", str(config_path)])

        assert exit_code == 1
        assert not (tmp_path / "state" / "state.sqlite3").exists()


# --- Direct call of the former stub API is gone ------------------------------


class TestPublicSurface:
    """The CLI module exposes exactly the documented entry points."""

    def test_entry_points_exist(self) -> None:
        assert callable(cli.main_reddit)
        assert callable(cli.main_arxiv)
        assert not hasattr(cli, "run_source_stub")

    def test_compute_run_window_still_usable_directly(self) -> None:
        window = compute_run_window("UTC")
        assert window.end_utc.tzinfo is UTC
