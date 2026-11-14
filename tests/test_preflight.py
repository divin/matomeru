"""Tests for preflight checks (offline by default, live probes opt-in)."""

from __future__ import annotations

import httpx
import pytest

from matomeru.config import load_app_config, load_runtime_config
from matomeru.preflight import FAIL, PASS, WARN, CheckResult, preflight_ok, run_preflight

MINIMAL_TOML = """[profile]
[[profile.sections]]
name = "About me"
text = "Interested in: exotic particles and slow loris conservation."

[reddit]
subreddits = ["python"]

[arxiv]
categories = []

[notes]
reddit_folder = ""
arxiv_folder = ""
"""

ARXIV_TOML = """[profile]
[[profile.sections]]
name = "About me"
text = "Interested in: exotic particles."

[reddit]
subreddits = ["python"]

[arxiv]
categories = ["cs.LG"]

[notes]
reddit_folder = ""
arxiv_folder = ""
"""


def _config(tmp_path, toml: str, **env_overrides):
    """Build AppConfig + RuntimeConfig from env overrides, like the CLI does."""
    toml_path = tmp_path / "matomeru.toml"
    toml_path.write_text(toml)
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
        **env_overrides,
    }
    for key in ("LLM_API_KEY", "SMTP_USER", "SMTP_PASSWORD", "SAGASHI_MCP_TOKEN", "NOTES_MCP_TOKEN"):
        env.setdefault(key, "")
    return toml_path, env


# --- Offline checks -----------------------------------------------------------


class TestOfflineChecks:
    def test_all_pass_clean_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        toml_path, env = _config(tmp_path, MINIMAL_TOML)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        app = load_app_config(str(toml_path))
        runtime = load_runtime_config()

        checks = run_preflight(app, runtime, "reddit")

        assert preflight_ok(checks)
        by_name = {c.name: c for c in checks}
        assert by_name["state-db"].status == PASS
        assert by_name["smtp-config"].status == PASS
        assert by_name["reranker-config"].status == PASS
        assert by_name["source-config"].status == PASS

    def test_state_db_missing_parents_reported_writable_without_creating(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        toml_path, env = _config(tmp_path, MINIMAL_TOML)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        app = load_app_config(str(toml_path))
        runtime = load_runtime_config()

        checks = run_preflight(app, runtime, "reddit")

        assert preflight_ok(checks)
        assert not (tmp_path / "state").exists()

    def test_state_db_parent_not_writable_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        readonly = tmp_path / "readonly"
        readonly.mkdir()
        readonly.chmod(0o555)
        toml_path, env = _config(tmp_path, MINIMAL_TOML, MATOMERU_STATE_DB=str(readonly / "nested" / "state.sqlite3"))
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        app = load_app_config(str(toml_path))
        runtime = load_runtime_config()

        try:
            checks = run_preflight(app, runtime, "reddit")
            assert not preflight_ok(checks)
            assert next(c for c in checks if c.name == "state-db").status == FAIL
        finally:
            readonly.chmod(0o755)

    def test_smtp_user_without_password_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        toml_path, env = _config(tmp_path, MINIMAL_TOML, SMTP_USER="alice")
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        app = load_app_config(str(toml_path))
        runtime = load_runtime_config()

        checks = run_preflight(app, runtime, "reddit")

        assert not preflight_ok(checks)
        smtp = next(c for c in checks if c.name == "smtp-config")
        assert smtp.status == FAIL
        assert "together" in smtp.detail

    def test_llm_key_missing_warns_not_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        toml_path, env = _config(tmp_path, MINIMAL_TOML)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        app = load_app_config(str(toml_path))
        runtime = load_runtime_config()

        checks = run_preflight(app, runtime, "reddit")

        assert preflight_ok(checks)
        assert next(c for c in checks if c.name == "llm-config").status == WARN

    def test_arxiv_empty_categories_warns(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        toml_path, env = _config(tmp_path, MINIMAL_TOML)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        app = load_app_config(str(toml_path))
        runtime = load_runtime_config()

        checks = run_preflight(app, runtime, "arxiv")

        assert preflight_ok(checks)
        source = next(c for c in checks if c.name == "source-config")
        assert source.status == WARN
        assert "categories" in source.detail

    def test_arxiv_with_categories_passes(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        toml_path, env = _config(tmp_path, ARXIV_TOML)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        app = load_app_config(str(toml_path))
        runtime = load_runtime_config()

        checks = run_preflight(app, runtime, "arxiv")

        assert preflight_ok(checks)
        assert next(c for c in checks if c.name == "source-config").status == PASS

    def test_notes_folder_empty_warns(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        toml_path, env = _config(tmp_path, MINIMAL_TOML)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        app = load_app_config(str(toml_path))
        runtime = load_runtime_config()

        checks = run_preflight(app, runtime, "reddit")

        notes = next(c for c in checks if c.name == "notes-config")
        assert notes.status == WARN
        assert "skipped" in notes.detail


# --- Live probes --------------------------------------------------------------


class TestLiveProbes:
    def _setup(self, monkeypatch: pytest.MonkeyPatch, tmp_path, toml: str = MINIMAL_TOML):
        toml_path, env = _config(tmp_path, toml)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        app = load_app_config(str(toml_path))
        runtime = load_runtime_config()
        return app, runtime

    @staticmethod
    def _stub_network(monkeypatch: pytest.MonkeyPatch, status_code: int = 200) -> None:
        """Stub out all live probes: TCP connect succeeds, HTTP returns a status."""
        import matomeru.preflight as pf

        class FakeSocket:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def __enter__(self) -> FakeSocket:
                return self

            def __exit__(self, *args: object) -> None:
                return None

        class FakeResponse:
            pass

        FakeResponse.status_code = status_code
        monkeypatch.setattr(pf.socket, "create_connection", FakeSocket)
        monkeypatch.setattr(pf.httpx, "get", lambda *a, **kw: FakeResponse())

    def test_live_off_by_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        app, runtime = self._setup(monkeypatch, tmp_path)

        offline = run_preflight(app, runtime, "reddit")

        assert all(not c.name.endswith("-reachable") for c in offline)

    def test_live_smtp_reachable(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        app, runtime = self._setup(monkeypatch, tmp_path)
        self._stub_network(monkeypatch)

        checks = run_preflight(app, runtime, "reddit", live=True)

        assert next(c for c in checks if c.name == "smtp-reachable").status == PASS

    def test_live_smtp_unreachable_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        app, runtime = self._setup(monkeypatch, tmp_path)

        import matomeru.preflight as pf

        def boom(*args: object, **kwargs: object) -> None:
            raise OSError("connection refused")

        monkeypatch.setattr(pf.socket, "create_connection", boom)

        checks = run_preflight(app, runtime, "reddit", live=True)

        assert not preflight_ok(checks)
        smtp = next(c for c in checks if c.name == "smtp-reachable")
        assert smtp.status == FAIL
        assert "refused" in smtp.detail

    def test_live_http_endpoints(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        app, runtime = self._setup(monkeypatch, tmp_path)
        self._stub_network(monkeypatch, status_code=405)

        checks = run_preflight(app, runtime, "reddit", live=True)

        assert preflight_ok(checks)
        names = {c.name for c in checks}
        assert {"sagashi-mcp-reachable", "notes-mcp-reachable", "llm-api-reachable"} <= names
        # 4xx responses still count as reachable
        assert all(c.status == PASS for c in checks if c.name.endswith("-reachable"))

    def test_live_http_unreachable_fails(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        app, runtime = self._setup(monkeypatch, tmp_path)

        import matomeru.preflight as pf

        def boom(*args: object, **kwargs: object) -> None:
            raise httpx.ConnectError("all connection attempts failed")

        monkeypatch.setattr(pf.httpx, "get", boom)

        checks = run_preflight(app, runtime, "reddit", live=True)

        assert not preflight_ok(checks)
        assert all(c.status == FAIL for c in checks if c.name.endswith("-reachable"))

    def test_live_skips_reranker_probe_when_unset(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        app, runtime = self._setup(monkeypatch, tmp_path)
        self._stub_network(monkeypatch)

        checks = run_preflight(app, runtime, "reddit", live=True)

        assert "reranker-reachable" not in {c.name for c in checks}

    def test_live_reranker_probed_when_set(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        toml_path, env = _config(tmp_path, MINIMAL_TOML, RERANKER_URL="http://localhost:8003/rerank")
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        app = load_app_config(str(toml_path))
        runtime = load_runtime_config()

        class FakeResponse:
            status_code = 200

        import matomeru.preflight as pf

        monkeypatch.setattr(pf.httpx, "get", lambda *a, **kw: FakeResponse())

        checks = run_preflight(app, runtime, "reddit", live=True)

        reranker = next(c for c in checks if c.name == "reranker-reachable")
        assert reranker.status == PASS


# --- Helpers ------------------------------------------------------------------


class TestHelpers:
    def test_preflight_ok_false_on_fail(self) -> None:
        checks = [CheckResult("a", PASS, "ok"), CheckResult("b", FAIL, "bad")]
        assert not preflight_ok(checks)

    def test_preflight_ok_true_on_warn(self) -> None:
        checks = [CheckResult("a", WARN, "hmm")]
        assert preflight_ok(checks)

    def test_log_checks_counts_failures(self, caplog: pytest.LogCaptureFixture) -> None:
        from matomeru.preflight import log_checks

        checks = [CheckResult("a", PASS, "ok"), CheckResult("b", FAIL, "bad")]
        with caplog.at_level("INFO"):
            failures = log_checks(checks)
        assert failures == 1
