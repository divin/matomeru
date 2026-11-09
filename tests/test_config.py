"""Tests for Matomeru configuration and logging."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from matomeru.config import (
    AppConfig,
    ConfigError,
    RuntimeConfig,
    _parse_category_terms,
    load_app_config,
    load_profile_text,
    load_runtime_config,
)
from matomeru.logging_setup import configure_logging


# TOML content for tests
def make_minimal_toml(profile_path: str, subreddits: list[str]) -> str:
    """Generate minimal valid TOML."""
    subreddits_str = ", ".join(f'"{s}"' for s in subreddits)
    return f"""[profile]
path = "{profile_path}"

[reddit]
subreddits = [{subreddits_str}]

[arxiv]
categories = []
search_terms = []

[notes]
reddit_folder = ""
arxiv_folder = ""
"""


class TestLoadAppConfig:
    """Tests for load_app_config."""

    def test_happy_path(self, tmp_path: Path) -> None:
        """TOML parsing happy path with all fields."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
path = "/home/user/profile.md"

[reddit]
subreddits = ["python", "rust"]
posts_per_subreddit = 5
comments_per_post = 30

[arxiv]
categories = ["cs.AI", "cs.LG"]
search_terms = ["machine learning", "LLM"]

[notes]
reddit_folder = "Reddit"
arxiv_folder = "arXiv"
"""
        )
        config = load_app_config(str(toml_path))

        assert config.config_path == str(toml_path)
        assert config.profile_path == "/home/user/profile.md"
        assert config.reddit_subreddits == ["python", "rust"]
        assert config.reddit_posts_per_subreddit == 5
        assert config.reddit_comments_per_post == 30
        assert config.arxiv_categories == ["cs.AI", "cs.LG"]
        assert config.arxiv_search_terms == ["machine learning", "LLM"]
        assert config.arxiv_max_results == 100
        assert config.notes_reddit_folder == "Reddit"
        assert config.notes_arxiv_folder == "arXiv"

    def test_defaults_applied(self, tmp_path: Path) -> None:
        """Default values are applied when omitted."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
path = "/home/user/profile.md"

[reddit]
subreddits = ["python"]

[arxiv]
categories = []
search_terms = []

[notes]
reddit_folder = ""
arxiv_folder = ""
"""
        )
        config = load_app_config(str(toml_path))

        assert config.reddit_posts_per_subreddit == 3
        assert config.reddit_comments_per_post == 20
        assert config.arxiv_categories == []
        assert config.arxiv_search_terms == []
        assert config.arxiv_max_results == 100

    def test_arxiv_max_results_default_and_override(self, tmp_path: Path) -> None:
        """[arxiv] max_results defaults to 100 and can be overridden."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
path = "/home/user/profile.md"

[reddit]
subreddits = ["python"]

[arxiv]
categories = []
search_terms = []
max_results = 250

[notes]
reddit_folder = ""
arxiv_folder = ""
"""
        )
        config = load_app_config(str(toml_path))
        assert config.arxiv_max_results == 250

    def test_arxiv_max_results_invalid(self, tmp_path: Path) -> None:
        """Non-positive or non-integer max_results raises ConfigError."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
path = "/home/user/profile.md"

[reddit]
subreddits = ["python"]

[arxiv]
categories = []
search_terms = []
max_results = 0

[notes]
reddit_folder = ""
arxiv_folder = ""
"""
        )
        with pytest.raises(ConfigError) as exc_info:
            load_app_config(str(toml_path))
        assert "max_results" in str(exc_info.value)

    def test_arxiv_category_terms_happy_path(self, tmp_path: Path) -> None:
        """[arxiv.category_terms] parses into a dict with raw keys kept as-is."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
path = "/home/user/profile.md"

[reddit]
subreddits = ["python"]

[arxiv]
categories = ["cs.AI", "cs.LG", "cs.CV"]
search_terms = []

[arxiv.category_terms]
"cs.AI" = ["large language model", "agent"]
"cat:cs.LG" = ["graph neural network"]
"cs.CV" = []

[notes]
reddit_folder = ""
arxiv_folder = ""
"""
        )
        config = load_app_config(str(toml_path))

        # Keys are stored raw (no lowercasing, no cat: stripping)
        assert config.arxiv_category_terms == {
            "cs.AI": ["large language model", "agent"],
            "cat:cs.LG": ["graph neural network"],
            "cs.CV": [],
        }

    def test_arxiv_category_terms_default_empty(self, tmp_path: Path) -> None:
        """Absent [arxiv.category_terms] section yields {}."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
path = "/home/user/profile.md"

[reddit]
subreddits = ["python"]

[arxiv]
categories = []
search_terms = []

[notes]
reddit_folder = ""
arxiv_folder = ""
"""
        )
        config = load_app_config(str(toml_path))
        assert config.arxiv_category_terms == {}

    def test_arxiv_category_terms_non_table(self, tmp_path: Path) -> None:
        """A non-table category_terms raises ConfigError."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
path = "/home/user/profile.md"

[reddit]
subreddits = ["python"]

[arxiv]
categories = []
search_terms = []
category_terms = "oops"

[notes]
reddit_folder = ""
arxiv_folder = ""
"""
        )
        with pytest.raises(ConfigError) as exc_info:
            load_app_config(str(toml_path))
        assert "category_terms" in str(exc_info.value)

    def test_arxiv_category_terms_value_not_list(self, tmp_path: Path) -> None:
        """A non-list value raises ConfigError naming the key."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
path = "/home/user/profile.md"

[reddit]
subreddits = ["python"]

[arxiv]
categories = []
search_terms = []

[arxiv.category_terms]
"cs.AI" = "transformer"

[notes]
reddit_folder = ""
arxiv_folder = ""
"""
        )
        with pytest.raises(ConfigError) as exc_info:
            load_app_config(str(toml_path))
        assert "cs.AI" in str(exc_info.value)

    def test_arxiv_category_terms_value_non_string_items(self, tmp_path: Path) -> None:
        """A list with non-string items raises ConfigError naming the key."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
path = "/home/user/profile.md"

[reddit]
subreddits = ["python"]

[arxiv]
categories = []
search_terms = []

[arxiv.category_terms]
"cs.AI" = ["transformer", 42]

[notes]
reddit_folder = ""
arxiv_folder = ""
"""
        )
        with pytest.raises(ConfigError) as exc_info:
            load_app_config(str(toml_path))
        assert "cs.AI" in str(exc_info.value)

    def test_arxiv_category_terms_non_string_key_rejected(self) -> None:
        """A non-string key raises ConfigError naming the key.

        TOML keys are always strings, so this is only reachable via the
        parsing helper directly (defensive validation).
        """
        with pytest.raises(ConfigError) as exc_info:
            _parse_category_terms({42: ["transformer"]})
        assert "42" in str(exc_info.value)

    def test_arxiv_category_terms_empty_key_rejected(self) -> None:
        """An empty-string key raises ConfigError naming the key."""
        with pytest.raises(ConfigError) as exc_info:
            _parse_category_terms({"": ["transformer"]})
        assert "''" in str(exc_info.value) or '""' in str(exc_info.value)

    def test_missing_toml_file(self, tmp_path: Path) -> None:
        """Missing TOML file raises ConfigError."""
        nonexistent = tmp_path / "nonexistent.toml"
        with pytest.raises(ConfigError) as exc_info:
            load_app_config(str(nonexistent))
        assert "Configuration file not found" in str(exc_info.value)

    def test_missing_required_reddit_subreddits(self, tmp_path: Path) -> None:
        """Missing subreddits field raises ConfigError."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
path = "/home/user/profile.md"

[arxiv]
categories = []
search_terms = []

[notes]
reddit_folder = ""
arxiv_folder = ""
"""
        )
        with pytest.raises(ConfigError) as exc_info:
            load_app_config(str(toml_path))
        assert "missing required 'subreddits' field" in str(exc_info.value)

    def test_missing_required_profile_path(self, tmp_path: Path) -> None:
        """Missing profile path raises ConfigError."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[reddit]
subreddits = ["python"]

[arxiv]
categories = []
search_terms = []

[notes]
reddit_folder = ""
arxiv_folder = ""
"""
        )
        with pytest.raises(ConfigError) as exc_info:
            load_app_config(str(toml_path))
        assert "missing required [profile] section" in str(exc_info.value)

    def test_env_var_path_resolution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """MATOMERU_CONFIG env var is used when no explicit path."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
path = "/home/user/profile.md"

[reddit]
subreddits = ["python"]

[arxiv]
categories = []
search_terms = []

[notes]
reddit_folder = ""
arxiv_folder = ""
"""
        )
        monkeypatch.setenv("MATOMERU_CONFIG", str(toml_path))

        config = load_app_config()
        assert config.config_path == str(toml_path)

    def test_cwd_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falls back to matomeru.toml in cwd when no path and no env var."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
path = "/home/user/profile.md"

[reddit]
subreddits = ["python"]

[arxiv]
categories = []
search_terms = []

[notes]
reddit_folder = ""
arxiv_folder = ""
"""
        )
        monkeypatch.chdir(tmp_path)
        config = load_app_config()
        assert config.config_path == str(toml_path)


class TestLoadRuntimeConfig:
    """Tests for load_runtime_config."""

    def test_full_env_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """load_runtime_config succeeds with a full env set."""
        env = {
            "SAGASHI_MCP_URL": "http://sagashi:8000",
            "NOTES_MCP_URL": "http://notes:9008/mcp",
            "LLM_BASE_URL": "http://localhost:11434/v1",
            "LLM_MODEL": "llama3",
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "587",
            "SMTP_FROM": "matomeru@example.com",
            "SMTP_TO": "user@example.com",
            "TZ": "America/New_York",
            "LLM_API_KEY": "sk-1234",
            "SMTP_USER": "matomeru",
            "SMTP_PASSWORD": "secret",
        }
        for k, v in env.items():
            monkeypatch.setenv(k, v)

        config = load_runtime_config()

        assert isinstance(config, RuntimeConfig)
        assert config.sagashi_mcp_url == "http://sagashi:8000"
        assert config.notes_mcp_url == "http://notes:9008/mcp"
        assert config.llm_base_url == "http://localhost:11434/v1"
        assert config.llm_model == "llama3"
        assert config.smtp_host == "smtp.example.com"
        assert config.smtp_port == 587
        assert config.smtp_from == "matomeru@example.com"
        assert config.smtp_to == "user@example.com"
        assert config.tz_iana == "America/New_York"
        assert config.llm_api_key == "sk-1234"
        assert config.smtp_user == "matomeru"
        assert config.smtp_password == "secret"
        assert str(config.state_db_path).endswith("matomeru/state.sqlite3")

    def test_state_db_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MATOMERU_STATE_DB defaults to ~/.local/state/matomeru/state.sqlite3."""
        env = {
            "SAGASHI_MCP_URL": "http://sagashi:8000",
            "NOTES_MCP_URL": "http://notes:9008/mcp",
            "LLM_BASE_URL": "http://localhost:11434/v1",
            "LLM_MODEL": "llama3",
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "587",
            "SMTP_FROM": "matomeru@example.com",
            "SMTP_TO": "user@example.com",
            "TZ": "America/New_York",
        }
        for k, v in env.items():
            monkeypatch.setenv(k, v)

        config = load_runtime_config()

        expected = str(Path.home() / ".local" / "state" / "matomeru" / "state.sqlite3")
        assert config.state_db_path == expected
        assert config.sagashi_mcp_token is None
        assert config.notes_mcp_token is None

    def test_notes_mcp_token_amendment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NOTES_MCP_TOKEN is optional and defaults to None (phase 5a amendment)."""
        env = {
            "SAGASHI_MCP_URL": "http://sagashi:8000",
            "NOTES_MCP_URL": "http://notes:9008/mcp",
            "LLM_BASE_URL": "http://localhost:11434/v1",
            "LLM_MODEL": "llama3",
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "587",
            "SMTP_FROM": "matomeru@example.com",
            "SMTP_TO": "user@example.com",
            "TZ": "America/New_York",
        }
        for k, v in env.items():
            monkeypatch.setenv(k, v)

        config = load_runtime_config()
        assert config.notes_mcp_token is None

        monkeypatch.setenv("NOTES_MCP_TOKEN", "notes-secret")
        config = load_runtime_config()
        assert config.notes_mcp_token == "notes-secret"

    def test_missing_required_vars_reported_together(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing required env vars are reported together."""
        env = {
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "587",
        }
        for k, v in env.items():
            monkeypatch.setenv(k, v)

        with pytest.raises(ConfigError) as exc_info:
            load_runtime_config()
        assert "Missing required environment variables" in str(exc_info.value)
        assert "SAGASHI_MCP_URL" in str(exc_info.value)
        assert "NOTES_MCP_URL" in str(exc_info.value)
        assert "LLM_BASE_URL" in str(exc_info.value)
        assert "LLM_MODEL" in str(exc_info.value)
        assert "SMTP_FROM" in str(exc_info.value)
        assert "SMTP_TO" in str(exc_info.value)
        assert "TZ" in str(exc_info.value)


class TestLoadProfileText:
    """Tests for load_profile_text."""

    def test_happy_path(self, tmp_path: Path) -> None:
        """Profile file is read successfully."""
        profile_path = tmp_path / "profile.md"
        profile_path.write_text("# Interests\n\nPython, Rust, AI")
        app = AppConfig(
            config_path=str(tmp_path / "matomeru.toml"),
            profile_path=str(profile_path),
            reddit_subreddits=["python"],
            reddit_posts_per_subreddit=3,
            reddit_comments_per_post=20,
            arxiv_categories=[],
            arxiv_search_terms=[],
            arxiv_max_results=100,
            arxiv_category_terms={},
            notes_reddit_folder="",
            notes_arxiv_folder="",
        )

        profile = load_profile_text(app)

        assert "# Interests" in profile
        assert "Python, Rust, AI" in profile

    def test_missing_profile_file(self, tmp_path: Path) -> None:
        """Missing profile file raises ConfigError."""
        app = AppConfig(
            config_path=str(tmp_path / "matomeru.toml"),
            profile_path=str(tmp_path / "nonexistent.md"),
            reddit_subreddits=["python"],
            reddit_posts_per_subreddit=3,
            reddit_comments_per_post=20,
            arxiv_categories=[],
            arxiv_search_terms=[],
            arxiv_max_results=100,
            arxiv_category_terms={},
            notes_reddit_folder="",
            notes_arxiv_folder="",
        )

        with pytest.raises(ConfigError) as exc_info:
            load_profile_text(app)
        assert "Profile file not found" in str(exc_info.value)


class TestConfigureLogging:
    """Tests for configure_logging."""

    def test_valid_log_level(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Valid log level is applied."""
        import os

        os.environ["MATOMERU_LOG_LEVEL"] = "DEBUG"
        configure_logging()

        logger = logging.getLogger("test")
        logger.debug("test message")

        captured = capsys.readouterr()
        assert "DEBUG" in captured.err
        assert "test message" in captured.err

    def test_invalid_log_level_falls_back(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Invalid log level falls back to INFO with warning."""
        import os

        os.environ["MATOMERU_LOG_LEVEL"] = "INVALID"
        configure_logging()

        captured = capsys.readouterr()
        assert "falling back to INFO" in captured.err

    def test_default_log_level_is_info(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Default log level is INFO when env var is absent."""
        import os

        os.environ.pop("MATOMERU_LOG_LEVEL", None)
        configure_logging()

        logger = logging.getLogger("test3")
        logger.info("info message")

        captured = capsys.readouterr()
        assert "INFO" in captured.err
        assert "info message" in captured.err


class TestOptionalEnvVars:
    """Empty-string optional env vars must behave like unset ones."""

    def _full_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        values = {
            "SAGASHI_MCP_URL": "http://sagashi/mcp",
            "NOTES_MCP_URL": "http://notes/mcp",
            "LLM_BASE_URL": "http://llm/v1",
            "LLM_MODEL": "test-model",
            "SMTP_HOST": "localhost",
            "SMTP_PORT": "25",
            "SMTP_FROM": "a@b.c",
            "SMTP_TO": "d@e.f",
            "TZ": "UTC",
        }
        for key, value in values.items():
            monkeypatch.setenv(key, value)

    def test_empty_tokens_and_key_become_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicitly empty SAGASHI_MCP_TOKEN / NOTES_MCP_TOKEN / LLM_API_KEY yield None."""
        self._full_env(monkeypatch)
        monkeypatch.setenv("SAGASHI_MCP_TOKEN", "")
        monkeypatch.setenv("NOTES_MCP_TOKEN", "")
        monkeypatch.setenv("LLM_API_KEY", "")

        runtime = load_runtime_config()

        assert runtime.sagashi_mcp_token is None
        assert runtime.notes_mcp_token is None
        assert runtime.llm_api_key is None

    def test_unset_optionals_remain_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._full_env(monkeypatch)
        monkeypatch.delenv("SAGASHI_MCP_TOKEN", raising=False)
        monkeypatch.delenv("NOTES_MCP_TOKEN", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        runtime = load_runtime_config()

        assert runtime.sagashi_mcp_token is None
        assert runtime.notes_mcp_token is None
        assert runtime.llm_api_key is None
