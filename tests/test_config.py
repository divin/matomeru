"""Tests for Matomeru configuration and logging."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from matomeru.config import (
    ConfigError,
    ProfileSection,
    RuntimeConfig,
    _parse_category_terms,
    load_app_config,
    load_runtime_config,
    render_profile_markdown,
)
from matomeru.logging_setup import configure_logging

# TOML content for tests


class TestLoadAppConfig:
    """Tests for load_app_config."""

    def test_happy_path(self, tmp_path: Path) -> None:
        """TOML parsing happy path with all fields."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
[[profile.sections]]
name = "About me"
text = "Python, Rust, AI"

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
        assert config.profile_sections[0].name == "About me"
        assert config.profile_sections[0].text == "Python, Rust, AI"
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
[[profile.sections]]
name = "About me"
text = "Python, Rust, AI"

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
[[profile.sections]]
name = "About me"
text = "Python, Rust, AI"

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
[[profile.sections]]
name = "About me"
text = "Python, Rust, AI"

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
[[profile.sections]]
name = "About me"
text = "Python, Rust, AI"

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
[[profile.sections]]
name = "About me"
text = "Python, Rust, AI"

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
[[profile.sections]]
name = "About me"
text = "Python, Rust, AI"

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
[[profile.sections]]
name = "About me"
text = "Python, Rust, AI"

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
[[profile.sections]]
name = "About me"
text = "Python, Rust, AI"

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
[[profile.sections]]
name = "About me"
text = "Python, Rust, AI"

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

    def test_missing_required_profile_section(self, tmp_path: Path) -> None:
        """Missing [profile] section raises ConfigError."""
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

    def test_missing_profile_sections(self, tmp_path: Path) -> None:
        """A [profile] section without any section raises ConfigError."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]

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
        with pytest.raises(ConfigError) as exc_info:
            load_app_config(str(toml_path))
        assert "missing required 'sections'" in str(exc_info.value)

    def test_profile_sections_defaults(self, tmp_path: Path) -> None:
        """profile_sections is parsed from the TOML array-of-tables."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
[[profile.sections]]
name = "About me"
text = "Python, Rust, AI"

[[profile.sections]]
name = "Hobbies"
text = "Gaming"

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
        assert [s.name for s in config.profile_sections] == ["About me", "Hobbies"]
        assert [s.text for s in config.profile_sections] == ["Python, Rust, AI", "Gaming"]

    def test_profile_sections_empty_name(self, tmp_path: Path) -> None:
        """An empty section name raises ConfigError."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
[[profile.sections]]
name = ""
text = "Python, Rust, AI"

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
        with pytest.raises(ConfigError) as exc_info:
            load_app_config(str(toml_path))
        assert "needs a non-empty 'name'" in str(exc_info.value)

    def test_profile_sections_empty_text(self, tmp_path: Path) -> None:
        """An empty section text raises ConfigError."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
[[profile.sections]]
name = "About me"
text = ""

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
        with pytest.raises(ConfigError) as exc_info:
            load_app_config(str(toml_path))
        assert "needs a non-empty 'text'" in str(exc_info.value)

    def test_profile_sections_duplicate_name(self, tmp_path: Path) -> None:
        """Duplicate section names raise ConfigError."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
[[profile.sections]]
name = "About me"
text = "Python, Rust, AI"

[[profile.sections]]
name = "About me"
text = "Gaming"

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
        with pytest.raises(ConfigError) as exc_info:
            load_app_config(str(toml_path))
        assert "names must be unique" in str(exc_info.value)

    def test_reranker_defaults(self, tmp_path: Path) -> None:
        """[reranker] defaults apply when the table is absent."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
[[profile.sections]]
name = "About me"
text = "Python, Rust, AI"

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
        assert config.reranker_reddit_top_x == 3
        assert config.reranker_arxiv_min_score == -6.0
        assert config.reranker_min_score is None

    def test_reranker_overrides(self, tmp_path: Path) -> None:
        """[reranker] overrides are applied."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
[[profile.sections]]
name = "About me"
text = "Python, Rust, AI"

[reddit]
subreddits = ["python"]

[arxiv]
categories = []
search_terms = []

[reranker]
reddit_top_x = 5
arxiv_min_score = -5.5
min_score = -8.0

[notes]
reddit_folder = ""
arxiv_folder = ""
"""
        )
        config = load_app_config(str(toml_path))
        assert config.reranker_reddit_top_x == 5
        assert config.reranker_arxiv_min_score == -5.5
        assert config.reranker_min_score == -8.0

    def test_reranker_invalid_reddit_top_x(self, tmp_path: Path) -> None:
        """A non-positive reddit_top_x raises ConfigError."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
[[profile.sections]]
name = "About me"
text = "Python, Rust, AI"

[reddit]
subreddits = ["python"]

[arxiv]
categories = []
search_terms = []

[reranker]
reddit_top_x = 0

[notes]
reddit_folder = ""
arxiv_folder = ""
"""
        )
        with pytest.raises(ConfigError) as exc_info:
            load_app_config(str(toml_path))
        assert "reddit_top_x" in str(exc_info.value)

    def test_env_var_path_resolution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """MATOMERU_CONFIG env var is used when no explicit path."""
        toml_path = tmp_path / "matomeru.toml"
        toml_path.write_text(
            """[profile]
[[profile.sections]]
name = "About me"
text = "Python, Rust, AI"

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
[[profile.sections]]
name = "About me"
text = "Python, Rust, AI"

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
            "RERANKER_URL": "http://reranker:9093/v1/rerank",
            "RERANKER_TOKEN": "rerank-secret",
        }
        for k, v in env.items():
            monkeypatch.setenv(k, v)

        config = load_runtime_config()

        assert isinstance(config, RuntimeConfig)
        assert config.reranker_url == "http://reranker:9093/v1/rerank"
        assert config.reranker_token == "rerank-secret"
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
        assert config.reranker_url is None
        assert config.reranker_token is None

    def test_state_db_empty_env_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicitly empty MATOMERU_STATE_DB falls back to the default path."""
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
            "MATOMERU_STATE_DB": "",
        }
        for k, v in env.items():
            monkeypatch.setenv(k, v)

        config = load_runtime_config()

        expected = str(Path.home() / ".local" / "state" / "matomeru" / "state.sqlite3")
        assert config.state_db_path == expected

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


class TestRenderProfileMarkdown:
    """Tests for render_profile_markdown."""

    def test_render_exact_output_single_section(self) -> None:
        """The exact output matches the profile.md format for one section."""
        sections = [ProfileSection(name="About me", text="Python, Rust, AI")]

        rendered = render_profile_markdown(sections)

        assert rendered == "# Interest profile\n\n## About me\n\nPython, Rust, AI\n"

    def test_render_exact_output_multiple_sections(self) -> None:
        """The exact output matches the profile.md format for multiple sections."""
        sections = [
            ProfileSection(name="About me", text="Python, Rust, AI"),
            ProfileSection(name="Hobbies", text="Gaming\n- factorio"),
        ]

        rendered = render_profile_markdown(sections)

        assert rendered == (
            "# Interest profile\n\n## About me\n\nPython, Rust, AI\n\n## Hobbies\n\nGaming\n- factorio\n"
        )


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
        monkeypatch.setenv("RERANKER_URL", "")
        monkeypatch.setenv("RERANKER_TOKEN", "")

        runtime = load_runtime_config()

        assert runtime.sagashi_mcp_token is None
        assert runtime.notes_mcp_token is None
        assert runtime.llm_api_key is None
        assert runtime.reranker_url is None
        assert runtime.reranker_token is None

    def test_unset_optionals_remain_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._full_env(monkeypatch)
        monkeypatch.delenv("SAGASHI_MCP_TOKEN", raising=False)
        monkeypatch.delenv("NOTES_MCP_TOKEN", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("RERANKER_URL", raising=False)
        monkeypatch.delenv("RERANKER_TOKEN", raising=False)

        runtime = load_runtime_config()

        assert runtime.sagashi_mcp_token is None
        assert runtime.notes_mcp_token is None
        assert runtime.llm_api_key is None
        assert runtime.reranker_url is None
        assert runtime.reranker_token is None
