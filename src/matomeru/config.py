"""Configuration loading for Matomeru."""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


@dataclass
class AppConfig:
    """Application-level configuration from TOML."""

    config_path: str
    profile_path: str
    reddit_subreddits: list[str]
    reddit_posts_per_subreddit: int
    reddit_comments_per_post: int
    arxiv_categories: list[str]
    arxiv_search_terms: list[str]
    arxiv_max_results: int
    notes_reddit_folder: str
    notes_arxiv_folder: str


@dataclass
class RuntimeConfig:
    """Runtime configuration from environment."""

    sagashi_mcp_url: str
    sagashi_mcp_token: str | None
    notes_mcp_url: str
    notes_mcp_token: str | None
    llm_base_url: str
    llm_api_key: str | None
    llm_model: str
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    smtp_from: str
    smtp_to: str
    tz_iana: str
    state_db_path: str
    arxiv_api_base: str


def load_app_config(path: str | None = None) -> AppConfig:
    """Load application configuration from TOML.

    Path resolution:
    - Explicit path argument
    - MATOMERU_CONFIG environment variable
    - matomeru.toml in current working directory

    Raises:
        ConfigError: If file is missing, unreadable, or missing required fields.
    """
    if path:
        config_file = Path(path)
    else:
        env_path = os.environ.get("MATOMERU_CONFIG")
        if env_path:
            config_file = Path(env_path)
        else:
            config_file = Path.cwd() / "matomeru.toml"

    if not config_file.exists():
        raise ConfigError(f"Configuration file not found: {config_file}")

    try:
        with config_file.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Failed to parse configuration file {config_file}: {e}") from e
    except OSError as e:
        raise ConfigError(f"Failed to read configuration file {config_file}: {e}") from e

    # Extract required profile section
    if "profile" not in data:
        raise ConfigError("Configuration file missing required [profile] section")
    profile = data.get("profile", {})
    if "path" not in profile:
        raise ConfigError("Configuration file [profile] section missing 'path' field")

    profile_path = profile["path"]

    # Extract Reddit section with defaults
    reddit = data.get("reddit", {})
    subreddits = reddit.get("subreddits", [])
    if not subreddits:
        raise ConfigError("Configuration file [reddit] section missing required 'subreddits' field")
    posts_per_subreddit = reddit.get("posts_per_subreddit", 3)
    comments_per_post = reddit.get("comments_per_post", 20)

    # Extract arXiv section with defaults
    arxiv = data.get("arxiv", {})
    arxiv_categories = arxiv.get("categories", [])
    arxiv_search_terms = arxiv.get("search_terms", [])
    arxiv_max_results = arxiv.get("max_results", 100)
    if not isinstance(arxiv_max_results, int) or isinstance(arxiv_max_results, bool) or arxiv_max_results < 1:
        raise ConfigError(
            f"Configuration file [arxiv] 'max_results' must be a positive integer, got: {arxiv_max_results!r}"
        )

    # Extract notes section
    notes = data.get("notes", {})
    notes_reddit_folder = notes.get("reddit_folder", "")
    notes_arxiv_folder = notes.get("arxiv_folder", "")

    return AppConfig(
        config_path=str(config_file),
        profile_path=profile_path,
        reddit_subreddits=subreddits,
        reddit_posts_per_subreddit=posts_per_subreddit,
        reddit_comments_per_post=comments_per_post,
        arxiv_categories=arxiv_categories,
        arxiv_search_terms=arxiv_search_terms,
        arxiv_max_results=arxiv_max_results,
        notes_reddit_folder=notes_reddit_folder,
        notes_arxiv_folder=notes_arxiv_folder,
    )


def _optional_env(name: str) -> str | None:
    """Read an optional env var, treating unset and empty as None."""
    value = os.environ.get(name)
    return value if value else None


def _parse_smtp_port(value: str) -> int:
    """Parse SMTP_PORT, converting a bad value into a ConfigError."""
    try:
        port = int(value)
    except ValueError as e:
        raise ConfigError(f"SMTP_PORT must be an integer, got: {value!r}") from e
    if not 1 <= port <= 65535:
        raise ConfigError(f"SMTP_PORT must be between 1 and 65535, got: {value!r}")
    return port


def load_runtime_config() -> RuntimeConfig:
    """Load runtime configuration from environment variables.

    Required: SAGASHI_MCP_URL, NOTES_MCP_URL, LLM_BASE_URL, LLM_MODEL,
              SMTP_HOST, SMTP_PORT, SMTP_FROM, SMTP_TO, TZ

    Optional with defaults:
        SAGASHI_MCP_TOKEN (None) - bearer token for the Sagashi MCP server
        NOTES_MCP_TOKEN (None) - bearer token for the Notes MCP server
        LLM_API_KEY (None)
        SMTP_USER (empty string)
        SMTP_PASSWORD (empty string)
        MATOMERU_STATE_DB (~/.local/state/matomeru/state.sqlite3)
        ARXIV_API_BASE (http://export.arxiv.org) - base URL of the arXiv API

    Raises:
        ConfigError: If any required environment variable is missing.
    """
    load_dotenv()

    required_vars = [
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

    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

    state_db_default = str(Path.home() / ".local" / "state" / "matomeru" / "state.sqlite3")

    return RuntimeConfig(
        sagashi_mcp_url=os.environ["SAGASHI_MCP_URL"],
        sagashi_mcp_token=_optional_env("SAGASHI_MCP_TOKEN"),
        notes_mcp_url=os.environ["NOTES_MCP_URL"],
        notes_mcp_token=_optional_env("NOTES_MCP_TOKEN"),
        llm_base_url=os.environ["LLM_BASE_URL"],
        llm_api_key=_optional_env("LLM_API_KEY"),
        llm_model=os.environ["LLM_MODEL"],
        smtp_host=os.environ["SMTP_HOST"],
        smtp_port=_parse_smtp_port(os.environ["SMTP_PORT"]),
        smtp_user=os.environ.get("SMTP_USER", ""),
        smtp_password=os.environ.get("SMTP_PASSWORD", ""),
        smtp_from=os.environ["SMTP_FROM"],
        smtp_to=os.environ["SMTP_TO"],
        tz_iana=os.environ["TZ"],
        state_db_path=os.environ.get("MATOMERU_STATE_DB", state_db_default),
        arxiv_api_base=os.environ.get("ARXIV_API_BASE", "http://export.arxiv.org"),
    )


def load_profile_text(app: AppConfig) -> str:
    """Load the profile Markdown text.

    Args:
        app: Application configuration with profile_path.

    Returns:
        Profile file contents as string.

    Raises:
        ConfigError: If profile file is missing or unreadable.
    """
    profile_path = Path(app.profile_path)
    if not profile_path.exists():
        raise ConfigError(f"Profile file not found: {profile_path}")
    try:
        return profile_path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"Failed to read profile file {profile_path}: {e}") from e
