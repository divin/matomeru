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
class ProfileSection:
    """One interest-profile section from the TOML [[profile.sections]] tables."""

    name: str
    text: str


@dataclass
class AppConfig:
    """Application-level configuration from TOML."""

    config_path: str
    profile_sections: list[ProfileSection]
    reddit_subreddits: list[str]
    reddit_posts_per_subreddit: int
    reddit_comments_per_post: int
    arxiv_categories: list[str]
    arxiv_search_terms: list[str]
    arxiv_category_terms: dict[str, list[str]]
    arxiv_max_results: int
    notes_reddit_folder: str
    notes_arxiv_folder: str
    reranker_reddit_top_x: int = 3
    reranker_arxiv_min_score: float = -6.0
    reranker_min_score: float | None = None


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
    reranker_url: str | None = None
    reranker_token: str | None = None
    state_db_path: str = ""


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

    # Extract required profile section (array-of-tables, not a file path)
    if "profile" not in data:
        raise ConfigError("Configuration file missing required [profile] section")
    profile = data.get("profile", {})
    if "sections" not in profile:
        raise ConfigError("Configuration file [profile] section missing required 'sections'")
    profile_sections = _parse_profile_sections(profile["sections"])

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
    arxiv_category_terms = _parse_category_terms(arxiv.get("category_terms", {}))

    # Extract optional reranker section
    reranker = data.get("reranker", {})
    reranker_reddit_top_x = reranker.get("reddit_top_x", 3)
    reranker_arxiv_min_score = reranker.get("arxiv_min_score", -6.0)
    reranker_min_score = reranker.get("min_score")
    _validate_reranker(
        reranker_reddit_top_x,
        reranker_arxiv_min_score,
        reranker_min_score,
    )

    # Extract notes section
    notes = data.get("notes", {})
    notes_reddit_folder = notes.get("reddit_folder", "")
    notes_arxiv_folder = notes.get("arxiv_folder", "")

    return AppConfig(
        config_path=str(config_file),
        profile_sections=profile_sections,
        reddit_subreddits=subreddits,
        reddit_posts_per_subreddit=posts_per_subreddit,
        reddit_comments_per_post=comments_per_post,
        arxiv_categories=arxiv_categories,
        arxiv_search_terms=arxiv_search_terms,
        arxiv_category_terms=arxiv_category_terms,
        arxiv_max_results=arxiv_max_results,
        reranker_reddit_top_x=reranker_reddit_top_x,
        reranker_arxiv_min_score=reranker_arxiv_min_score,
        reranker_min_score=reranker_min_score,
        notes_reddit_folder=notes_reddit_folder,
        notes_arxiv_folder=notes_arxiv_folder,
    )


def _parse_profile_sections(raw: object) -> list[ProfileSection]:
    """Validate and copy the [[profile.sections]] array-of-tables.

    Every section needs a non-empty name and text; names must be unique and
    at least one section must exist. Trailing whitespace is trimmed from
    text so a multi-line TOML string can be written with the closing
    delimiter on its own line.
    """
    if not isinstance(raw, list):
        raise ConfigError(f"Configuration file [profile] 'sections' must be an array of tables, got: {raw!r}")
    if not raw:
        raise ConfigError("Configuration file [profile] 'sections' must contain at least one section")
    sections: list[ProfileSection] = []
    for i, raw_section in enumerate(raw):
        context = f"Configuration file [profile] section {i}"
        if not isinstance(raw_section, dict):
            raise ConfigError(f"{context} must be a table, got: {raw_section!r}")
        name = raw_section.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"{context} needs a non-empty 'name'")
        text = raw_section.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ConfigError(f"{context} needs a non-empty 'text'")
        sections.append(ProfileSection(name=name, text=text.rstrip()))
    names = [section.name for section in sections]
    if len(names) != len(set(names)):
        raise ConfigError("Configuration file [profile] 'sections' names must be unique")
    return sections


def _validate_reranker(top_x: object, arxiv_min_score: object, min_score: object) -> None:
    """Validate the [reranker] table fields."""
    if not isinstance(top_x, int) or isinstance(top_x, bool) or top_x < 1:
        raise ConfigError(f"Configuration file [reranker] 'reddit_top_x' must be a positive integer, got: {top_x!r}")
    if not isinstance(arxiv_min_score, (int, float)) or isinstance(arxiv_min_score, bool):
        raise ConfigError(f"Configuration file [reranker] 'arxiv_min_score' must be a number, got: {arxiv_min_score!r}")
    if min_score is not None and (not isinstance(min_score, (int, float)) or isinstance(min_score, bool)):
        raise ConfigError(f"Configuration file [reranker] 'min_score' must be null or a number, got: {min_score!r}")


def render_profile_markdown(sections: list[ProfileSection]) -> str:
    """Render the profile sections as the Markdown text the LLM sees.

    Produces ``# Interest profile\n\n## {name}\n\n{text}\n`` blocks joined by
    a blank line — the same content the old profile.md file delivered, so
    summarization quality is unchanged.
    """
    if not sections:
        return "# Interest profile\n\n"
    blocks: list[str] = [f"# Interest profile\n\n## {sections[0].name}\n\n{sections[0].text}\n"]
    blocks.extend(f"## {section.name}\n\n{section.text}\n" for section in sections[1:])
    return "\n".join(blocks)


def _parse_category_terms(raw: object) -> dict[str, list[str]]:
    """Validate and copy the [arxiv.category_terms] table.

    Keys are stored exactly as written (normalization of category names
    happens in arxiv.py). Every key must be a non-empty string and every
    value a list of strings; violations raise ConfigError naming the key.
    """
    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration file [arxiv] 'category_terms' must be a table, got: {raw!r}")
    category_terms: dict[str, list[str]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise ConfigError(
                f"Configuration file [arxiv] 'category_terms' keys must be non-empty strings, got: {key!r}"
            )
        if not isinstance(value, list) or not all(isinstance(term, str) for term in value):
            raise ConfigError(
                f"Configuration file [arxiv] 'category_terms' entry {key!r} must be a list of strings, got: {value!r}"
            )
        category_terms[key] = list(value)
    return category_terms


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
        RERANKER_URL (None) - reranker endpoint URL; unset skips the funnel
        RERANKER_TOKEN (None) - bearer token for the reranker endpoint
        LLM_API_KEY (None)
        SMTP_USER (empty string)
        SMTP_PASSWORD (empty string)
        MATOMERU_STATE_DB (~/.local/state/matomeru/state.sqlite3)

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
        state_db_path=_optional_env("MATOMERU_STATE_DB") or state_db_default,
        reranker_url=_optional_env("RERANKER_URL"),
        reranker_token=_optional_env("RERANKER_TOKEN"),
    )
