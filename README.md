# Matomeru

**Matomeru** (まとめる, “to bring together / organize”) is planned as a cron-driven Python package for producing daily Reddit and arXiv digests, emailing each digest, and saving it to a notes vault.

> **Status: planning.** The repository is currently the generated Python package scaffold. The commands and integrations below describe the agreed design; they are not implemented yet.

## Planned commands

- `uv run matomeru-reddit` — collect configured top posts from selected subreddits, fetch their threads and top comments through Sagashi, summarize the posts and discussion, and explain their relevance to the profile.
- `uv run matomeru-arxiv` — find new papers matching configured arXiv categories/searches, screen abstracts against the profile, and summarize relevant papers.

The commands run independently. Each sends its own email and writes one dated note to its configured notes folder. They are intended to be invoked by cron; Matomeru is not a long-running service and does not require Docker.

## Agreed design

- Each invocation processes the 24 elapsed hours ending when it starts. `TZ` from `.env` determines the report date and local display; cron's timezone must be configured consistently on the server.
- A shared Markdown profile will guide both sources, with general and source-specific sections.
- A local OpenAI-compatible LLM endpoint will be called through PydanticAI with structured outputs.
- Reddit collection will use Sagashi's general-purpose MCP tools: extend `search_reddit` with an optional subreddit filter, add a general subreddit-post listing tool, and add a batch thread-fetch tool. Matomeru will compose these primitives; Sagashi will not contain daily-digest or profile logic. Use the official Reddit API and AsyncPRAW's rate-limit handling, without an arbitrary fixed cooldown.
- Notes will be written through the Notes MCP server to the separate Reddit and arXiv folders you create. The Notes MCP `create_note` tool supports creating or updating a note at a path.
- A local SQLite state file will support run tracking and idempotency. Its default location is `~/.local/state/matomeru/state.sqlite3`, configurable with `MATOMERU_STATE_DB`; it is not a service or container.
- Empty-result runs still produce an email and note. Item and delivery failures are logged; Matomeru continues where possible and exits non-zero if the run is incomplete.

## Documentation

- [Implementation plan and decisions](PLAN.md)
- [Reddit rate-limit research](docs/research/reddit-rate-limits.md)

## Development

The package currently has no application dependencies or working digest commands. See [PLAN.md](PLAN.md) for the implementation sequence, configuration boundaries, operational behavior, and remaining deployment values.
