# Matomeru 📰

**Matomeru (まとめる, “to bring together / organize”) is a cron-driven, self-hostable digest generator for Reddit and arXiv.** Every day it collects the posts and papers from a rolling 24-hour window, summarizes them against your personal Markdown interest profile using a local LLM, emails the result, and files one dated note per source into your notes vault. It is not a service: two standalone CLI commands, no Docker, built to be invoked by cron.

AI tools were used during development.

## 🙋 Why Matomeru?

Feed readers show you everything; Matomeru reads for you. The interest profile and the summarization prompts are yours, the LLM runs on your own hardware, and the digest lands in the two places you actually read — your inbox and your notes vault. Reddit collection goes through [Sagashi](https://github.com/divin/sagashi), a self-hosted MCP server, and notes go through your own Notes MCP server, so no third-party service sees your reading list.

## 🧭 How it works

```mermaid
flowchart LR
    cron[Cron] --> R[matomeru-reddit]
    cron --> A[matomeru-arxiv]
    R -->|top posts + threads| S[Sagashi MCP]
    A -->|new papers| X[arXiv RSS feeds]
    S --> L[Local LLM via pydantic-ai]
    X --> L
    P[Interest profile<br/>matomeru.toml sections] -. embedded in every prompt .-> L
    L --> T[Deterministic digest templates]
    T --> M[SMTP email]
    T --> N[Notes MCP dated note]
    T --> DB[(SQLite state)]
```

Two sources, two commands, each with its own digest, email, and notes folder:

- `uv run matomeru-reddit` collects the configured top posts from selected subreddits and fetches their threads and top comments through the Sagashi MCP server. When a reranker endpoint is configured (`RERANKER_URL`), every collected thread is scored against each profile section in one batched request and only the top `reddit_top_x` threads per subreddit are summarized; each kept post is summarized with its discussion, explaining its relevance to the profile. Without the reranker, all collected threads are summarized.
- `uv run matomeru-arxiv` queries the per-category arXiv RSS announcement feeds for new papers in configured categories and abstract search terms. When a reranker is configured, LLM screening is replaced by deterministic threshold selection — every paper scoring at least `arxiv_min_score` is summarized, and no screening LLM calls are made. Without the reranker, each abstract is screened against the profile by the LLM and only the relevant papers are summarized.

**The determinism principle.** Configuration owns everything structural: paths, folders, recipients, window boundaries, and the digest layout — templates live in code, not in the model. The LLM only fills text slots via structured output: `summary` and `why_relevant` for each Reddit thread, `summary` and `why_relevant` for each included arXiv paper, and the `include`/`reason` screening decision for arXiv papers. The agent registers no tools and never receives authority to act; prompt content is embedded purely as data with fixed truncation caps.

With the reranker configured, this principle extends from structure to **selection**: which threads and papers enter the digest is decided by code from reranker scores (top-X per subreddit for Reddit, score threshold for arXiv), not by the model. The LLM keeps its generation-only role of slot-filling the summaries of the selected items.

**Conservative exclusion.** arXiv screening prompts instruct the model to decide `include=false` when uncertain, and a screening call that *fails* (model or transport error) is also treated as excluded — a paper is left out rather than guessed into the digest. A Reddit thread whose summarization fails is likewise excluded and recorded in the digest's Errors section.

## ⚡ Performance

The first real run took **~6 h for Reddit**: 110 threads summarized sequentially at ~3.3 min per LLM call on the deployment's local model, and arXiv screening adds up to ~100 more LLM calls per run. A cross-encoder **reranker sidecar** ([llama-server](https://github.com/ggml-org/llama.cpp) with `--reranking --pooling rank`, e.g. running `bge-reranker-v2-m3` via `llama-server -m bge-reranker-v2-m3-Q8_0.gguf --reranking --pooling rank --port <port> --ubatch-size 4096`) scores all collected candidates in one cheap request, so only the selected subset ever reaches the LLM:

- **Reddit:** ~110 thread summarizations shrink to the top `reddit_top_x` per subreddit (default 3).
- **arXiv:** the LLM screening stage drops to zero calls; selection is a deterministic score threshold.

Measured before/after numbers will be added here after the first live run with the sidecar. The model choice and evaluation are documented in [reranker-models.md](docs/research/reranker-models.md); the sidecar is configured via `RERANKER_URL` (see the [configuration reference](#%EF%B8%8F-configuration-reference)).

## 🚀 Quick start

Prerequisites: [uv](https://docs.astral.sh/uv/) (provides Python 3.12), a running **Sagashi MCP** server (≥ 0.3.1, streamable HTTP), a running **Notes MCP** server (streamable HTTP) exposing `create_folder` / `create_note`, a **local OpenAI-compatible LLM endpoint** with structured-output support, and an **SMTP account**.

1. Clone and install:

   ```sh
   git clone <repository-url>
   cd matomeru
   uv sync
   ```

2. Copy `example.matomeru.toml` to `matomeru.toml` and fill in your values, including the `[[profile.sections]]` tables — your interest profile as `name`/`text` sections (at least one; embedded into every prompt and used as the reranker's per-section scoring queries). The `[[profile.sections]]` and `[reddit] subreddits` fields are required; see the [configuration reference](#%EF%B8%8F-configuration-reference). The file is looked up at the explicit `--config` path, the `MATOMERU_CONFIG` environment variable, or `matomeru.toml` in the current working directory, in that order.
3. Copy `example.env` to `.env` and fill in the TODO values. All variables (required and optional) are listed there; the required ones are enforced at startup.
4. Verify with a dry run — runs preflight checks (SMTP credentials, LLM/MCP configuration, state database location) and prints the computed window plan, without touching state or the network. Exits 1 if any check fails:

   ```sh
   uv run matomeru-reddit --dry-run
   ```

   Add `--live` to also probe network reachability (TCP connect to the SMTP server, HTTP probes of the Sagashi/Notes/LLM endpoints). `--live` requires `--dry-run`:

   ```sh
   uv run matomeru-reddit --dry-run --live
   ```

   Do the same for `matomeru-arxiv` if you use it.

5. First real run:

   ```sh
   uv run matomeru-reddit
   uv run matomeru-arxiv
   ```

   Each run writes its note, sends its email, and records the run in the state database. Exit code 0 means the digest was delivered (or the run was skipped because it already completed).

Notes folders (`notes.reddit_folder` / `notes.arxiv_folder`) are created automatically on first use (`ensure_folder` is idempotent), so no manual setup is needed — though an empty folder value disables notes delivery for that source.

## ⚙️ Configuration reference

### `.env` variables

All variables are loaded from `.env` in the working directory (via `python-dotenv`) or the environment. Missing *required* variables raise a `ConfigError` and the run exits 1.

| Variable | Required | Default | Meaning |
| --- | --- | --- | --- |
| `SAGASHI_MCP_URL` | required | — | Base URL of the Sagashi MCP server (streamable HTTP). |
| `SAGASHI_MCP_TOKEN` | optional | none | Bearer token sent as `Authorization: Bearer …` to Sagashi. |
| `NOTES_MCP_URL` | required | — | Base URL of the Notes MCP server (streamable HTTP). |
| `NOTES_MCP_TOKEN` | optional | none | Bearer token for the Notes MCP server. |
| `LLM_BASE_URL` | required | — | Base URL of the OpenAI-compatible endpoint (chat completions with structured output). |
| `LLM_API_KEY` | optional | `"local"` | API key for the endpoint; local endpoints typically ignore it, but it must be non-empty for the OpenAI client, so Matomeru defaults to `local`. |
| `LLM_MODEL` | required | — | Model name passed to the endpoint. |
| `SMTP_HOST` | required | — | SMTP server hostname. |
| `SMTP_PORT` | required | — | SMTP port (integer, 1–65535). Port `465` uses implicit TLS; any other port uses STARTTLS when the server offers it, plain otherwise. |
| `SMTP_USER` | optional | empty | SMTP username; login only happens when a username is set. |
| `SMTP_PASSWORD` | optional | empty | SMTP password (never logged). |
| `SMTP_FROM` | required | — | Sender address (`From` header and envelope sender). |
| `SMTP_TO` | required | — | Comma-separated recipient addresses. |
| `TZ` | required | — | IANA timezone name (e.g. `Europe/Berlin`) used for the report date and local display. |
| `RERANKER_URL` | optional | unset | Full URL of the reranker sidecar's `/v1/rerank` endpoint. **Off-switch:** when unset, the reranker funnel is disabled and behavior is identical to v1 (all Reddit threads summarized; arXiv uses LLM screening). |
| `RERANKER_TOKEN` | optional | none | Bearer token sent as `Authorization: Bearer …` to the reranker. |
| `MATOMERU_STATE_DB` | optional | `~/.local/state/matomeru/state.sqlite3` | Path to the SQLite state database. |
| `MATOMERU_LOG_LEVEL` | optional | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`); invalid values fall back to `INFO` with a warning. |

### `matomeru.toml`

| Section | Key | Required | Default | Meaning |
| --- | --- | --- | --- | --- |
| `[[profile.sections]]` | `name` | required | — | Name of one interest-profile section (non-empty, unique across sections). |
| `[[profile.sections]]` | `text` | required | — | Section body; at least one section is required. Sections are rendered into the LLM prompts and each serves as a reranker scoring query. |
| `[reddit]` | `subreddits` | required | — | List of subreddits to monitor (at least one). |
| `[reddit]` | `posts_per_subreddit` | optional | `3` | Number of top posts collected per subreddit. |
| `[reddit]` | `comments_per_post` | optional | `20` | Number of top comments fetched per post. |
| `[arxiv]` | `categories` | optional | `[]` | arXiv categories such as `"cs.AI"`; at least one category or search term must be set. |
| `[arxiv]` | `search_terms` | optional | `[]` | Abstract search terms (quoted phrases); fallback selector for categories without a `category_terms` entry; at least one category or search term must be set. |
| `[arxiv]` | `category_terms` | optional | `{}` | Per-category search-term overrides (`category = ["term", ...]`); an explicitly empty list keeps all papers of that category; keys without a matching category are inert. |
| `[arxiv]` | `max_results` | optional | `100` | Maximum results requested from the arXiv API per run (positive integer). |
| `[notes]` | `reddit_folder` | optional | empty | Vault folder for Reddit notes; empty disables notes delivery for Reddit. |
| `[notes]` | `arxiv_folder` | optional | empty | Vault folder for arXiv notes; empty disables notes delivery for arXiv. |
| `[reranker]` | `reddit_top_x` | optional | `3` | Threads kept per subreddit for summarization when the reranker is configured (capped by what was collected). |
| `[reranker]` | `arxiv_min_score` | optional | `-6.0` | Minimum reranker score (raw logit space) for an arXiv paper to be included when the reranker is configured. |
| `[reranker]` | `min_score` | optional | off | Optional global minimum-score floor applied after top-X selection; leave unset to disable. |

Notes on the arXiv section:

- Both `categories` and `search_terms` empty is a **`ConfigError`** — retrieval needs at least one selector. The two combine as `(categories) AND (search terms)` when both are present.
- `max_results` must be a positive integer or the run exits 1 with a `ConfigError`.

Notes on the reranker section:

- The `[reranker]` settings apply only when `RERANKER_URL` is set; without it, the funnel is disabled and these values are ignored.
- A reranker request that fails (unreachable, error, malformed/incomplete response) fails the run (exit 1) — there is no silent fallback to full summarization.

## ⏰ Cron setup

Run each command from cron, logging output to files:

```cron
# m  h  dom mon dow  command
30 7  *   *   *   cd /opt/matomeru && uv run matomeru-reddit >> /var/log/matomeru/reddit.log 2>&1
45 7  *   *   *   cd /opt/matomeru && uv run matomeru-arxiv  >> /var/log/matomeru/arxiv.log 2>&1
```

Timezone notes:

- **cron uses the server's timezone** — setting `TZ` in `.env` labels the report date but does not change *when* cron starts the command. Align the server's cron timezone with `TZ` in `.env` (e.g. run both from a server whose timezone matches).
- The run window is a rolling 24 hours computed on UTC instants, independent of the run hour. Around a daylight-saving transition, two consecutive daily runs at the same local time are 23 or 25 elapsed hours apart, so the rolling window can briefly overlap the previous window or leave a one-hour gap. This is an accepted trade-off; items that straddle the boundary may appear twice or once.
- Each daily run is keyed by the local report date: if the previous run completed, it is skipped (see below). Use `--force` to reprocess.

## 🔁 Operations

### State database

- Location: `MATOMERU_STATE_DB`, default `~/.local/state/matomeru/state.sqlite3`. Not a service or container; it is created automatically with parent directories.
- Backup: stop nothing — the file is only touched during runs; copy it like any other SQLite file (`sqlite3 <path> ".backup backup.sqlite3"` or `cp` between runs).
- Reset: delete the SQLite file (and its `-wal`/`-shm` siblings if present). The next run treats every run as new — **completed runs are re-run**, which re-sends the email and overwrites the notes (notes are upserted by path, so no duplicates).

### Rerun semantics

- Each run is keyed by `(source, report_date)` where `report_date` is the local date in `TZ` when the run starts.
- A **completed** run is skipped on the next invocation (exit 0, nothing touched).
- `--force` restarts even a completed run: the saved 24-hour window is reprocessed with fresh LLM calls, the note is overwritten at the same path, and the email is **sent again**.
- Running/failed rows are always restartable with or without `--force`; state updates after the run starts are best-effort and never block delivery.
- **SMTP exactly-once caveat:** the email is sent before the `email_sent` delivery flag is recorded. If the process crashes between the server accepting the message and the state record, the rerun will send a duplicate. Delivery is therefore *at-least-once* for email.

### Exit codes

- `0` — the digest was delivered, the run was skipped (already completed), or it was a `--dry-run` (preflight passed).
- `1` — configuration error (`ConfigError`, e.g. missing required env vars, bad TOML, empty arXiv selectors, invalid timezone), an incomplete run (partial content, failed collection, or a failed destination), or a failed `--dry-run` preflight.

### Logging

- Output goes to **stderr** in a cron-friendly format (`timestamp LEVEL name message`); redirect `2>&1` into your cron log files.
- `MATOMERU_LOG_LEVEL` controls verbosity. Credentials and passwords are never logged.

### Partial runs

- The digest always renders — even an empty window produces a "no results" digest that is delivered normally and marked complete.
- Item-level failures (a Reddit thread or arXiv paper that could not be summarized) are excluded from the digest but listed in its **Errors** section, and the run exits 1 (it was still delivered — the exit code tells you content is incomplete).
- arXiv screening failures do *not* mark the run partial (the paper is conservatively excluded), but they appear in the Errors section.
- Destination failures are isolated: an email failure never prevents the note (and vice versa), but any failed destination makes the run exit 1 so the next cron invocation retries it.

## 🧪 Development

Use Python 3.12, [`uv`](https://docs.astral.sh/uv/), and [`just`](https://just.systems/man/en/):

```sh
just sync  # Install the locked dev environment
just       # Format, lint, and sort imports
just check # Type, format, and lint checks
just test  # Test suite
```

The test suite currently contains **321 tests** and passes.

## 📚 Documentation

- [Implementation plan and decisions](PLAN.md)
- [Configuration sample](example.matomeru.toml)
- [Environment variable template](example.env)
- [Reddit rate-limit research](docs/research/reddit-rate-limits.md)
- [Reranker model research](docs/research/reranker-models.md)
