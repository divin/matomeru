"""Reddit collection: list recent hot posts, dedupe, fetch threads in one batch.

This module contains collection logic only — no state writes, no LLM calls.
Orchestration (window selection, run lifecycle, digest rendering) stays in a
later phase.
"""

from __future__ import annotations

import logging

from matomeru.models import RedditPost, RedditThread, ThreadFetchError
from matomeru.sagashi_client import SagashiClient
from matomeru.window import RunWindow

logger = logging.getLogger(__name__)

LISTING_OVERFETCH = 20


async def collect_reddit(
    client: SagashiClient,
    subreddits: list[str],
    window: RunWindow,
    *,
    posts_per_subreddit: int = 3,
    comments_per_post: int = 20,
) -> tuple[list[RedditThread], list[ThreadFetchError]]:
    """Collect top posts across subreddits within the run window.

    For each subreddit the top listing is fetched with ``sort="top"``,
    ``time_filter="all"`` and an ``after`` bound at the window start (Sagashi's
    ``after`` is an exclusive created_utc bound in epoch seconds). Results are
    additionally filtered client-side to ``(start_utc, end_utc)`` exclusive,
    sorted by score descending and truncated to ``posts_per_subreddit``.

    Posts are deduplicated across subreddits by id (first occurrence wins;
    duplicate source subreddits are logged). Threads for the surviving ids are
    then fetched with exactly ONE batch ``fetch_reddit_threads`` call.

    Per-subreddit listing failures are logged at WARNING level and skipped —
    they do not abort the run.

    Args:
        client: Sagashi MCP client.
        subreddits: Subreddit names to collect from.
        window: The 24-hour run window.
        posts_per_subreddit: How many top posts to keep per subreddit.
        comments_per_post: Comment count requested per thread.

    Returns:
        Tuple of (fetched threads, per-thread fetch errors).
    """
    start_ts = window.start_utc.timestamp()
    end_ts = window.end_utc.timestamp()

    # Deduped post ids in selection order; map id -> chosen post for logging.
    selected: dict[str, tuple[RedditPost, str]] = {}
    listing_failures = 0

    for subreddit in subreddits:
        try:
            listing = await client.list_subreddit_posts(
                subreddit,
                sort="top",
                time_filter="all",
                limit=min(100, posts_per_subreddit + LISTING_OVERFETCH),
                after=str(start_ts),
            )
        except Exception as e:  # noqa: BLE001 - one broken subreddit must not kill the run
            logger.warning("listing failed for subreddit=%s: %s", subreddit, e)
            listing_failures += 1
            continue

        in_window = [p for p in listing.post_items if start_ts < p.created_utc < end_ts]
        in_window.sort(key=lambda p: p.score, reverse=True)
        top = in_window[:posts_per_subreddit]

        for post in top:
            if post.id in selected:
                logger.debug(
                    "duplicate post id=%s: keeping source subreddit=%s, dropping subreddit=%s",
                    post.id,
                    selected[post.id][1],
                    subreddit,
                )
                continue
            selected[post.id] = (post, subreddit)

        logger.info(
            "subreddit=%s listed=%d in_window=%d kept=%d total_selected=%d",
            subreddit,
            len(listing.post_items),
            len(in_window),
            len(top),
            len(selected),
        )

    if listing_failures == len(subreddits) and len(subreddits) > 0:
        # Total listing failure is a run failure, not a quiet day: raising
        # aborts the run (fail_run + exit 1) instead of emitting a misleading
        # "no results" digest.
        raise RuntimeError(
            f"all {len(subreddits)} subreddit listing(s) failed; Sagashi may be unreachable or misconfigured"
        )

    item_ids = list(selected)
    if not item_ids:
        logger.warning("no posts collected for window=[%s, %s]; skipping thread fetch", start_ts, end_ts)
        return [], []

    logger.info("fetching %d deduped threads with comment_count=%d", len(item_ids), comments_per_post)
    batch = await client.fetch_reddit_threads(item_ids, comment_count=comments_per_post)
    return batch.threads, batch.errors
