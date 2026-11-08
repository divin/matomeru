# Reddit API rate limits and AsyncPRAW behavior

**Checked:** 2026-11-08  
**Purpose:** Decide whether Matomeru or Sagashi needs a fixed Reddit API cooldown for the daily subreddit digest.

## Findings

1. Reddit's published API-facts page states a free Data API limit of **100 queries per minute per OAuth client ID** when using OAuth, and 10 queries per minute without OAuth. This article was published in 2023, so treat 100/minute as Reddit's published figure, not a permanent guarantee. [Reddit: Key Facts to Understanding Reddit's Recent API Updates](https://redditinc.com/news/apifacts)
2. Reddit's Data API Terms reserve the right to set and enforce API limits. The actual limit can therefore depend on Reddit's current policy and the response headers; do not hard-code the published figure as an assumption that every request may safely run at that rate. [Reddit Data API Terms, section 2.9](https://redditinc.com/policies/data-api-terms)
3. AsyncPRAW documents that it uses Reddit's rate-limit response headers and waits as appropriate. For rate-limit messages returned in API response data, it waits when the requested delay is within its configured `ratelimit_seconds` limit (default 5 seconds) and otherwise raises an exception. [AsyncPRAW: Rate limits](https://asyncpraw.readthedocs.io/en/stable/getting_started/ratelimits.html)
4. The Sagashi checkout inspected for this design pins `asyncpraw` 8.0.3 in `sagashi/uv.lock`; its Reddit client uses AsyncPRAW and currently contains no application-level fixed sleep. See [Sagashi Reddit implementation](../../../sagashi/src/sagashi/reddit/__init__.py) and [Sagashi dependency lock](../../../sagashi/uv.lock).

## Design decision

- Do **not** add an arbitrary fixed cooldown in Matomeru.
- Implement the general Sagashi `fetch_reddit_threads` batch operation sequentially within one AsyncPRAW session so the client can use response-header information as it traverses the batch. Keep the tool generic; Matomeru supplies its own selected post IDs and comment limit.
- Keep logging visible for rate-limit waits and errors. If real runs show the library's default maximum wait is too short for the service, tune AsyncPRAW's setting based on observed responses rather than add a blind per-request sleep.
- Reddit API usage by other Sagashi clients may share the same OAuth application's allowance. The daily job should not assume it owns the whole quota.

## Sources

- Reddit, “Key Facts to Understanding Reddit's Recent API Updates,” https://redditinc.com/news/apifacts
- Reddit Data API Terms, https://redditinc.com/policies/data-api-terms
- AsyncPRAW, “Rate limits,” https://asyncpraw.readthedocs.io/en/stable/getting_started/ratelimits.html
- Local Sagashi source snapshot: `sagashi/src/sagashi/reddit/__init__.py`, `sagashi/uv.lock`
