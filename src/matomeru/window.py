"""Time window computation for Matomeru pipeline runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from matomeru.config import ConfigError


@dataclass
class RunWindow:
    """A 24-hour run window with computed metadata."""

    start_utc: datetime
    end_utc: datetime
    report_date: date
    tz_name: str


def compute_run_window(tz_name: str, now: datetime | None = None) -> RunWindow:
    """Compute a 24-hour run window for the given timezone.

    Args:
        tz_name: IANA timezone name (e.g. 'Europe/Berlin').
        now: UTC datetime to use as end of window. Defaults to datetime.now(timezone.utc).

    Returns:
        RunWindow with start_utc, end_utc, report_date, and tz_name.

    Raises:
        ConfigError: If tz_name is invalid or unknown.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    try:
        tz = ZoneInfo(tz_name)
    except KeyError as e:
        raise ConfigError(f"Unknown timezone: {tz_name}") from e

    # end_utc is the current UTC time
    end_utc = now

    # start_utc is exactly 24 hours earlier (on UTC instants, handles DST correctly)
    start_utc = end_utc - timedelta(hours=24)

    # report_date is the date in the configured timezone at end_utc
    end_in_tz = end_utc.astimezone(tz)
    report_date = end_in_tz.date()

    return RunWindow(
        start_utc=start_utc,
        end_utc=end_utc,
        report_date=report_date,
        tz_name=tz_name,
    )
