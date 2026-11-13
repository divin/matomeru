"""Tests for window computation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from matomeru.config import ConfigError
from matomeru.window import RunWindow, compute_run_window


class TestComputeRunWindow:
    """Tests for compute_run_window."""

    def test_default_now_uses_utc_now(self) -> None:
        """When now is None, uses datetime.now(timezone.utc)."""
        import time

        before = datetime.now(timezone.utc)
        result = compute_run_window("UTC")
        time.sleep(0.01)  # Small delay to ensure after > before
        after = datetime.now(timezone.utc)

        assert isinstance(result, RunWindow)
        assert result.tz_name == "UTC"
        assert result.start_utc <= before <= result.end_utc <= after
        assert result.report_date == before.date()

    def test_end_utc_set_explicitly(self) -> None:
        """end_utc is the provided now value."""
        now = datetime(2025, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
        result = compute_run_window("UTC", now=now)

        assert result.end_utc == now
        assert result.start_utc == now - timedelta(hours=24)
        assert result.report_date == now.date()
        assert result.tz_name == "UTC"

    def test_europe_berlin_correct_timezone_conversion(self) -> None:
        """Report date is computed in the configured IANA timezone."""
        # June 15, 2025 at 02:30 UTC = June 15, 2025 at 04:30 CEST (UTC+2)
        now = datetime(2025, 6, 15, 2, 30, 0, tzinfo=timezone.utc)
        result = compute_run_window("Europe/Berlin", now=now)

        assert result.end_utc == now
        assert result.start_utc == now - timedelta(hours=24)
        # In Europe/Berlin at this time, it's still June 15
        assert result.report_date == datetime(2025, 6, 15).date()
        assert result.tz_name == "Europe/Berlin"

    def test_crossing_day_boundary(self) -> None:
        """Report date changes correctly when window crosses midnight."""
        # June 15, 2025 at 01:00 UTC = June 15, 2025 at 03:00 CEST
        # 24h earlier = June 14, 2025 at 01:00 UTC = June 13, 2025 at 03:00 CEST
        now = datetime(2025, 6, 15, 1, 0, 0, tzinfo=timezone.utc)
        result = compute_run_window("Europe/Berlin", now=now)

        assert result.end_utc == now
        assert result.start_utc == now - timedelta(hours=24)
        assert result.report_date == datetime(2025, 6, 15).date()

    def test_invalid_tz_raises_config_error(self) -> None:
        """Invalid timezone name raises ConfigError."""
        with pytest.raises(ConfigError) as exc_info:
            compute_run_window("Invalid/Timezone")

        assert "Unknown timezone" in str(exc_info.value)

    def test_dst_handling_via_utc_arithmetic(self) -> None:
        """DST is handled correctly by subtracting on UTC instants."""
        # March 10, 2024 at 05:00 UTC = March 10, 2024 at 00:00 EST (UTC-5)
        # March 10, 2024 is DST transition day in America/New_York (Spring forward at 02:00 local)
        # 24h earlier = March 09, 2024 at 05:00 UTC = March 09, 2024 at 00:00 EST
        now = datetime(2024, 3, 10, 5, 0, 0, tzinfo=timezone.utc)
        result = compute_run_window("America/New_York", now=now)

        assert result.end_utc == now
        assert result.start_utc == now - timedelta(hours=24)
        # End time in America/New_York is 00:00 EDT on March 10
        end_in_tz = result.end_utc.astimezone(timezone.utc)
        assert end_in_tz.date() == datetime(2024, 3, 10).date()

    def test_window_duration_is_exactly_24_hours(self) -> None:
        """The window is always exactly 24 hours regardless of DST."""
        now = datetime(2024, 3, 10, 12, 0, 0, tzinfo=timezone.utc)
        result = compute_run_window("America/New_York", now=now)

        delta = result.end_utc - result.start_utc
        assert delta == timedelta(hours=24)
