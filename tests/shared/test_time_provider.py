from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.append(str(Path(__file__).resolve().parents[2]))

from shared.time_provider import TIME_FORMAT, TIMEZONE, TimeProvider


def test_time_provider_now_returns_formatted_string_and_timezone() -> None:
    timestamp = TimeProvider.now()
    assert isinstance(timestamp, str)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", timestamp)

    moment = TimeProvider.now_datetime()
    assert isinstance(moment.tzinfo, ZoneInfo)
    assert moment.tzinfo.key == TIMEZONE
    assert moment.utcoffset() == timedelta(hours=-3)

    parsed = datetime.strptime(timestamp, TIME_FORMAT).replace(tzinfo=moment.tzinfo)
    delta = abs((moment - parsed).total_seconds())
    assert delta < 2, "Timestamp string should be in sync with aware datetime"


def test_time_provider_now_matches_now_datetime_format() -> None:
    moment_before = TimeProvider.now_datetime()
    timestamp = TimeProvider.now()
    moment_after = TimeProvider.now_datetime()

    assert isinstance(moment_before.tzinfo, ZoneInfo)
    assert moment_before.tzinfo.key == TIMEZONE
    assert isinstance(moment_after.tzinfo, ZoneInfo)
    assert moment_after.tzinfo.key == TIMEZONE

    formatted_candidates = {
        moment_before.strftime(TIME_FORMAT),
        moment_after.strftime(TIME_FORMAT),
    }
    assert timestamp in formatted_candidates
