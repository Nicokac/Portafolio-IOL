from __future__ import annotations

import re
from datetime import timedelta

from zoneinfo import ZoneInfo

from shared.time_provider import TIMEZONE, TimeProvider


def test_time_provider_now_returns_formatted_string_and_timezone() -> None:
    timestamp = TimeProvider.now()

    assert re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", timestamp)

    moment = TimeProvider.now_datetime()
    assert isinstance(moment.tzinfo, ZoneInfo)
    assert moment.tzinfo.key == TIMEZONE
    assert moment.utcoffset() == timedelta(hours=-3)
