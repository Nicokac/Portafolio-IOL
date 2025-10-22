from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional

import pytest

from infrastructure.macro.fred_client import (
    FredClient,
    FredSeriesObservation,
    MacroAPIError,
    MacroAuthenticationError,
    MacroRateLimitError,
)


class DummyResponse:
    def __init__(self, status_code: int, payload: Optional[Dict[str, Any]] = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Dict[str, Any]:
        if self._payload is None:
            raise ValueError("no payload")
        return self._payload


class DummySession:
    def __init__(self, responses: List[DummyResponse]) -> None:
        self._responses = itertools.cycle(responses)
        self.calls: List[Dict[str, Any]] = []

    def get(self, url: str, params: Dict[str, Any]) -> DummyResponse:
        self.calls.append({"url": url, "params": params})
        return next(self._responses)


def test_client_adds_authentication_and_defaults() -> None:
    responses = [
        DummyResponse(
            200,
            {
                "observations": [
                    {"value": "1.5", "observation_date": "2023-07-01"},
                    {"value": "nan", "observation_date": "2023-06-01"},
                ]
            },
        )
    ]
    session = DummySession(responses)
    client = FredClient(
        "secret",
        session=session,  # type: ignore[arg-type]
        calls_per_minute=0,
    )
    observation = client.get_latest_observation("IPUSN")
    assert observation == FredSeriesObservation(series_id="IPUSN", value=1.5, as_of="2023-07-01")
    assert session.calls
    params = session.calls[0]["params"]
    assert params["api_key"] == "secret"
    assert params["file_type"] == "json"
    assert params["series_id"] == "IPUSN"


def test_client_handles_rate_limit_sleep() -> None:
    responses = [
        DummyResponse(200, {"observations": [{"value": "2", "observation_date": "2023-08-01"}]}),
        DummyResponse(200, {"observations": [{"value": "3", "observation_date": "2023-09-01"}]}),
    ]
    session = DummySession(responses)

    timeline = itertools.count(start=0, step=0.2)
    sleeps: List[float] = []

    def fake_monotonic() -> float:
        return next(timeline)

    def fake_sleep(duration: float) -> None:
        sleeps.append(duration)

    client = FredClient(
        "key",
        session=session,  # type: ignore[arg-type]
        calls_per_minute=120,
        monotonic=fake_monotonic,
        sleeper=fake_sleep,
    )

    client.get_latest_observation("SERIES1")
    client.get_latest_observation("SERIES2")

    assert sleeps  # second call should have required sleeping
    assert pytest.approx(sleeps[0], rel=1e-5) == max(0.0, 0.5 - 0.2)


def test_client_raises_for_auth_and_rate_limit_errors() -> None:
    auth_session = DummySession([DummyResponse(401, {"error_message": "Bad key"})])
    client_auth = FredClient(
        "bad",
        session=auth_session,  # type: ignore[arg-type]
        calls_per_minute=0,
    )
    with pytest.raises(MacroAuthenticationError):
        client_auth.get_latest_observation("S")

    rate_session = DummySession([DummyResponse(429, {"error_message": "Slow down"})])
    client_rate = FredClient(
        "slow",
        session=rate_session,  # type: ignore[arg-type]
        calls_per_minute=0,
    )
    with pytest.raises(MacroRateLimitError):
        client_rate.get_latest_observation("S")


def test_client_raises_for_invalid_payload() -> None:
    session = DummySession([DummyResponse(200, None)])
    client = FredClient(
        "key",
        session=session,  # type: ignore[arg-type]
        calls_per_minute=0,
    )
    with pytest.raises(MacroAPIError):
        client.get_latest_observation("S")
