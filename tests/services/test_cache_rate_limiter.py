import sys
from pathlib import Path

import pytest

from services import cache as svc_cache

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_rate_limiter_waits_when_bucket_empty() -> None:
    now = 0.0
    sleep_calls: list[float] = []

    def monotonic() -> float:
        return now

    def sleeper(duration: float) -> None:
        nonlocal now
        sleep_calls.append(duration)
        now += duration

    limiter = svc_cache.RateLimiter(
        capacity=2,
        refill_rate=1.0,
        monotonic=monotonic,
        sleeper=sleeper,
    )

    limiter.acquire()
    limiter.acquire()

    assert sleep_calls == []

    limiter.acquire()
    assert sleep_calls == [1.0]
    assert now == pytest.approx(1.0)


def test_rate_limiter_handles_partial_refill() -> None:
    now = 0.0
    sleep_calls: list[float] = []

    def monotonic() -> float:
        return now

    def sleeper(duration: float) -> None:
        nonlocal now
        sleep_calls.append(duration)
        now += duration

    limiter = svc_cache.RateLimiter(
        capacity=1,
        refill_rate=2.0,
        monotonic=monotonic,
        sleeper=sleeper,
    )

    limiter.acquire()

    now += 0.25
    limiter.acquire()

    assert sleep_calls == [0.25]
    assert now == pytest.approx(0.5)


def test_rate_limiter_validates_arguments() -> None:
    with pytest.raises(ValueError):
        svc_cache.RateLimiter(capacity=0, refill_rate=1.0)

    with pytest.raises(ValueError):
        svc_cache.RateLimiter(capacity=1, refill_rate=0)
