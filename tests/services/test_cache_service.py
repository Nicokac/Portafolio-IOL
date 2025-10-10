from __future__ import annotations

import pytest

from services.cache import CacheService


class _FakeClock:
    def __init__(self) -> None:
        self._now = 1_000.0

    def monotonic(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += float(seconds)


def test_predictive_cache_expiration() -> None:
    clock = _FakeClock()
    cache = CacheService(namespace="test_predictive", monotonic=clock.monotonic, ttl_override=0.001)

    cache.set("alpha", 42)
    assert cache.get("alpha") == 42
    assert cache.hits == 1

    clock.advance(0.01)
    assert cache.get("alpha") is None
    assert cache.misses > 0


def test_get_effective_ttl_override() -> None:
    cache = CacheService(namespace="ttl_override", ttl_override=5.0)
    assert cache.get_effective_ttl() == pytest.approx(5.0)

    cache.set_ttl_override(10.0)
    assert cache.get_effective_ttl() == pytest.approx(10.0)
    cache.set("beta", "value", ttl=1.0)
    assert cache.get_effective_ttl() == pytest.approx(10.0)

    cache.set_ttl_override(None)
    assert cache.get_effective_ttl(3.0) == pytest.approx(3.0)
