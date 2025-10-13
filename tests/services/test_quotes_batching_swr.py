import time

from services.cache.core import CacheService
from services.cache.market_data_cache import StaleWhileRevalidateCache


def test_swr_returns_stale_then_refreshes_background() -> None:
    cache = CacheService()
    swr = StaleWhileRevalidateCache(cache, default_ttl=0.1, grace_ttl=0.5, max_workers=1)

    calls: list[int] = []

    def loader() -> dict[str, int]:
        value = len(calls)
        calls.append(value)
        return {"value": value}

    try:
        first = swr.get_or_refresh("key", loader)
        assert first.was_refreshed is True
        assert first.is_stale is False
        assert first.refresh_scheduled is False
        assert first.value == {"value": 0}

        time.sleep(0.12)

        stale = swr.get_or_refresh("key", loader)
        assert stale.was_refreshed is False
        assert stale.is_stale is True
        assert stale.refresh_scheduled is True
        assert stale.value == {"value": 0}

        swr.wait()
        assert len(calls) >= 2

        fresh = swr.get_or_refresh("key", loader)
        assert fresh.was_refreshed is False
        assert fresh.is_stale is False
        assert fresh.value == {"value": 1}
        assert len(calls) >= 2
    finally:
        swr.shutdown()


def test_swr_forces_refresh_after_grace_window() -> None:
    cache = CacheService()
    swr = StaleWhileRevalidateCache(cache, default_ttl=0.05, grace_ttl=0.05, max_workers=1)

    counter = {"value": 0}

    def loader() -> dict[str, int]:
        counter["value"] += 1
        return {"value": counter["value"]}

    try:
        initial = swr.get_or_refresh("asset", loader)
        assert initial.value == {"value": 1}
        assert counter["value"] == 1

        time.sleep(0.2)

        refreshed = swr.get_or_refresh("asset", loader)
        assert refreshed.was_refreshed is True
        assert refreshed.is_stale is False
        assert refreshed.refresh_scheduled is False
        assert counter["value"] == 2
        assert refreshed.value == {"value": 2}
    finally:
        swr.shutdown()
