from __future__ import annotations

from datetime import datetime, timezone

import pytest

import application.predictive_service as predictive_service
from application.predictive_service import PredictiveCacheSnapshot
from services.cache.core import get_cache_stats as service_cache_stats


def test_predictive_cache_stats_match_service_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictive_service.reset_cache()
    state = predictive_service.PredictiveCacheState(
        hits=4,
        misses=2,
        last_updated="2024-06-10 10:00:00",
        ttl_hours=0.5,
    )
    monkeypatch.setattr(predictive_service, "_CACHE_STATE", state)

    cache = predictive_service._CACHE
    cache.clear()
    cache.set_ttl_override(1800.0)
    cache.set("alpha", {"value": 1}, ttl=1800.0)
    cache._hits = state.hits  # type: ignore[attr-defined]
    cache._misses = state.misses  # type: ignore[attr-defined]
    cache._last_updated = datetime(2024, 6, 10, 10, 0, tzinfo=timezone.utc)  # type: ignore[attr-defined]

    snapshot = predictive_service.get_cache_stats()
    assert isinstance(snapshot, PredictiveCacheSnapshot)
    adapter_stats = service_cache_stats()

    assert adapter_stats["namespace"] == snapshot.namespace
    assert adapter_stats["hits"] == snapshot.hits
    assert adapter_stats["misses"] == snapshot.misses
    assert adapter_stats["last_updated"] == snapshot.last_updated
    assert adapter_stats["hit_ratio"] == pytest.approx(snapshot.hit_ratio)
    assert adapter_stats["ttl_seconds"] == pytest.approx(snapshot.ttl_seconds or 0.0)
    if snapshot.remaining_ttl is not None and adapter_stats["remaining_ttl"] is not None:
        assert adapter_stats["remaining_ttl"] == pytest.approx(snapshot.remaining_ttl, rel=0.05)


def test_predictive_cache_snapshot_dict_contains_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictive_service.reset_cache()
    state = predictive_service.PredictiveCacheState(
        hits=1,
        misses=1,
        last_updated="2024-01-01 00:00:00",
        ttl_hours=1.0,
    )
    monkeypatch.setattr(predictive_service, "_CACHE_STATE", state)
    cache = predictive_service._CACHE
    cache.clear()
    cache.set_ttl_override(3600.0)
    cache.set("beta", {"value": 2}, ttl=3600.0)

    snapshot = predictive_service.get_cache_stats()
    payload = snapshot.to_dict()

    assert payload["ttl_hours"] == pytest.approx(1.0)
    assert payload["ttl_seconds"] == pytest.approx(3600.0)
    assert payload["remaining_ttl"] is None or payload["remaining_ttl"] >= 0
