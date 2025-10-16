from __future__ import annotations

import contextlib

import pytest

from services.cache import quotes


class DummyClient:
    get_quotes_bulk = None


@pytest.fixture(autouse=True)
def reset_adaptive_controller(monkeypatch):
    controller = quotes.AdaptiveBatchController(
        default_size=6,
        min_size=4,
        max_size=10,
    )
    monkeypatch.setattr(quotes, "_ADAPTIVE_BATCH_CONTROLLER", controller)
    return controller


def test_adaptive_batch_controller_reacts_to_latency(reset_adaptive_controller):
    controller = reset_adaptive_controller

    assert controller.current(20) == 6

    next_size = controller.observe(900.0, 20)
    assert next_size == 5
    assert controller.current(20) == 5

    next_size = controller.observe(350.0, 20)
    assert next_size == 9
    assert controller.current(20) == 9


def test_fetch_quotes_bulk_records_adaptive_metrics(monkeypatch, reset_adaptive_controller):
    recorded_extras: list[dict | None] = []

    @contextlib.contextmanager
    def fake_timer(label: str, *, extra=None):
        recorded_extras.append(extra)
        yield

    monkeypatch.setattr(quotes, "performance_timer", fake_timer)
    monkeypatch.setattr(quotes, "record_quote_load", lambda *args, **kwargs: None)
    monkeypatch.setattr(quotes, "record_quote_provider_usage", lambda *args, **kwargs: None)

    def fake_get_quote_cached(_cli, market, symbol, panel, ttl, stats):
        stats.record_payload({"provider": "cache", "last": 1})
        return {"provider": "cache", "last": 1}

    monkeypatch.setattr(quotes, "_get_quote_cached", fake_get_quote_cached)

    items = [("bcba", f"SYM{i}", None) for i in range(10)]

    result = quotes.fetch_quotes_bulk(DummyClient(), items)

    assert isinstance(result, dict)
    assert recorded_extras, "performance telemetry should be captured"
    telemetry = recorded_extras[-1]
    assert telemetry is not None
    assert telemetry.get("adaptive_batch_size") == 6
    assert telemetry.get("avg_batch_time_ms") is not None
    assert telemetry.get("next_adaptive_batch_size") >= 8
