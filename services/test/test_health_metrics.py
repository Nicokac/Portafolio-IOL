from __future__ import annotations

from types import SimpleNamespace

import pytest

import services.health as health_service


def _reset_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        health_service,
        "st",
        SimpleNamespace(session_state={}),
    )


def test_record_macro_api_usage_accumulates_stats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_state(monkeypatch)

    health_service.record_macro_api_usage(
        attempts=[{"provider": "FRED", "label": "FRED", "status": "success", "elapsed_ms": 100.0}],
        latest={"provider": "FRED", "label": "FRED", "status": "success", "elapsed_ms": 100.0, "ts": 1.0},
    )
    health_service.record_macro_api_usage(
        attempts=[{"provider": "FRED", "label": "FRED", "status": "error", "elapsed_ms": 900.0, "detail": "boom"}],
        latest={"provider": "FRED", "label": "FRED", "status": "error", "elapsed_ms": 900.0, "detail": "boom", "ts": 2.0},
    )
    health_service.record_macro_api_usage(
        attempts=[{"provider": "ECB", "label": "ECB", "status": "success", "elapsed_ms": 400.0, "fallback": True}],
        latest={"provider": "ECB", "label": "ECB", "status": "success", "elapsed_ms": 400.0, "fallback": True, "ts": 3.0},
    )

    metrics = health_service.get_health_metrics()
    macro = metrics.get("macro_api")
    assert macro
    assert macro.get("latest", {}).get("provider") == "ecb"

    providers = macro.get("providers") or {}
    fred = providers.get("fred") or {}
    assert fred.get("label") == "FRED"
    assert fred.get("count") == 2
    assert fred.get("status_counts", {}).get("success") == 1
    assert fred.get("status_counts", {}).get("error") == 1
    assert fred.get("error_count") == 1
    assert fred.get("error_ratio") == pytest.approx(0.5)

    latency_counts = fred.get("latency_buckets", {}).get("counts") or {}
    assert latency_counts.get("fast") == 1
    assert latency_counts.get("slow") == 1

    ecb = providers.get("ecb") or {}
    assert ecb.get("fallback_count") == 1
    assert ecb.get("fallback_ratio") == pytest.approx(1.0)

    overall = macro.get("overall") or {}
    assert overall.get("count") == 3
    assert overall.get("error_count") == 1
    assert overall.get("fallback_count") == 1
    overall_latency = overall.get("latency_buckets", {}).get("counts") or {}
    assert overall_latency.get("fast") == 1
    assert overall_latency.get("medium") == 1
    assert overall_latency.get("slow") == 1


def test_record_macro_api_usage_tracks_missing_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_state(monkeypatch)

    health_service.record_macro_api_usage(
        attempts=[{"provider": "FRED", "label": "FRED", "status": "success", "elapsed_ms": None}],
        latest={"provider": "FRED", "label": "FRED", "status": "success", "elapsed_ms": None, "ts": 4.0},
    )

    macro = health_service.get_health_metrics().get("macro_api") or {}
    providers = macro.get("providers") or {}
    fred = providers.get("fred") or {}
    latency_counts = fred.get("latency_buckets", {}).get("counts") or {}
    assert latency_counts.get("missing") == 1
