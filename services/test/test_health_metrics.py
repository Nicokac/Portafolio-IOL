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


def test_record_yfinance_usage_tracks_history(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_state(monkeypatch)

    limit = health_service._PROVIDER_HISTORY_LIMIT
    base_events: list[tuple[str, str, bool, str]] = []
    providers = ["yfinance", "fallback", "error"]
    for index in range(limit + 2):
        provider = providers[index % len(providers)]
        status = "error" if provider == "error" else "success"
        fallback = provider == "fallback"
        base_events.append((provider, status, fallback, f"entry-{index}"))
    base_events.append(("error", "error", False, f"entry-{len(base_events)}"))

    timestamps = (float(index) for index in range(len(base_events)))
    monkeypatch.setattr(health_service.time, "time", lambda: next(timestamps))

    for provider, status, fallback, detail in base_events:
        health_service.record_yfinance_usage(
            provider,
            detail=detail,
            status=status,
            fallback=fallback,
        )

    metrics = health_service.get_health_metrics()
    yf_metrics = metrics.get("yfinance") or {}
    history = yf_metrics.get("history") or []

    assert yf_metrics.get("latest_provider") == "error"
    assert yf_metrics.get("latest_result") == "error"
    assert yf_metrics.get("detail") == base_events[-1][3]
    assert yf_metrics.get("fallback") is False
    assert len(history) == limit
    assert history[-1]["provider"] == "error"
    assert history[-1]["result"] == "error"
    assert history[-1]["detail"] == base_events[-1][3]
    assert any(entry.get("fallback") for entry in history)

    expected_tail = [event[0] for event in base_events][-limit:]
    assert [entry.get("provider") for entry in history] == expected_tail


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
    history = fred.get("history") or []
    assert history and history[-1]["status"] == "error"
    assert history[-1]["detail"] == "boom"

    latency_counts = fred.get("latency_buckets", {}).get("counts") or {}
    assert latency_counts.get("fast") == 1
    assert latency_counts.get("slow") == 1

    ecb = providers.get("ecb") or {}
    assert ecb.get("fallback_count") == 1
    assert ecb.get("fallback_ratio") == pytest.approx(1.0)
    ecb_history = ecb.get("history") or []
    assert any(entry.get("fallback") for entry in ecb_history)

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
