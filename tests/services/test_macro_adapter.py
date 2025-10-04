"""Tests for :mod:`services.macro_adapter`."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

import services.health as health_service
from infrastructure.macro import MacroAPIError, MacroSeriesObservation
from services import base_adapter
from services.macro_adapter import MacroAdapter


class _DummySettings(SimpleNamespace):
    """Namespace helper providing sensible defaults for tests."""

    macro_api_provider: Iterable[str] = ("fred", "worldbank")
    fred_sector_series: Dict[str, str] = {"energy": "FRED_ENERGY"}
    world_bank_sector_series: Dict[str, str] = {"energy": "WB_ENERGY"}
    macro_sector_fallback: Dict[str, Dict[str, object]] = {}


def _build_adapter(
    *,
    settings: SimpleNamespace | None = None,
    client_factories: Dict[str, Callable[[], object]] | None = None,
    timer: Callable[[], float] | None = None,
    clock: Callable[[], float] | None = None,
) -> MacroAdapter:
    return MacroAdapter(
        settings=settings or _DummySettings(),
        client_factories=client_factories,
        timer=timer,
        clock=clock,
    )


def test_fetch_records_attempts_and_fallback_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    fallback_events: list[tuple[str, str, str, bool]] = []

    def _record(adapter: str, provider: str, status: str, fallback: bool) -> None:
        fallback_events.append((adapter, provider, status, fallback))

    monkeypatch.setattr(base_adapter, "record_adapter_fallback", _record)

    timer_values = iter([0.0, 0.1, 0.2, 0.35])
    clock_values = iter([10.0, 20.0])

    def _timer() -> float:
        return next(timer_values)

    def _clock() -> float:
        return next(clock_values)

    class _FailingFred:
        def get_latest_observations(self, request: Dict[str, str]) -> Dict[str, MacroSeriesObservation]:
            raise MacroAPIError("FRED indisponible")

    class _SuccessfulWorldBank:
        def get_latest_observations(self, request: Dict[str, str]) -> Dict[str, MacroSeriesObservation]:
            return {
                "Energy": MacroSeriesObservation(
                    series_id="WB_ENERGY",
                    value=3.21,
                    as_of="2024-01-01",
                )
            }

    adapter = _build_adapter(
        client_factories={
            "fred": _FailingFred,
            "worldbank": _SuccessfulWorldBank,
        },
        timer=_timer,
        clock=_clock,
    )

    result = adapter.fetch(["Energy"])

    assert result.provider == "worldbank"
    assert result.entries["Energy"]["value"] == pytest.approx(3.21)

    assert [attempt["status"] for attempt in result.attempts] == ["error", "success"]
    assert result.attempts[0]["provider"] == "fred"
    assert "indisponible" in result.attempts[0]["detail"]
    assert result.attempts[1]["provider"] == "worldbank"

    assert fallback_events == [
        ("MacroAdapter", "fred", "error", False),
        ("MacroAdapter", "worldbank", "success", True),
    ]


def test_fetch_uses_static_fallback_and_updates_health(monkeypatch: pytest.MonkeyPatch) -> None:
    fallback_events: list[tuple[str, str, str, bool]] = []

    def _record(adapter: str, provider: str, status: str, fallback: bool) -> None:
        fallback_events.append((adapter, provider, status, fallback))

    monkeypatch.setattr(base_adapter, "record_adapter_fallback", _record)
    monkeypatch.setattr(
        health_service,
        "st",
        SimpleNamespace(session_state={}),
    )

    time_values = iter([100.0, 200.0, 300.0, 400.0, 500.0])
    monkeypatch.setattr(health_service.time, "time", lambda: next(time_values))

    timer_values = iter([0.0, 0.05, 0.2, 0.45, 0.7, 0.9])
    clock_values = iter([1.0, 2.0, 3.0])

    def _timer() -> float:
        return next(timer_values)

    def _clock() -> float:
        return next(clock_values)

    class _FailingFred:
        def get_latest_observations(self, request: Dict[str, str]) -> Dict[str, MacroSeriesObservation]:
            raise MacroAPIError("FRED sin credenciales configuradas")

    class _FailingWorldBank:
        def get_latest_observations(self, request: Dict[str, str]) -> Dict[str, MacroSeriesObservation]:
            raise MacroAPIError("World Bank no disponible")

    settings = _DummySettings(
        macro_sector_fallback={
            "energy": {"value": 2.5, "as_of": "2023-12-01"},
        }
    )

    adapter = _build_adapter(
        settings=settings,
        client_factories={
            "fred": _FailingFred,
            "worldbank": _FailingWorldBank,
        },
        timer=_timer,
        clock=_clock,
    )

    result = adapter.fetch(["Energy"])

    assert result.provider is None
    assert result.fallback_entries == {
        "Energy": {"value": 2.5, "as_of": "2023-12-01"}
    }
    assert [attempt["provider"] for attempt in result.attempts] == [
        "fred",
        "worldbank",
        "fallback",
    ]
    assert result.attempts[-1]["status"] == "success"
    assert result.attempts[-1]["fallback"] is True

    health_service.record_macro_api_usage(
        attempts=result.attempts,
        notes=result.notes,
        metrics={"macro_source": "fallback"},
        latest=result.latest,
    )

    metrics = health_service.get_health_metrics()
    macro_metrics = metrics.get("macro_api") or {}
    providers = macro_metrics.get("providers") or {}

    fred_stats = providers.get("fred") or {}
    assert fred_stats.get("status_counts", {}).get("error") == 1

    worldbank_stats = providers.get("worldbank") or {}
    assert worldbank_stats.get("status_counts", {}).get("error") == 1

    fallback_stats = providers.get("fallback") or {}
    assert fallback_stats.get("status_counts", {}).get("success") == 1
    assert fallback_stats.get("fallback_count") == 1

    latest = macro_metrics.get("latest") or {}
    assert latest.get("provider") == "fallback"

    overall = macro_metrics.get("overall") or {}
    assert overall.get("fallback_count") == 1

    assert fallback_events == [
        ("MacroAdapter", "fred", "error", False),
        ("MacroAdapter", "worldbank", "error", True),
    ]
