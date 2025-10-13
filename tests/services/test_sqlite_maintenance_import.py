"""Regression tests for SQLite maintenance lazy imports."""
from __future__ import annotations

import importlib
import sys
import types

import pandas as pd


def test_app_import_triggers_sqlite_maintenance(monkeypatch):
    """Importing the app should request maintenance start without ImportError."""

    import services.maintenance as maintenance

    called: dict[str, bool] = {}

    monkeypatch.setenv(
        "FASTAPI_TOKENS_KEY", "MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA="
    )
    monkeypatch.setenv(
        "IOL_TOKENS_KEY", "MTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTE="
    )

    def _fake_start() -> bool:
        called["invoked"] = True
        return True

    def _make_stub(name: str, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    def _missing_attr(_name: str):
        raise ModuleNotFoundError("fastapi stubbed for tests")

    fastapi_stub = _make_stub("fastapi")
    fastapi_stub.__getattr__ = _missing_attr  # type: ignore[attr-defined]
    fastapi_security_stub = _make_stub("fastapi.security")
    fastapi_security_stub.__getattr__ = _missing_attr  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "fastapi", fastapi_stub)
    monkeypatch.setitem(sys.modules, "fastapi.security", fastapi_security_stub)
    monkeypatch.setitem(sys.modules, "controllers", _make_stub("controllers", __path__=[]))
    monkeypatch.setitem(
        sys.modules,
        "controllers.portfolio",
        _make_stub("controllers.portfolio", __path__=[]),
    )
    monkeypatch.setitem(
        sys.modules,
        "controllers.portfolio.portfolio",
        _make_stub(
            "controllers.portfolio.portfolio",
            default_notifications_service_factory=lambda: None,
            default_view_model_service_factory=lambda: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "controllers.auth",
        _make_stub("controllers.auth", build_iol_client=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "ui.tabs.recommendations",
        _make_stub("ui.tabs.recommendations", render_recommendations_tab=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "ui.controllers.portfolio_ui",
        _make_stub("ui.controllers.portfolio_ui", render_portfolio_ui=lambda: None),
    )
    class _StubCacheService:
        def __init__(self, *args, **kwargs):
            self.ttl = None

        def set_ttl_override(self, ttl: float | None) -> None:
            self.ttl = ttl

        def get(self, *args, **kwargs):
            return None

        def set(self, *args, **kwargs):
            return None

    class _StubPredictiveState:
        def __init__(self, *args, **kwargs):
            pass

    cache_service_cls = _StubCacheService
    predictive_state_cls = _StubPredictiveState
    monkeypatch.setitem(
        sys.modules,
        "services.cache",
        _make_stub(
            "services.cache",
            __path__=[],
            get_fx_rates_cached=lambda: {},
            CacheService=cache_service_cls,
            PredictiveCacheState=predictive_state_cls,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "services.cache.core",
        _make_stub(
            "services.cache.core",
            CacheService=cache_service_cls,
            PredictiveCacheState=predictive_state_cls,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "services.cache.market_data_cache",
        _make_stub(
            "services.cache.market_data_cache",
            get_sqlite_backend_path=lambda: None,
            get_market_data_cache=lambda *args, **kwargs: _make_stub(
                "_market_cache",
                prediction_cache=_make_stub(
                    "_prediction_cache",
                    set_ttl_override=lambda *a, **k: None,
                ),
            ),
            run_persistent_cache_maintenance=lambda **kwargs: {},
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "services.health",
        _make_stub(
            "services.health",
            get_health_metrics=lambda: {},
            record_dependency_status=lambda *args, **kwargs: None,
            record_yfinance_usage=lambda *args, **kwargs: None,
            record_adapter_fallback=lambda *args, **kwargs: None,
            record_market_data_incident=lambda *args, **kwargs: None,
            record_environment_snapshot=lambda *args, **kwargs: None,
            record_diagnostics_snapshot=lambda *args, **kwargs: None,
        ),
    )

    monkeypatch.setattr(maintenance, "ensure_sqlite_maintenance_started", _fake_start)

    sys.modules.pop("app", None)
    monkeypatch.setattr(
        "pandas.read_csv",
        lambda *args, **kwargs: pd.DataFrame([{"symbol": "TEST"}]),
    )

    importlib.import_module("app")

    assert called.get("invoked") is True


def test_ensure_sqlite_maintenance_started_returns_bool(monkeypatch):
    """The maintenance starter returns True while avoiding real threads during tests."""

    sys.modules.pop("services.maintenance.sqlite_maintenance", None)
    sqlite_module = importlib.import_module("services.maintenance.sqlite_maintenance")

    class _DummyScheduler:
        def ensure_running(self) -> bool:
            return True

    monkeypatch.setattr(sqlite_module, "_get_scheduler", lambda: _DummyScheduler())

    import services.maintenance as maintenance

    assert maintenance.ensure_sqlite_maintenance_started() is True
