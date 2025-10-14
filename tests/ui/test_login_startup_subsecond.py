from __future__ import annotations

import types
from pathlib import Path
import builtins
import importlib
import sys

import pytest

@pytest.fixture
def main_app(monkeypatch: pytest.MonkeyPatch):
    original_import = builtins.__import__
    saved_modules: dict[str, object | None] = {}

    def install_stub(name: str, **attrs):
        saved_modules[name] = sys.modules.get(name)
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        sys.modules[name] = module

    class DummyContainer:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def markdown(self, *args, **kwargs):
            return None

        def write(self, *args, **kwargs):
            return None

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("fastapi") or name.startswith("application"):
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    install_stub(
        "streamlit",
        session_state={},
        stop=lambda: None,
        container=lambda *a, **k: DummyContainer(),
        columns=lambda *a, **k: (DummyContainer(), DummyContainer()),
        empty=lambda: DummyContainer(),
        tabs=lambda labels: [DummyContainer() for _ in labels],
        markdown=lambda *a, **k: None,
        rerun=lambda: None,
        toast=lambda *a, **k: None,
        experimental_rerun=lambda: None,
        write=lambda *a, **k: None,
        sidebar=types.SimpleNamespace(expander=lambda *a, **k: DummyContainer()),
        caption=lambda *a, **k: None,
        warning=lambda *a, **k: None,
    )
    components_stub = types.SimpleNamespace(v1=types.SimpleNamespace(html=lambda *a, **k: None))
    sys.modules["streamlit"].components = components_stub
    install_stub("streamlit.components", __path__=[])
    sys.modules["streamlit.components"].v1 = components_stub.v1
    install_stub("streamlit.components.v1", html=lambda *a, **k: None)
    install_stub("ui.ui_settings", init_ui=lambda *a, **k: None, render_ui_controls=lambda *a, **k: None)
    install_stub("ui.header", render_header=lambda *a, **k: None)
    install_stub("ui.actions", render_action_menu=lambda *a, **k: None)
    install_stub(
        "ui.health_sidebar",
        render_health_monitor_tab=lambda *a, **k: None,
        summarize_health_status=lambda metrics: ("✅", "OK", "", "ok", None),
    )
    install_stub("ui.login", render_login_page=lambda *a, **k: None)
    install_stub("ui.footer", render_footer=lambda *a, **k: None)
    install_stub("ui.helpers.preload", ensure_scientific_preload_ready=lambda *a, **k: True)
    install_stub("ui.controllers", __path__=[])
    install_stub("ui.controllers.portfolio_ui", render_portfolio_ui=lambda *a, **k: None)
    install_stub("ui.tabs", __path__=[])
    install_stub("ui.tabs.recommendations", render_recommendations_tab=lambda *a, **k: None)
    install_stub("services.cache", get_fx_rates_cached=lambda: ({}, None))
    install_stub("controllers", __path__=[])
    install_stub("controllers.portfolio", __path__=[])
    install_stub(
        "controllers.portfolio.portfolio",
        default_view_model_service_factory=lambda *a, **k: None,
        default_notifications_service_factory=lambda *a, **k: None,
    )
    install_stub("controllers.auth", build_iol_client=lambda: None)
    install_stub("services.health", get_health_metrics=lambda: {}, record_dependency_status=lambda *a, **k: None)
    settings_stub = types.SimpleNamespace(
        cache_ttl_portfolio=3600,
        cache_ttl_last_price=3600,
        cache_ttl_fx=3600,
        cache_ttl_quotes=3600,
        cache_ttl_yf_indicators=3600,
        cache_ttl_yf_history=3600,
        cache_ttl_yf_fundamentals=3600,
        cache_ttl_yf_portfolio_fundamentals=3600,
        quotes_hist_maxlen=1,
        max_quote_workers=1,
        min_score_threshold=1,
        max_results=10,
        app_env="test",
        ENABLE_PROMETHEUS=False,
        PERFORMANCE_STORE_TTL_DAYS=7,
        SQLITE_MAINTENANCE_INTERVAL_HOURS=24,
        SQLITE_MAINTENANCE_SIZE_THRESHOLD_MB=512,
    )
    install_stub(
        "shared.config",
        configure_logging=lambda *a, **k: None,
        ensure_tokens_key=lambda: None,
        get_config=lambda *a, **k: {},
        settings=settings_stub,
    )
    install_stub(
        "shared.security_env_validator",
        validate_security_environment=lambda: None,
        SecurityValidationError=type("SecurityValidationError", (Exception,), {}),
    )
    install_stub("services.notifications", build_notification_badges=lambda *a, **k: None)
    install_stub(
        "shared.settings",
        FEATURE_OPPORTUNITIES_TAB=False,
        enable_prometheus=False,
        performance_store_ttl_days=7,
        sqlite_maintenance_interval_hours=24,
        sqlite_maintenance_size_threshold_mb=512,
        settings=settings_stub,
    )
    install_stub(
        "shared.time_provider",
        TimeProvider=types.SimpleNamespace(
            now=lambda: "now",
            from_timestamp=lambda ts: types.SimpleNamespace(text="ts", moment=None),
        ),
    )

    module = importlib.import_module("app")
    yield module

    for name, original in saved_modules.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture(autouse=True)
def reset_session_state(main_app, monkeypatch: pytest.MonkeyPatch):  # noqa: D401 - fixture
    monkeypatch.setattr(main_app.st, "session_state", {}, raising=False)
    monkeypatch.setattr(main_app.st, "stop", lambda: None, raising=False)
    yield
    main_app.st.session_state.clear()


def test_render_login_phase_marks_preload_pending(
    main_app, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorded: dict[str, object] = {}

    def fake_start_preload_worker(*, paused: bool, libraries=None) -> bool:  # type: ignore[override]
        recorded["paused"] = paused
        recorded["libraries"] = tuple(libraries) if libraries else None
        return True

    monkeypatch.setattr(main_app, "start_preload_worker", fake_start_preload_worker)
    monkeypatch.setattr(main_app, "render_login_page", lambda: None)

    base_time = main_app._TOTAL_LOAD_START
    monkeypatch.setattr(main_app.time, "perf_counter", lambda: base_time + 0.0008)

    main_app._render_login_phase()

    assert recorded["paused"] is True
    assert main_app.st.session_state["ui_startup_load_ms"] == pytest.approx(0.8)
    assert main_app.st.session_state["scientific_preload_ready"] is False


def test_main_schedules_preload_resume_after_auth(
    main_app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resume_calls: list[dict[str, object]] = []

    def fake_resume_preload_worker(*, delay_seconds: float, libraries=None) -> bool:  # type: ignore[override]
        resume_calls.append({"delay_seconds": delay_seconds, "libraries": libraries})
        return True

    monkeypatch.setattr(main_app, "resume_preload_worker", fake_resume_preload_worker)
    monkeypatch.setattr(main_app, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(main_app, "ensure_tokens_key", lambda: None)
    monkeypatch.setattr(main_app, "_check_critical_dependencies", lambda: None)
    monkeypatch.setattr(main_app, "get_fx_rates_cached", lambda: ({}, None))
    monkeypatch.setattr(main_app, "render_header", lambda *a, **k: None)
    monkeypatch.setattr(main_app, "render_action_menu", lambda *a, **k: None)
    monkeypatch.setattr(main_app, "render_footer", lambda *a, **k: None)
    monkeypatch.setattr(main_app, "build_iol_client", lambda: None)
    monkeypatch.setattr(main_app, "get_health_metrics", lambda: {})
    monkeypatch.setattr(
        main_app,
        "summarize_health_status",
        lambda metrics: ("✅", "OK", "", "ok", None),
    )
    monkeypatch.setattr(main_app, "render_health_monitor_tab", lambda *a, **k: None)
    monkeypatch.setattr(main_app, "_lazy_attr", lambda *a, **k: lambda *args, **kwargs: None)
    monkeypatch.setattr(main_app, "FEATURE_OPPORTUNITIES_TAB", False, raising=False)

    ensure_calls: list[object] = []
    monkeypatch.setattr(
        main_app,
        "ensure_scientific_preload_ready",
        lambda container: ensure_calls.append(container) or True,
    )

    main_app.st.session_state.update({"authenticated": True})

    class DummyContainer:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def markdown(self, *args, **kwargs):
            return None

    monkeypatch.setattr(main_app.st, "container", lambda: DummyContainer())
    monkeypatch.setattr(
        main_app.st,
        "columns",
        lambda *args, **kwargs: (DummyContainer(), DummyContainer()),
    )
    monkeypatch.setattr(main_app.st, "tabs", lambda labels: [DummyContainer() for _ in labels])
    monkeypatch.setattr(main_app.st, "empty", lambda: DummyContainer())
    monkeypatch.setattr(main_app.st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(
        main_app.st,
        "sidebar",
        types.SimpleNamespace(expander=lambda *a, **k: DummyContainer()),
    )
    monkeypatch.setattr(main_app.st, "rerun", lambda: None, raising=False)
    monkeypatch.setattr(main_app.st, "toast", lambda *a, **k: None, raising=False)

    dummy_time = types.SimpleNamespace(now=lambda: "now", from_timestamp=lambda ts: types.SimpleNamespace(text="ts"))
    monkeypatch.setattr(main_app, "TimeProvider", dummy_time)
    monkeypatch.setattr(main_app, "ANALYSIS_LOG_PATH", tmp_path / "analysis.log")
    monkeypatch.setattr(main_app, "log_ui_total_load_metric", lambda *_a, **_k: None)
    monkeypatch.setattr(main_app, "log_startup_event", lambda *_a, **_k: None)

    main_app.main([])

    assert resume_calls
    assert resume_calls[0]["delay_seconds"] == pytest.approx(0.5, rel=0.1)
    assert ensure_calls  # preload gate invoked post-login


def test_app_import_guard_handles_preload_failure(
    main_app, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    importlib.invalidate_caches()
    original_app = main_app
    original_preload = sys.modules.get("services.preload_worker")
    sys.modules.pop("app", None)
    if "services.preload_worker" in sys.modules:
        sys.modules.pop("services.preload_worker")

    original_import_module = importlib.import_module

    def failing_import(name: str, package: str | None = None):
        if name == "services.preload_worker":
            raise ImportError("forced preload failure")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", failing_import)

    with caplog.at_level("WARNING"):
        guarded_app = importlib.import_module("app")

    try:
        assert getattr(guarded_app, "_PRELOAD_WORKER", None) is None
        assert guarded_app.start_preload_worker(paused=True) is False
        assert guarded_app.resume_preload_worker(delay_seconds=0.1) is False
        assert guarded_app.is_preload_complete() is False
        assert any(
            "Lazy preload skipped on startup" in record.message
            for record in caplog.records
        )
    finally:
        sys.modules["app"] = original_app
        if original_preload is None:
            sys.modules.pop("services.preload_worker", None)
        else:
            sys.modules["services.preload_worker"] = original_preload
