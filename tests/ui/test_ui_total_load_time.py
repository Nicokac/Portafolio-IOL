
from __future__ import annotations

from datetime import datetime
import importlib
import sys
import importlib
import sys
from types import ModuleType, SimpleNamespace

_VALID_FASTAPI_KEY = "MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA="
_VALID_IOL_KEY = "MTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTE="


class _DummyPlaceholder:
    def __init__(self, store: list[str]) -> None:
        self._store = store

    def markdown(self, value: str, **kwargs) -> None:  # noqa: D401 - mimic Streamlit API
        self._store.append(value)


def _identity_decorator(*_args, **_kwargs):
    def _wrap(func):
        return func

    return _wrap


def _build_streamlit_stub() -> ModuleType:
    streamlit_module = ModuleType("streamlit")
    streamlit_module.session_state = {}
    streamlit_module.stop = lambda: None
    streamlit_module.container = lambda *a, **k: SimpleNamespace(__enter__=lambda self: self, __exit__=lambda *b: False)
    streamlit_module.columns = lambda *a, **k: ()
    streamlit_module.markdown = lambda *a, **k: None
    streamlit_module.empty = lambda: _DummyPlaceholder([])
    streamlit_module.cache_data = _identity_decorator
    streamlit_module.cache_resource = _identity_decorator

    components_module = ModuleType("streamlit.components")
    v1_module = ModuleType("streamlit.components.v1")
    v1_module.html = lambda *a, **k: None
    components_module.v1 = v1_module
    streamlit_module.components = components_module

    sys.modules['streamlit'] = streamlit_module
    sys.modules['streamlit.components'] = components_module
    sys.modules['streamlit.components.v1'] = v1_module
    return streamlit_module


def _build_maintenance_stub() -> None:
    maintenance_module = ModuleType("services.maintenance")
    maintenance_module.SQLiteMaintenanceConfiguration = lambda **kwargs: SimpleNamespace(**kwargs)
    maintenance_module.configure_sqlite_maintenance = lambda *a, **k: None
    maintenance_module.ensure_sqlite_maintenance_started = lambda: None
    sys.modules['services.maintenance'] = maintenance_module


def _build_diagnostics_stub() -> None:
    diagnostics_module = ModuleType("services.system_diagnostics")
    diagnostics_module.SystemDiagnosticsConfiguration = lambda *a, **k: None
    diagnostics_module.configure_system_diagnostics = lambda *a, **k: None
    diagnostics_module.ensure_system_diagnostics_started = lambda: None
    diagnostics_module.get_system_diagnostics_snapshot = lambda: {}
    sys.modules['services.system_diagnostics'] = diagnostics_module


def _build_controllers_stub() -> None:
    controllers_module = ModuleType("controllers")
    controllers_module.__path__ = []  # type: ignore[attr-defined]

    portfolio_package = ModuleType("controllers.portfolio")
    portfolio_package.__path__ = []  # type: ignore[attr-defined]

    portfolio_module = ModuleType("controllers.portfolio.portfolio")
    portfolio_module.default_notifications_service_factory = lambda: None
    portfolio_module.default_view_model_service_factory = lambda: None
    portfolio_module.render_portfolio_section = lambda *a, **k: None

    auth_module = ModuleType("controllers.auth")
    auth_module.build_iol_client = lambda: None

    portfolio_package.portfolio = portfolio_module
    controllers_module.portfolio = portfolio_package
    controllers_module.auth = auth_module
    recommendations_module = ModuleType("controllers.recommendations_controller")
    controllers_module.recommendations_controller = recommendations_module

    sys.modules['controllers'] = controllers_module
    sys.modules['controllers.portfolio'] = portfolio_package
    sys.modules['controllers.portfolio.portfolio'] = portfolio_module
    sys.modules['controllers.auth'] = auth_module
    sys.modules['controllers.recommendations_controller'] = recommendations_module


def test_total_load_time_is_recorded_and_rendered(monkeypatch) -> None:
    placeholders: list[str] = []

    monkeypatch.setenv("FASTAPI_TOKENS_KEY", _VALID_FASTAPI_KEY)
    monkeypatch.setenv("IOL_TOKENS_KEY", _VALID_IOL_KEY)

    streamlit_stub = _build_streamlit_stub()
    _build_maintenance_stub()
    _build_diagnostics_stub()
    _build_controllers_stub()

    recommendations_stub = ModuleType('ui.tabs.recommendations')
    recommendations_stub.render_recommendations_tab = lambda *a, **k: None
    sys.modules['ui.tabs.recommendations'] = recommendations_stub

    sys.modules.pop('app', None)
    main_app = importlib.import_module('app')

    record_calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        main_app,
        'record_stage',
        lambda *args, **kwargs: record_calls.append((args, kwargs)),
    )

    main_app._TOTAL_LOAD_START = 0.0
    monkeypatch.setattr(main_app.time, 'perf_counter', lambda: 5.0)

    session_state = {}
    streamlit_stub.session_state = session_state

    placeholder = _DummyPlaceholder(placeholders)
    main_app._render_total_load_indicator(placeholder)

    assert session_state['total_load_ms'] == 5000
    assert any('🕒 Tiempo total de carga: 5 000 ms' in block for block in placeholders)
    assert record_calls == [(('ui_total_load',), {'total_ms': 5000, 'status': 'success'})]
