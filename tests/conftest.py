from __future__ import annotations

from collections.abc import Iterable, Sequence
import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
import warnings

import pytest

from tests.fixtures.common import DummyCtx
from tests.fixtures.streamlit import BaseFakeStreamlit, FakeStreamlit, UIFakeStreamlit
from tests.fixtures.time import FakeTime

warnings.filterwarnings(
    "ignore",
    message="`infrastructure.iol.legacy` está deprecado",
    category=DeprecationWarning,
)


@pytest.fixture
def dummy_ctx_fixture() -> DummyCtx:
    """Provide a reusable no-op context manager for tests."""

    return DummyCtx()


@pytest.fixture
def fake_time() -> FakeTime:
    """Provide a deterministic fake clock for time-sensitive tests."""

    return FakeTime()


class _DummyStreamlitSecretNotFoundError(KeyError):
    """Minimal replacement for streamlit.runtime.secrets error type."""


class _DummySecrets(dict):
    """Secrets mapping that keeps the stub core in sync."""

    def __init__(self, initial: dict[str, Any] | None = None, *, core: "_DummyStreamlitCore | None" = None) -> None:
        super().__init__(initial or {})
        if core is None:
            core = globals().get("_streamlit_core")
        self._core = core
        if self._core is not None:
            self._core._register_secrets(self)

    def __getitem__(self, key: str) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError:  # pragma: no cover - compatibility branch
            raise _DummyStreamlitSecretNotFoundError(str(key)) from None



class _DummySidebar:
    def __init__(self, core: "_DummyStreamlitCore | None" = None) -> None:
        self._core = core
        self.reset()

    def _context(self, entry: dict[str, Any]) -> _DummyContext:
        if self._core is None:
            return _DummyContext(_streamlit_core, entry, host=self)
        return _DummyContext(self._core, entry, host=self)

    def reset(self) -> None:
        self.headers: list[str] = []
        self.captions: list[str] = []
        self.markdowns: list[str] = []
        self.download_buttons: list[dict[str, Any]] = []
        self.elements: list[dict[str, Any]] = []

    def __enter__(self) -> "_DummySidebar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return False

    def header(self, text: object) -> None:
        value = str(text)
        self.headers.append(value)
        self.elements.append({"type": "header", "text": value})

    def caption(self, text: object) -> None:
        value = str(text)
        self.captions.append(value)
        self.elements.append({"type": "caption", "text": value})

    def markdown(self, text: object, **_: Any) -> None:
        value = str(text)
        self.markdowns.append(value)
        self.elements.append({"type": "markdown", "text": value})

    def container(self, *_, border: bool | None = None, **__) -> _DummyContext:
        entry = {"type": "container", "border": bool(border), "children": []}
        self.elements.append(entry)
        return self._context(entry)

    def line_chart(
        self,
        data: Any,
        *,
        height: int | None = None,
        width: object | None = None,
    ) -> None:
        record = {
            "type": "line_chart",
            "data": data,
            "height": height,
            "width": width,
        }
        self.elements.append(record)

    def area_chart(
        self,
        data: Any,
        *,
        height: int | None = None,
        width: object | None = None,
    ) -> None:
        record = {
            "type": "area_chart",
            "data": data,
            "height": height,
            "width": width,
        }
        self.elements.append(record)

    def download_button(
        self,
        label: str,
        data: Any,
        *,
        file_name: str | None = None,
        mime: str | None = None,
        key: str | None = None,
    ) -> None:
        record = {
            "type": "download_button",
            "label": str(label),
            "data": data,
            "file_name": file_name,
            "mime": mime,
            "key": key,
        }
        self.download_buttons.append(record)
        self.elements.append(record)


class _DummyContext:
    def __init__(
        self,
        core: "_DummyStreamlitCore",
        entry: dict[str, Any],
        *,
        host: "_DummySidebar | None" = None,
    ) -> None:
        self._core = core
        self._entry = entry
        self._host = host

    def __enter__(self) -> "_DummyContext":
        self._core._context_stack.append(self._entry.setdefault("children", []))
        if self._host is not None:
            self._core._host_stack.append(self._host)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._core._context_stack.pop()
        if self._host is not None and self._core._host_stack:
            self._core._host_stack.pop()
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._core, name)


class _DummyColumn:
    def __init__(self, core: "_DummyStreamlitCore", entry: dict[str, Any]) -> None:
        self._core = core
        self._entry = entry

    def __enter__(self) -> "_DummyColumn":
        self._core._context_stack.append(self._entry.setdefault("children", []))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._core._context_stack.pop()
        return False

    def metric(
        self,
        label: object,
        value: object,
        delta: object | None = None,
        *,
        help: object | None = None,
        **kwargs: Any,
    ) -> None:
        record = {
            "type": "metric",
            "label": str(label),
            "value": value,
            "delta": delta,
            "help": help,
        }
        if kwargs:
            record.update(kwargs)
        self._entry.setdefault("children", []).append(record)

    def caption(self, text: object) -> None:
        record = {"type": "caption", "text": str(text)}
        self._entry.setdefault("children", []).append(record)

    def markdown(self, text: object) -> None:
        record = {"type": "markdown", "text": str(text)}
        self._entry.setdefault("children", []).append(record)

    def write(self, text: object) -> None:
        record = {"type": "write", "text": text}
        self._entry.setdefault("children", []).append(record)


class _DummyTab(_DummyContext):
    def __init__(self, core: "_DummyStreamlitCore", entry: dict[str, Any]) -> None:
        super().__init__(core, entry)
        self.label = entry.get("label")


class _DummyPlaceholder:
    def __init__(self, core: "_DummyStreamlitCore", entry: dict[str, Any]) -> None:
        self._core = core
        self._entry = entry

    def empty(self) -> "_DummyPlaceholder":
        placeholder_entry = {"type": "placeholder", "children": []}
        self._entry.setdefault("children", []).append(placeholder_entry)
        return _DummyPlaceholder(self._core, placeholder_entry)

    def container(self) -> _DummyContext:
        container_entry = {"type": "container", "children": []}
        self._entry.setdefault("children", []).append(container_entry)
        return _DummyContext(self._core, container_entry)


class _DummyColumnConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _DummyStreamlitCore:
    def __init__(self) -> None:
        self.sidebar = _DummySidebar(core=self)
        self.session_state: dict[str, Any] = {}
        self._secrets_obj: _DummySecrets | None = None
        self.secrets = _DummySecrets(core=self)
        self._calls: list[dict[str, Any]] = []
        self._all_records: list[dict[str, Any]] = []
        self._context_stack: list[list[dict[str, Any]]] = [self._calls]
        self._host_stack: list[Any] = []
        self._button_returns: dict[str, bool] = {}
        self._checkbox_returns: dict[str, bool] = {}
        self._toggle_returns: dict[str, bool] = {}
        self._form_returns: dict[str, bool] = {}
        self.page_links: list[dict[str, Any]] = []

    # Public helpers -----------------------------------------------------
    def reset(self) -> None:
        self.sidebar._core = self
        self.sidebar.reset()
        self.session_state.clear()
        if isinstance(self.secrets, _DummySecrets):
            self.secrets.clear()
            self.secrets._core = self
        else:  # pragma: no cover - compatibility branch
            self.secrets = _DummySecrets(core=self)
        self._calls.clear()
        self._all_records.clear()
        self._context_stack = [self._calls]
        self._host_stack.clear()
        self._button_returns.clear()
        self._checkbox_returns.clear()
        self._toggle_returns.clear()
        self._form_returns.clear()
        self.page_links.clear()
        runtime = getattr(self, "runtime", None)
        if runtime is not None and hasattr(runtime, "_page_registry"):
            runtime._page_registry = {}

    # Internal wiring ----------------------------------------------------
    def _register_secrets(self, secrets: _DummySecrets) -> None:
        self._secrets_obj = secrets
        self.secrets = secrets
        module = globals().get("_streamlit_module")
        if module is not None:
            module.secrets = secrets

    def set_button_result(self, key: str, value: bool) -> None:
        self._button_returns[key] = bool(value)

    def set_checkbox_result(self, key: str, value: bool) -> None:
        self._checkbox_returns[key] = bool(value)

    def set_form_submit_result(self, key: str, value: bool) -> None:
        self._form_returns[key] = bool(value)

    def get_records(self, kind: str) -> list[dict[str, Any]]:
        return [entry for entry in self._all_records if entry.get("type") == kind]

    # Internal utilities -------------------------------------------------
    def _record(self, kind: str, **payload: Any) -> dict[str, Any]:
        entry = {"type": kind, **payload, "children": []}
        self._context_stack[-1].append(entry)
        self._all_records.append(entry)
        host = self._host_stack[-1] if self._host_stack else None
        if isinstance(host, _DummySidebar) and kind == "markdown":
            text = payload.get("text")
            if text is not None:
                value = str(text)
                host.markdowns.append(value)
                host.elements.append({"type": "markdown", "text": value})
        return entry

    # Streamlit API subset ----------------------------------------------
    def header(self, text: object) -> None:
        self._record("header", text=str(text))

    def subheader(self, text: object) -> None:
        self._record("subheader", text=str(text))

    def caption(self, text: object) -> None:
        self._record("caption", text=str(text))

    def markdown(self, text: object, *, unsafe_allow_html: bool = False) -> None:
        self._record("markdown", text=str(text), unsafe=bool(unsafe_allow_html))

    def plotly_chart(self, fig: object, **kwargs: Any) -> None:
        self._record("plotly_chart", fig=fig, kwargs=kwargs)

    def metric(
        self,
        label: object,
        value: object,
        delta: object | None = None,
        *,
        help: object | None = None,
        **kwargs: Any,
    ) -> None:
        self._record(
            "metric",
            label=str(label),
            value=value,
            delta=delta,
            help=help,
            kwargs=dict(kwargs),
        )

    def line_chart(
        self,
        data: Any,
        *,
        height: int | None = None,
        width: object | None = None,
    ) -> None:
        self._record(
            "line_chart",
            data=data,
            height=height,
            width=width,
        )

    def area_chart(
        self,
        data: Any,
        *,
        height: int | None = None,
        width: object | None = None,
    ) -> None:
        self._record(
            "area_chart",
            data=data,
            height=height,
            width=width,
        )

    def write(self, text: object) -> None:
        self._record("write", text=text)

    def info(self, text: object) -> None:
        self._record("info", text=str(text))

    def warning(self, text: object) -> None:
        self._record("warning", text=str(text))

    def error(self, text: object) -> None:
        self._record("error", text=str(text))

    def success(self, text: object) -> None:
        self._record("success", text=str(text))

    def toast(self, message: object) -> None:
        self._record("toast", message=str(message))

    def link_button(self, label: str, url: str) -> None:
        self._record("link_button", label=label, url=url)

    def page_link(self, page: str, *, label: str, icon: str | None = None) -> None:
        runtime = getattr(self, "runtime", None)
        registry = getattr(runtime, "_page_registry", None)
        if registry and page not in registry:
            raise KeyError("url_pathname")
        entry = self._record("page_link", page=page, label=label, icon=icon)
        self.page_links.append(entry)

    def stop(self) -> None:
        self._record("stop")
        raise RuntimeError("streamlit.stop called")

    def spinner(self, text: object) -> _DummyContext:
        entry = self._record("spinner", text=str(text))
        return _DummyContext(self, entry)

    def status(self, label: object, *, state: str | None = None) -> _DummyContext:
        entry = self._record("status", label=str(label), state=state)
        return _DummyContext(self, entry)

    def expander(self, label: object, *, expanded: bool | None = None) -> _DummyContext:
        entry = self._record("expander", label=str(label), expanded=bool(expanded) if expanded is not None else None)
        return _DummyContext(self, entry)

    def form(self, key: str) -> _DummyContext:
        entry = self._record("form", key=key)
        return _DummyContext(self, entry)

    def form_submit_button(self, label: str, *, key: str | None = None) -> bool:
        entry_key = key or label
        entry = self._record("form_submit_button", label=label, key=entry_key)
        result = self._form_returns.get(entry_key, False)
        entry["result"] = result
        return result

    def text_input(self, label: str, *, key: str | None = None, value: str | None = None, help: str | None = None) -> str:
        result = value or ""
        if key:
            self.session_state[key] = result
        self._record("text_input", label=label, key=key, help=help, result=result)
        return result

    def number_input(
        self,
        label: str,
        *,
        min_value: object | None = None,
        max_value: object | None = None,
        value: object | None = None,
        step: object | None = None,
        key: str | None = None,
        help: str | None = None,
        format: str | None = None,
    ) -> object:
        result = value
        if key:
            self.session_state[key] = result
        self._record(
            "number_input",
            label=label,
            min_value=min_value,
            max_value=max_value,
            value=value,
            step=step,
            key=key,
            help=help,
            format=format,
        )
        return result

    def slider(
        self,
        label: str,
        *,
        min_value: object | None = None,
        max_value: object | None = None,
        value: object | None = None,
        step: object | None = None,
        key: str | None = None,
        help: str | None = None,
    ) -> object:
        result = value
        if key:
            self.session_state[key] = result
        self._record(
            "slider",
            label=label,
            min_value=min_value,
            max_value=max_value,
            value=value,
            step=step,
            key=key,
            help=help,
        )
        return result

    def checkbox(
        self,
        label: str,
        *,
        value: bool | None = None,
        key: str | None = None,
        help: str | None = None,
        disabled: bool | None = None,
    ) -> bool:
        entry_key = key or label
        if disabled:
            result = bool(value)
        else:
            result = self._checkbox_returns.get(entry_key, bool(value))
        if key:
            self.session_state[key] = result
        self._record(
            "checkbox",
            label=label,
            value=value,
            key=key,
            help=help,
            disabled=disabled,
            result=result,
        )
        return result

    def toggle(
        self,
        label: str,
        *,
        value: bool | None = None,
        key: str | None = None,
        help: str | None = None,
        on_change=None,
    ) -> bool:
        entry_key = key or label
        result = self._toggle_returns.get(entry_key, bool(value))
        if key:
            self.session_state[key] = result
        self._record("toggle", label=label, value=value, key=key, help=help, result=result, on_change=on_change)
        return result

    def multiselect(
        self,
        label: str,
        options: Sequence[object],
        *,
        default: Iterable[object] | None = None,
        key: str | None = None,
        help: str | None = None,
    ) -> list[object]:
        result = list(default or [])
        if key:
            self.session_state[key] = result
        self._record(
            "multiselect",
            label=label,
            options=list(options),
            default=list(default or []),
            key=key,
            help=help,
        )
        return result

    def selectbox(
        self,
        label: str,
        options: Sequence[object],
        *,
        index: int = 0,
        key: str | None = None,
        help: str | None = None,
        on_change=None,
        format_func=None,
    ) -> object:
        options_list = list(options)
        if key and key in self.session_state:
            result = self.session_state[key]
        else:
            if 0 <= index < len(options_list):
                result = options_list[index]
            else:
                result = options_list[0] if options_list else None
            if key:
                self.session_state[key] = result
        self._record(
            "selectbox",
            label=label,
            options=options_list,
            index=index,
            key=key,
            help=help,
            format_func=format_func,
        )
        return result

    def button(self, label: str, *, key: str | None = None, help: str | None = None, **kwargs: Any) -> bool:
        entry_key = key or label
        result = self._button_returns.get(entry_key, False)
        record = {"label": label, "key": entry_key, "help": help, "result": result}
        if kwargs:
            record.update(kwargs)
        self._record("button", **record)
        return result

    def tabs(self, labels: Sequence[str]) -> list[_DummyTab]:
        entry = self._record("tabs", labels=list(labels))
        tabs: list[_DummyTab] = []
        for index, label in enumerate(labels):
            tab_entry: dict[str, Any] = {"type": "tab", "label": label, "index": index, "children": []}
            entry.setdefault("children", []).append(tab_entry)
            tabs.append(_DummyTab(self, tab_entry))
        return tabs

    def columns(self, spec: int | Sequence[object], *, gap: str | None = None) -> list[_DummyColumn]:
        if isinstance(spec, int):
            count = spec
        else:
            count = len(tuple(spec))
        entry = self._record("columns", spec=spec, gap=gap)
        columns: list[_DummyColumn] = []
        for index in range(count):
            column_entry: dict[str, Any] = {"type": "column", "index": index, "children": []}
            entry.setdefault("children", []).append(column_entry)
            columns.append(_DummyColumn(self, column_entry))
        return columns

    def empty(self) -> _DummyPlaceholder:
        entry = self._record("empty")
        return _DummyPlaceholder(self, entry)

    def container(self, *, border: bool | None = None) -> _DummyContext:
        entry = self._record("container", border=border)
        return _DummyContext(self, entry)

    def dataframe(
        self,
        data: Any,
        *,
        column_config=None,
        column_order=None,
        hide_index: bool | None = None,
        width: object | None = None,
    ) -> None:
        self._record(
            "dataframe",
            data=data,
            column_config=column_config,
            column_order=column_order,
            hide_index=hide_index,
            width=width,
        )

    def download_button(
        self,
        label: str,
        data: Any,
        *,
        file_name: str | None = None,
        mime: str | None = None,
        key: str | None = None,
    ) -> None:
        self._record(
            "download_button",
            label=label,
            data=data,
            file_name=file_name,
            mime=mime,
            key=key,
        )

    def altair_chart(
        self,
        chart: Any,
        *,
        key: str | None = None,
        width: object | None = None,
    ) -> None:
        self._record(
            "altair_chart",
            chart=chart,
            key=key,
            width=width,
        )

    def cache_resource(self, func=None, **kwargs):
        if func is None:
            def decorator(wrapped):
                return wrapped

            return decorator
        return func

    cache_data = cache_resource


_streamlit_core = _DummyStreamlitCore()
_streamlit_module = ModuleType("streamlit")


def _delegate(name: str):
    attribute = getattr(_streamlit_core, name)
    setattr(_streamlit_module, name, attribute)


for _name in (
    "header",
    "subheader",
    "caption",
    "markdown",
    "metric",
    "write",
    "info",
    "warning",
    "error",
    "success",
    "spinner",
    "expander",
    "form",
    "form_submit_button",
    "text_input",
    "number_input",
    "slider",
    "checkbox",
    "multiselect",
    "selectbox",
    "button",
    "link_button",
    "page_link",
    "tabs",
    "columns",
    "empty",
    "dataframe",
    "download_button",
    "line_chart",
    "area_chart",
    "altair_chart",
    "cache_resource",
    "cache_data",
):
    _delegate(_name)

_streamlit_module.sidebar = _streamlit_core.sidebar
_streamlit_module.session_state = _streamlit_core.session_state
_streamlit_module.secrets = _streamlit_core.secrets
_streamlit_module.set_button_result = _streamlit_core.set_button_result
_streamlit_module.set_checkbox_result = _streamlit_core.set_checkbox_result
_streamlit_module.set_form_submit_result = _streamlit_core.set_form_submit_result
_streamlit_module.get_records = _streamlit_core.get_records
_streamlit_module.reset = _streamlit_core.reset
_streamlit_module._core = _streamlit_core
_streamlit_module.delta_generator = SimpleNamespace(DeltaGenerator=object)
_streamlit_module.column_config = SimpleNamespace(Column=_DummyColumnConfig, LinkColumn=_DummyColumnConfig)


def __getattr__(name: str):
    return getattr(_streamlit_core, name)


_streamlit_module.__getattr__ = __getattr__  # type: ignore[attr-defined]

sys.modules.setdefault("streamlit", _streamlit_module)

_streamlit_runtime_module = ModuleType("streamlit.runtime")
_streamlit_runtime_secrets_module = ModuleType("streamlit.runtime.secrets")
_streamlit_runtime_secrets_module.Secrets = _DummySecrets  # type: ignore[attr-defined]
_streamlit_runtime_secrets_module.StreamlitSecretNotFoundError = (
    _DummyStreamlitSecretNotFoundError
)
_streamlit_runtime_module.secrets = _streamlit_runtime_secrets_module  # type: ignore[attr-defined]

_runtime_namespace = SimpleNamespace(
    secrets=_streamlit_runtime_secrets_module,
    _page_registry={},
)
_streamlit_module.runtime = _runtime_namespace
_streamlit_core.runtime = _runtime_namespace

sys.modules.setdefault("streamlit.runtime", _streamlit_runtime_module)
sys.modules.setdefault("streamlit.runtime.secrets", _streamlit_runtime_secrets_module)


class _DummyAltairExpr:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class _DummyAltairChart:
    def __init__(self, data: Any) -> None:
        self.data = data

    def mark_arc(self, **kwargs: Any) -> "_DummyAltairChart":
        return self

    def encode(self, **kwargs: Any) -> "_DummyAltairChart":
        return self

    def properties(self, **kwargs: Any) -> "_DummyAltairChart":
        return self


_altair_module = ModuleType("altair")
_altair_module.Chart = lambda data: _DummyAltairChart(data)  # type: ignore[attr-defined]
_altair_module.Theta = _DummyAltairExpr
_altair_module.Color = _DummyAltairExpr
_altair_module.Tooltip = _DummyAltairExpr
_altair_module.Legend = _DummyAltairExpr

sys.modules.setdefault("altair", _altair_module)


import pytest


@pytest.fixture(autouse=True, scope="session")
def _tokens_env() -> None:
    previous_fastapi = os.environ.get("FASTAPI_TOKENS_KEY")
    previous_iol = os.environ.get("IOL_TOKENS_KEY")
    os.environ["FASTAPI_TOKENS_KEY"] = "MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA="
    os.environ["IOL_TOKENS_KEY"] = "MTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTE="
    try:
        yield
    finally:
        if previous_fastapi is not None:
            os.environ["FASTAPI_TOKENS_KEY"] = previous_fastapi
        else:
            os.environ.pop("FASTAPI_TOKENS_KEY", None)
        if previous_iol is not None:
            os.environ["IOL_TOKENS_KEY"] = previous_iol
        else:
            os.environ.pop("IOL_TOKENS_KEY", None)


@pytest.fixture(autouse=True)
def _reset_streamlit_stub() -> None:
    _streamlit_core.reset()


@pytest.fixture
def streamlit_stub() -> _DummyStreamlitCore:
    return _streamlit_core


@pytest.fixture(params=("base", "logging", "ui"))
def fake_st(request: pytest.FixtureRequest) -> BaseFakeStreamlit:
    param = request.param
    config: dict[str, Any]
    if isinstance(param, tuple):
        variant, config = param
    elif isinstance(param, dict):
        variant, config = "ui", param
    else:
        variant, config = str(param), {}

    if variant == "logging":
        return FakeStreamlit()

    if variant == "ui":
        kwargs = {**config}
        return UIFakeStreamlit(**kwargs)

    return BaseFakeStreamlit()
