from __future__ import annotations

from collections.abc import Iterable, Sequence
from types import ModuleType, SimpleNamespace
import sys
from typing import Any


class _DummySidebar:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.headers: list[str] = []
        self.captions: list[str] = []
        self.markdowns: list[str] = []
        self.elements: list[dict[str, Any]] = []

    def header(self, text: object) -> None:
        value = str(text)
        self.headers.append(value)
        self.elements.append({"type": "header", "text": value})

    def caption(self, text: object) -> None:
        value = str(text)
        self.captions.append(value)
        self.elements.append({"type": "caption", "text": value})

    def markdown(self, text: object) -> None:
        value = str(text)
        self.markdowns.append(value)
        self.elements.append({"type": "markdown", "text": value})


class _DummyContext:
    def __init__(self, core: "_DummyStreamlitCore", entry: dict[str, Any]) -> None:
        self._core = core
        self._entry = entry

    def __enter__(self) -> "_DummyContext":
        self._core._context_stack.append(self._entry.setdefault("children", []))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._core._context_stack.pop()
        return False


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

    def metric(self, label: object, value: object, *, delta: object = None, help: object = None) -> None:
        record = {
            "type": "metric",
            "label": str(label),
            "value": value,
            "delta": delta,
            "help": help,
        }
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
        self.sidebar = _DummySidebar()
        self.session_state: dict[str, Any] = {}
        self.secrets: dict[str, Any] = {}
        self._calls: list[dict[str, Any]] = []
        self._all_records: list[dict[str, Any]] = []
        self._context_stack: list[list[dict[str, Any]]] = [self._calls]
        self._button_returns: dict[str, bool] = {}
        self._checkbox_returns: dict[str, bool] = {}
        self._form_returns: dict[str, bool] = {}

    # Public helpers -----------------------------------------------------
    def reset(self) -> None:
        self.sidebar.reset()
        self.session_state.clear()
        self.secrets.clear()
        self._calls.clear()
        self._all_records.clear()
        self._context_stack = [self._calls]
        self._button_returns.clear()
        self._checkbox_returns.clear()
        self._form_returns.clear()

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
        return entry

    # Streamlit API subset ----------------------------------------------
    def header(self, text: object) -> None:
        self._record("header", text=str(text))

    def subheader(self, text: object) -> None:
        self._record("subheader", text=str(text))

    def caption(self, text: object) -> None:
        self._record("caption", text=str(text))

    def markdown(self, text: object) -> None:
        self._record("markdown", text=str(text))

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

    def spinner(self, text: object) -> _DummyContext:
        entry = self._record("spinner", text=str(text))
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
    ) -> bool:
        entry_key = key or label
        result = self._checkbox_returns.get(entry_key, bool(value))
        if key:
            self.session_state[key] = result
        self._record("checkbox", label=label, value=value, key=key, help=help, result=result)
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

    def dataframe(self, data: Any, *, use_container_width: bool | None = None, column_config=None, column_order=None) -> None:
        self._record(
            "dataframe",
            data=data,
            use_container_width=use_container_width,
            column_config=column_config,
            column_order=column_order,
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

    def altair_chart(self, chart: Any, *, use_container_width: bool | None = None, key: str | None = None) -> None:
        self._record("altair_chart", chart=chart, use_container_width=use_container_width, key=key)

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
    "tabs",
    "columns",
    "empty",
    "dataframe",
    "download_button",
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


@pytest.fixture(autouse=True)
def _reset_streamlit_stub() -> None:
    _streamlit_core.reset()


@pytest.fixture
def streamlit_stub() -> _DummyStreamlitCore:
    return _streamlit_core
