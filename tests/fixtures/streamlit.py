from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd


class BaseFakeStreamlit:
    """Lightweight Streamlit stub shared across the test-suite."""

    @dataclass
    class _Spinner:
        owner: "BaseFakeStreamlit"
        message: str

        def __enter__(self) -> "BaseFakeStreamlit._Spinner":
            self.owner._enter_spinner(self.message)
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            self.owner._exit_spinner(self.message)
            return False

    def __init__(self) -> None:
        self.session_state: dict[str, Any] = {}
        self.messages: list[tuple[str, str]] = []
        self.spinner_messages: list[str] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self._rerun_called = False

    # --- Core behaviours -------------------------------------------------
    def spinner(self, message: str) -> "BaseFakeStreamlit._Spinner":
        return BaseFakeStreamlit._Spinner(self, message)

    def _enter_spinner(self, message: str) -> None:
        self.spinner_messages.append(message)

    def _exit_spinner(self, _: str) -> None:  # pragma: no cover - trivial hook
        return None

    def info(self, message: str) -> None:  # pragma: no cover - defensive
        value = str(message)
        self.messages.append(("info", value))
        self.infos.append(value)

    def warning(self, message: str) -> None:  # pragma: no cover - defensive
        value = str(message)
        self.messages.append(("warning", value))
        self.warnings.append(value)

    def error(self, message: str) -> None:  # pragma: no cover - defensive
        value = str(message)
        self.messages.append(("error", value))
        self.errors.append(value)

    def dataframe(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - noop
        return None

    def caption(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - noop
        return None

    def rerun(self) -> None:  # pragma: no cover - defensive
        self._rerun_called = True

    def stop(self) -> None:  # pragma: no cover - defensive
        raise RuntimeError("streamlit.stop() should not be invoked in tests")

    def clear_session_state(self) -> None:
        self.session_state.clear()


class LoggingMixin:
    """Augments the base stub with event logs used by integration tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.spinner_events: list[tuple[str, str]] = []

    def _enter_spinner(self, message: str) -> None:
        super()._enter_spinner(message)
        self.spinner_events.append(("start", message))

    def _exit_spinner(self, message: str) -> None:
        self.spinner_events.append(("stop", message))

    def info(self, message: str) -> None:  # pragma: no cover - defensive
        super().info(message)

    def warning(self, message: str) -> None:  # pragma: no cover - defensive
        super().warning(message)

    def error(self, message: str) -> None:  # pragma: no cover - defensive
        super().error(message)


class _DummyContainer:
    def __enter__(self) -> "_DummyContainer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - standard context signature
        return None


class _ContextManager:
    def __init__(self, owner: "UIFlowMixin") -> None:
        self._owner = owner

    def __enter__(self) -> "_ContextManager":  # noqa: D401 - thin wrapper
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - thin wrapper
        return None

    def number_input(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner.number_input(*args, **kwargs)

    def selectbox(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner.selectbox(*args, **kwargs)

    def plotly_chart(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner.plotly_chart(*args, **kwargs)

    def info(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner.info(*args, **kwargs)

    def metric(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner.metric(*args, **kwargs)

    def line_chart(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner.line_chart(*args, **kwargs)

    def bar_chart(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner.bar_chart(*args, **kwargs)

    def empty(self) -> "_Placeholder":
        return self._owner.empty()


class _SpinnerContext(_ContextManager):
    def __init__(self, owner: "UIFlowMixin", base_cm: BaseFakeStreamlit._Spinner) -> None:
        super().__init__(owner)
        self._base_cm = base_cm

    def __enter__(self) -> "_SpinnerContext":
        self._base_cm.__enter__()
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        super().__exit__(exc_type, exc, tb)
        return self._base_cm.__exit__(exc_type, exc, tb)


class _Placeholder:
    def __init__(self, owner: "UIFlowMixin") -> None:
        self._owner = owner

    def empty(self) -> None:
        return None

    def container(self) -> _ContextManager:
        return _ContextManager(self._owner)

    def markdown(self, body: str, *, unsafe_allow_html: bool = False) -> None:
        self._owner.markdowns.append({"body": body, "unsafe": unsafe_allow_html, "placeholder": True})

    def info(self, message: str) -> None:
        self._owner.info(message)

    def caption(self, text: str) -> None:
        self._owner.caption(text)

    def write(self, body: str) -> None:
        self._owner.markdowns.append({"body": body, "unsafe": False, "placeholder": True, "write": True})

    def checkbox(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner.checkbox(*args, **kwargs)

    def toggle(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner.toggle(*args, **kwargs)


class UIFlowMixin:
    """Rich UI oriented behaviours extracted from the UI contract tests."""

    def __init__(
        self,
        radio_sequence: Iterable[int] | None = None,
        selectbox_defaults: dict[str, Any] | None = None,
        *,
        multiselect_responses: dict[str, Sequence[Any]] | None = None,
        checkbox_values: dict[str, Sequence[bool] | bool] | None = None,
        slider_values: dict[str, Any] | None = None,
        button_clicks: dict[str, Sequence[bool] | bool] | None = None,
    ) -> None:
        super().__init__()
        radio_sequence = radio_sequence or []
        self._radio_iter: Iterator[int] = iter(radio_sequence)
        self._selectbox_defaults = selectbox_defaults or {}
        self._multiselect_responses = {key: list(value) for key, value in (multiselect_responses or {}).items()}
        self._checkbox_values: dict[str, list[bool]] = {}
        for raw_key, value in (checkbox_values or {}).items():
            key = str(raw_key)
            if isinstance(value, (list, tuple)):
                self._checkbox_values[key] = [bool(item) for item in value]
            else:
                self._checkbox_values[key] = [bool(value)]
        self._slider_values = dict(slider_values or {})
        self._button_clicks: dict[str, list[bool]] = {}
        for raw_key, value in (button_clicks or {}).items():
            key = str(raw_key)
            if isinstance(value, (list, tuple)):
                self._button_clicks[key] = [bool(item) for item in value]
            else:
                self._button_clicks[key] = [bool(value)]
        self.radio_calls: list[dict[str, Any]] = []
        self.selectbox_calls: list[dict[str, Any]] = []
        self.multiselect_calls: list[dict[str, Any]] = []
        self.number_input_calls: list[dict[str, Any]] = []
        self.headers: list[str] = []
        self.subheaders: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.successes: list[str] = []
        self.plot_calls: list[dict[str, Any]] = []
        self.line_charts: list[pd.DataFrame] = []
        self.bar_charts: list[dict[str, Any]] = []
        self.metrics: list[tuple[Any, Any, Any, dict[str, Any]]] = []
        self.markdowns: list[dict[str, Any]] = []
        self.captions: list[str] = []
        self.dataframes: list[tuple[Any, dict[str, Any]]] = []
        self.checkbox_calls: list[dict[str, Any]] = []
        self.slider_calls: list[dict[str, Any]] = []
        self.download_buttons: list[dict[str, Any]] = []
        self.button_calls: list[dict[str, Any]] = []
        self._placeholders: list[_Placeholder] = []

    # ---- Core widgets -------------------------------------------------
    def radio(
        self,
        label: str,
        *,
        options: Sequence[int],
        format_func,
        index: int = 0,
        horizontal: bool,
        **kwargs: Any,
    ) -> int:
        try:
            value = next(self._radio_iter)
        except StopIteration:
            value = options[index] if options else 0
        display_labels = [format_func(opt) for opt in options]
        record = {
            "label": label,
            "options": list(options),
            "index": index,
            "display_labels": display_labels,
        }
        key = kwargs.get("key")
        if key is not None:
            record["key"] = key
            self.session_state[key] = value
        self.radio_calls.append(record)
        return value

    def selectbox(
        self,
        label: str,
        options: Sequence[Any],
        index: int = 0,
        key: str | None = None,
        **_: Any,
    ) -> Any:
        self.selectbox_calls.append({"label": label, "options": list(options), "key": key})
        if label in self._selectbox_defaults:
            result = self._selectbox_defaults[label]
        else:
            result = options[index] if options else None
        if key is not None:
            self.session_state[key] = result
        return result

    def multiselect(
        self,
        label: str,
        options: Sequence[Any],
        *,
        default: Sequence[Any] | None = None,
        format_func=lambda x: x,
        key: str | None = None,
    ) -> list[Any]:
        rendered = [format_func(opt) for opt in options] if format_func else list(options)
        record = {
            "label": label,
            "options": list(options),
            "rendered": rendered,
            "default": list(default) if default is not None else [],
            "key": key,
        }
        self.multiselect_calls.append(record)
        state_key = key or label
        if state_key in self.session_state:
            selection = self.session_state[state_key]
        elif state_key in self._multiselect_responses:
            selection = list(self._multiselect_responses[state_key])
        else:
            selection = list(default) if default is not None else []
        if key is not None:
            self.session_state[key] = list(selection)
        return list(selection)

    def number_input(
        self,
        label: str,
        *,
        min_value: Any,
        max_value: Any,
        value: Any,
        step: Any,
    ) -> Any:
        self.number_input_calls.append(
            {
                "label": label,
                "min_value": min_value,
                "max_value": max_value,
                "value": value,
                "step": step,
            }
        )
        return value

    def checkbox(self, label: str, *, value: bool = False, key: str | None = None) -> bool:
        record = {"label": label, "value": value, "key": key}
        self.checkbox_calls.append(record)
        state_key = key or label
        queue = self._checkbox_values.get(str(state_key))
        if queue:
            result = queue.pop(0)
            if not queue:
                self._checkbox_values.pop(str(state_key), None)
        elif key is not None and key in self.session_state:
            result = bool(self.session_state[key])
        else:
            stored = self.session_state.get(state_key)
            result = bool(stored) if stored is not None else value
        if key is not None:
            self.session_state[key] = result
        return result

    def toggle(self, label: str, *, value: bool = False, key: str | None = None) -> bool:
        return self.checkbox(label, value=value, key=key)

    def slider(
        self,
        label: str,
        *,
        min_value: Any,
        max_value: Any,
        value: Any,
        step: Any,
        key: str | None = None,
    ) -> Any:
        record = {
            "label": label,
            "min_value": min_value,
            "max_value": max_value,
            "value": value,
            "step": step,
            "key": key,
        }
        self.slider_calls.append(record)
        state_key = key or label
        result = self._slider_values.get(state_key, value)
        if key is not None:
            self.session_state[key] = result
        return result

    def columns(self, layout: Sequence[Any] | int) -> list[_ContextManager]:
        if isinstance(layout, int):
            return [_ContextManager(self) for _ in range(layout)]
        return [_ContextManager(self) for _ in layout]

    def container(self) -> _ContextManager:
        return _ContextManager(self)

    def tabs(self, labels: Sequence[str]) -> list[_ContextManager]:
        return [_ContextManager(self) for _ in labels]

    def expander(self, label: str, *_, **__):  # noqa: ANN001 - mimics streamlit signature
        return _ContextManager(self)

    def spinner(self, text: str = "", *_, **__) -> _ContextManager:  # type: ignore[override]
        base_cm = super().spinner(text)
        return _SpinnerContext(self, base_cm)

    def empty(self) -> _Placeholder:
        placeholder = _Placeholder(self)
        self._placeholders.append(placeholder)
        return placeholder

    # ---- Feedback widgets ---------------------------------------------
    def subheader(self, text: str) -> None:
        self.subheaders.append(text)

    def header(self, text: str) -> None:
        self.headers.append(text)

    def info(self, message: str) -> None:  # type: ignore[override]
        super().info(message)
        self.warnings.append(str(message))

    def warning(self, message: str) -> None:  # type: ignore[override]
        super().warning(message)

    def error(self, message: str) -> None:  # type: ignore[override]
        super().error(message)

    def success(self, message: str) -> None:
        self.successes.append(message)

    def caption(self, text: str, *_: Any, **__: Any) -> None:  # type: ignore[override]
        super().caption(text)
        self.captions.append(text)

    def plotly_chart(self, fig: Any, **kwargs: Any) -> None:
        self.plot_calls.append({"fig": fig, "kwargs": kwargs})

    def line_chart(self, data: pd.DataFrame) -> None:
        self.line_charts.append(data)

    def bar_chart(self, data: pd.DataFrame, **kwargs: Any) -> None:
        self.bar_charts.append({"data": data, "kwargs": kwargs})

    def write(self, *_: Any, **__: Any) -> None:
        return None

    def dataframe(self, data: Any, **kwargs: Any) -> None:  # type: ignore[override]
        self.dataframes.append((data, kwargs))

    def metric(
        self,
        label: str,
        value: Any,
        delta: Any | None = None,
        *,
        help: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.metrics.append((label, value, delta, {"help": help, **kwargs}))

    def markdown(self, body: str, *, unsafe_allow_html: bool = False) -> None:
        self.markdowns.append({"body": body, "unsafe": unsafe_allow_html})

    def download_button(
        self,
        label: str,
        data: Any,
        *,
        file_name: str | None = None,
        mime: str | None = None,
        key: str | None = None,
        disabled: bool | None = None,
        **kwargs: Any,
    ) -> None:
        record = {
            "label": label,
            "data": data,
            "file_name": file_name,
            "mime": mime,
            "key": key,
            "disabled": disabled,
        }
        if kwargs:
            record.update(kwargs)
        self.download_buttons.append(record)

    def divider(self) -> None:
        return None

    def button(
        self,
        label: str,
        *,
        key: str | None = None,
        help: str | None = None,
        icon: str | None = None,
    ) -> bool:
        record = {"label": label, "key": key, "help": help, "icon": icon}
        self.button_calls.append(record)
        state_key = key or label
        queue = self._button_clicks.get(str(state_key))
        if queue:
            result = queue.pop(0)
            if not queue:
                self._button_clicks.pop(str(state_key), None)
            return result
        return False


class FakeStreamlit(LoggingMixin, BaseFakeStreamlit):
    """Default fake streamlit used by fixtures."""


class UIFakeStreamlit(UIFlowMixin, LoggingMixin, BaseFakeStreamlit):
    """UI oriented fake streamlit combining the mixins."""


__all__ = [
    "BaseFakeStreamlit",
    "LoggingMixin",
    "FakeStreamlit",
    "UIFakeStreamlit",
    "UIFlowMixin",
    "_DummyContainer",
    "_ContextManager",
    "_Placeholder",
]
