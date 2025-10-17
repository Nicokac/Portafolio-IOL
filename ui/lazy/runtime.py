"""Runtime helpers for Streamlit lazy fragments."""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

import streamlit as st

from shared.telemetry import log_default_telemetry

logger = logging.getLogger(__name__)

_SCOPE: ContextVar[str | None] = ContextVar("ui_lazy_fragment_scope", default=None)
_COMPONENT: ContextVar[str | None] = ContextVar("ui_lazy_fragment_component", default=None)
_DATASET: ContextVar[str | None] = ContextVar("ui_lazy_fragment_dataset", default=None)


@dataclass
class FragmentContext:
    """Information about the currently active lazy fragment."""

    name: str
    scope: str

    def stop(self) -> None:
        """Attempt to stop the Streamlit script when running outside fragments."""

        if self.scope == "fragment":
            return
        stop_callable = getattr(st, "stop", None)
        if callable(stop_callable):
            try:
                stop_callable()
            except Exception:  # pragma: no cover - defensive guard for stubs
                logger.debug("Lazy fragment stop failed for %s", self.name, exc_info=True)


def current_scope() -> str | None:
    """Return the scope of the active lazy fragment, if any."""

    return _SCOPE.get()


def in_form_scope() -> bool:
    """Whether the current lazy fragment fallback uses a Streamlit form."""

    return current_scope() == "form"


@contextmanager
def lazy_fragment(
    name: str,
    *,
    component: str,
    dataset_token: str | None = None,
) -> Iterator[FragmentContext]:
    """Context manager that isolates reruns for lazy components."""

    fragment_factory = _fragment_factory()
    form_callable = None if fragment_factory else _form_callable()

    if fragment_factory is not None:
        scope = "fragment"
    elif form_callable is not None:
        scope = "form"
    else:
        scope = "global"

    with _enter_scope(name, fragment_factory, form_callable, scope):
        scope_token = _SCOPE.set(scope)
        component_token = _COMPONENT.set(component)
        dataset_token = dataset_token or None
        dataset_token_var = _DATASET.set(dataset_token)
        _log_scope(scope, component, dataset_token)
        try:
            yield FragmentContext(name=name, scope=scope)
        finally:
            _SCOPE.reset(scope_token)
            _COMPONENT.reset(component_token)
            _DATASET.reset(dataset_token_var)


def _fragment_factory():
    for attr in ("fragment", "experimental_fragment"):
        factory = getattr(st, attr, None)
        if callable(factory):
            return lambda nm: factory(nm)
    return None


def _form_callable():
    form = getattr(st, "form", None)
    return form if callable(form) else None


@contextmanager
def _enter_scope(name: str, fragment_factory, form_callable, scope: str):
    if scope == "fragment" and fragment_factory is not None:
        with fragment_factory(name):
            yield
            return
    if scope == "form" and form_callable is not None:
        form_key = f"{name}__form"
        with form_callable(form_key):
            yield
            return
    with _container_context():
        yield


def _container_context():
    container = getattr(st, "container", None)
    if callable(container):
        return container()
    placeholder = getattr(st, "empty", None)
    if callable(placeholder):
        empty_container = placeholder()
        container_callable = getattr(empty_container, "container", None)
        if callable(container_callable):
            return container_callable()
    return nullcontext()


def _log_scope(scope: str, component: str, dataset_token: str | None) -> None:
    try:
        log_default_telemetry(
            phase="ui.rerun_scope",
            dataset_hash=str(dataset_token or ""),
            extra={
                "ui_rerun_scope": scope,
                "lazy_loaded_component": component,
            },
        )
    except Exception:  # pragma: no cover - defensive guard for telemetry failures
        logger.debug("No se pudo registrar ui_rerun_scope para %s", component, exc_info=True)
