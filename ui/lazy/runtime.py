"""Runtime helpers for Streamlit lazy fragments."""

from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
import time
from typing import Any, Iterator

import streamlit as st

from shared.fragment_state import (
    get_fragment_state_guardian,
    register_fragment_auto_load_context,
    reset_fragment_auto_load_context,
)
from shared.telemetry import log_default_telemetry

logger = logging.getLogger(__name__)

fragment_context_ready: bool = True

_FRAGMENT_CONTEXT_TIMEOUT_S = 0.25
_FRAGMENT_CONTEXT_POLL_INTERVAL_S = 0.05
_FRAGMENT_CONTEXT_RERUN_DATASETS: set[str] = set()
_PERSISTENT_PLACEHOLDERS: dict[str, Any] = {}

_SCOPE: ContextVar[str | None] = ContextVar("ui_lazy_fragment_scope", default=None)
_COMPONENT: ContextVar[str | None] = ContextVar("ui_lazy_fragment_component", default=None)
_DATASET: ContextVar[str | None] = ContextVar("ui_lazy_fragment_dataset", default=None)


def _set_persistent_placeholder(fragment_id: str, placeholder: Any) -> None:
    """Store a placeholder for future fallback renders."""

    if not fragment_id or placeholder is None:
        return
    try:
        _PERSISTENT_PLACEHOLDERS[fragment_id] = placeholder
    except Exception:  # pragma: no cover - defensive safeguard for unexpected values
        logger.debug(
            "[LazyRuntime] failed_to_store_persistent_placeholder fragment=%s",
            fragment_id,
            exc_info=True,
        )


def _get_persistent_placeholder(fragment_id: str) -> Any | None:
    """Return the previously stored placeholder if available."""

    if not fragment_id:
        return None
    return _PERSISTENT_PLACEHOLDERS.get(fragment_id)


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


def current_component() -> str | None:
    """Return the component name currently associated with the fragment."""

    return _COMPONENT.get()


def current_dataset_token() -> str | None:
    """Return the dataset token associated with the active fragment."""

    return _DATASET.get()


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

    hydration_ready = True
    hydration_wait_ms = 0
    dataset_marker = dataset_token or current_dataset_token()
    dataset_hash = str(dataset_marker or "")
    guardian = None
    try:
        guardian = get_fragment_state_guardian()
    except Exception:  # pragma: no cover - defensive fallback for tests
        guardian = None
    wait_method = getattr(guardian, "wait_for_hydration", None) if guardian else None
    if callable(wait_method):
        wait_start = time.perf_counter()
        try:
            hydration_ready = bool(wait_method(dataset_marker))
        except TypeError:
            hydration_ready = bool(wait_method())  # type: ignore[misc]
        except Exception:  # pragma: no cover - defensive safeguard
            hydration_ready = True
        hydration_wait_ms = int((time.perf_counter() - wait_start) * 1000)
        log_level = logger.info if hydration_ready else logger.warning
        log_level(
            "[LazyRuntime] fragment_hydration_complete",
            extra={
                "fragment": name,
                "component": component,
                "dataset_hash": str(dataset_marker or ""),
                "hydrated": hydration_ready,
                "wait_ms": hydration_wait_ms,
            },
        )

    fragment_factory_builder = _fragment_factory()
    form_callable = None if fragment_factory_builder else _form_callable()
    context_ready = True

    if fragment_factory_builder is not None:
        context_ready = _wait_for_fragment_context_ready(
            fragment=name,
            component=component,
            dataset_hash=dataset_hash,
            hydration_ready=hydration_ready,
            timeout=_FRAGMENT_CONTEXT_TIMEOUT_S,
            poll_interval=_FRAGMENT_CONTEXT_POLL_INTERVAL_S,
        )
        if not (hydration_ready and context_ready):
            fragment_factory_builder = None
        _record_fragment_visibility(
            component=component,
            dataset_hash=dataset_hash,
            visible=bool(fragment_factory_builder),
        )

    fragment_factory = fragment_factory_builder

    if fragment_factory is not None:
        scope = "fragment"
    elif form_callable is not None:
        scope = "form"
    else:
        scope = "global"

    fallback_placeholder = None
    if not context_ready or fragment_factory is None:
        fallback_placeholder = _handle_fragment_fallback(name, context_ready, scope)

    auto_load_token = register_fragment_auto_load_context(
        scope=scope,
        fragment_factory_builder=fragment_factory,
        context_ready=context_ready,
    )
    try:
        with _enter_scope(
            name,
            fragment_factory,
            form_callable,
            scope,
            fallback_placeholder=fallback_placeholder,
        ):
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
    finally:
        reset_fragment_auto_load_context(auto_load_token)


_FRAGMENT_WARNING_EMITTED = False


def _fragment_factory():
    factories = []
    for attr in ("fragment", "experimental_fragment"):
        factory = getattr(st, attr, None)
        if callable(factory):
            factories.append(factory)

    if not factories:
        _warn_fragment_fallback()
        return None

    def _build(name: str, _factories=factories):
        for base_factory in _factories:
            context = _resolve_fragment_context(base_factory, name)
            if context is not None:
                return context

        _warn_fragment_fallback()
        return _container_context()

    return _build


def _resolve_fragment_context(factory, name: str):
    try:
        candidate = factory()
    except TypeError:
        try:
            candidate = factory(name)
        except TypeError:
            logger.debug(
                "Streamlit fragment factory %r rejected lazy fragment %s", factory, name
            )
            return None

    manager = _coerce_fragment_candidate(candidate)
    if manager is None:
        logger.debug(
            "Streamlit fragment factory %r returned unsupported value %r", factory, candidate
        )
    return manager


def _coerce_fragment_candidate(candidate):
    if candidate is None:
        return None

    if hasattr(candidate, "__enter__") and hasattr(candidate, "__exit__"):
        return candidate

    if callable(candidate):
        try:
            resolved = candidate()
        except TypeError:
            return None
        return _coerce_fragment_candidate(resolved)

    return None


def _ensure_context_manager(candidate):
    """Normalize Streamlit fragment factories into context managers."""

    if candidate is None:
        return _container_context()

    if hasattr(candidate, "__enter__") and hasattr(candidate, "__exit__"):
        return candidate

    if callable(candidate):
        resolved = candidate()
        if hasattr(resolved, "__enter__") and hasattr(resolved, "__exit__"):
            return resolved

    logger.debug(
        "Streamlit fragment factory produced unexpected value %r; falling back to container",
        candidate,
    )
    return _container_context()


def _warn_fragment_fallback() -> None:
    global _FRAGMENT_WARNING_EMITTED
    if _FRAGMENT_WARNING_EMITTED:
        return
    _FRAGMENT_WARNING_EMITTED = True
    logger.warning("⚠️ Streamlit fragment factory fallback to container()")


def _form_callable():
    form = getattr(st, "form", None)
    return form if callable(form) else None


@contextmanager
def _enter_scope(
    name: str,
    fragment_factory,
    form_callable,
    scope: str,
    *,
    fallback_placeholder: Any | None = None,
):
    contexts = []

    if fallback_placeholder is not None:
        contexts.append(_ensure_context_manager(fallback_placeholder))

    primary_context = None
    if scope == "fragment" and fragment_factory is not None:
        primary_context = _ensure_context_manager(fragment_factory(name))
    elif scope == "form" and form_callable is not None:
        form_key = f"{name}__form"
        primary_context = _ensure_context_manager(form_callable(form_key))
    elif not contexts:
        primary_context = _container_context()

    if primary_context is not None:
        contexts.append(primary_context)

    if not contexts:
        contexts.append(nullcontext())

    with ExitStack() as stack:
        stored_placeholder: Any | None = None
        for ctx in contexts:
            entered = stack.enter_context(ctx)
            if stored_placeholder is None and entered is not None:
                stored_placeholder = entered
        if stored_placeholder is not None:
            _set_persistent_placeholder(name, stored_placeholder)
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


def _handle_fragment_fallback(
    fragment_id: str, context_ready: bool, scope: str
) -> Any | None:
    """Resolve the container to use when fragment rendering falls back."""

    try:  # pragma: no cover - optional telemetry hook
        from shared.telemetry import log as telemetry_log  # type: ignore
    except ImportError:  # pragma: no cover - telemetry helper may not exist
        telemetry_log = None

    reason = "context_not_ready" if not context_ready else "factory_none"
    logger.info("[LazyRuntime] fallback_triggered reason=%s", reason)

    placeholder = _get_persistent_placeholder(fragment_id)
    reused = placeholder is not None
    if placeholder is None:
        logger.warning(
            "[LazyRuntime] no_persistent_placeholder_found fragment=%s",
            fragment_id,
        )
        container_fn = getattr(st, "container", None)
        if callable(container_fn):
            placeholder = container_fn()
        else:
            empty_fn = getattr(st, "empty", None)
            if callable(empty_fn):
                try:
                    empty_placeholder = empty_fn()
                    container_method = getattr(empty_placeholder, "container", None)
                    if callable(container_method):
                        placeholder = container_method()
                except Exception:  # pragma: no cover - defensive safeguard
                    logger.debug(
                        "[LazyRuntime] failed_to_create_fallback_container",
                        exc_info=True,
                    )
        if placeholder is None:
            placeholder = _container_context()

    logger.info(
        "[LazyRuntime] fallback_reuse_persistent_container=%s",
        reused,
    )
    logger.info("[LazyRuntime] context_ready=%s scope=%s", context_ready, scope)

    if telemetry_log is not None:
        try:
            telemetry_log(
                "portfolio.fragment_visible",
                fragment_id=fragment_id,
                visible=bool(context_ready),
            )
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug(
                "[LazyRuntime] telemetry_log_failed fragment=%s",
                fragment_id,
                exc_info=True,
            )
    else:
        try:
            log_default_telemetry(
                phase="portfolio.fragment_visibility",
                dataset_hash="",
                extra={"portfolio.fragment_visible": bool(context_ready)},
            )
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug(
                "[LazyRuntime] fallback_telemetry_log_failed fragment=%s",
                fragment_id,
                exc_info=True,
            )

    return placeholder


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


def _wait_for_fragment_context_ready(
    *,
    fragment: str,
    component: str,
    dataset_hash: str,
    hydration_ready: bool,
    timeout: float,
    poll_interval: float,
) -> bool:
    start = time.perf_counter()
    logger.info(
        "[LazyRuntime] wait_for_fragment_context_start",
        extra={
            "fragment": fragment,
            "component": component,
            "dataset_hash": dataset_hash,
            "hydrated": hydration_ready,
        },
    )
    if not hydration_ready:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "[LazyRuntime] wait_for_fragment_context_end",
            extra={
                "fragment": fragment,
                "component": component,
                "dataset_hash": dataset_hash,
                "hydrated": hydration_ready,
                "context_ready": False,
                "duration_ms": duration_ms,
            },
        )
        return False

    wait_deadline = start + max(float(timeout), 0.0)
    poll_delay = poll_interval if poll_interval > 0 else _FRAGMENT_CONTEXT_POLL_INTERVAL_S
    if fragment_context_ready:
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "[LazyRuntime] wait_for_fragment_context_end",
            extra={
                "fragment": fragment,
                "component": component,
                "dataset_hash": dataset_hash,
                "hydrated": hydration_ready,
                "context_ready": True,
                "duration_ms": duration_ms,
            },
        )
        return True

    while time.perf_counter() < wait_deadline:
        remaining = wait_deadline - time.perf_counter()
        sleep_for = poll_delay if remaining > poll_delay else max(remaining, 0.0)
        if sleep_for > 0:
            time.sleep(sleep_for)
        if fragment_context_ready:
            duration_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                "[LazyRuntime] wait_for_fragment_context_end",
                extra={
                    "fragment": fragment,
                    "component": component,
                    "dataset_hash": dataset_hash,
                    "hydrated": hydration_ready,
                    "context_ready": True,
                    "duration_ms": duration_ms,
                },
            )
            return True

    ready = bool(fragment_context_ready)
    duration_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "[LazyRuntime] wait_for_fragment_context_end",
        extra={
            "fragment": fragment,
            "component": component,
            "dataset_hash": dataset_hash,
            "hydrated": hydration_ready,
            "context_ready": ready,
            "duration_ms": duration_ms,
        },
    )
    if not ready:
        _trigger_fragment_context_rerun(dataset_hash)
    return ready


def _trigger_fragment_context_rerun(dataset_hash: str) -> None:
    token = dataset_hash or "__default__"
    if token in _FRAGMENT_CONTEXT_RERUN_DATASETS:
        return
    rerun = getattr(st, "experimental_rerun", None)
    if not callable(rerun):
        return
    _FRAGMENT_CONTEXT_RERUN_DATASETS.add(token)
    try:
        rerun()
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug(
            "No se pudo solicitar experimental_rerun para el fragmento %s",
            token,
            exc_info=True,
        )


def _record_fragment_visibility(*, component: str, dataset_hash: str, visible: bool) -> None:
    try:
        log_default_telemetry(
            phase="portfolio.fragment_visibility",
            dataset_hash=dataset_hash,
            extra={
                "lazy_loaded_component": component,
                "portfolio.fragment_visible": visible,
            },
        )
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug(
            "No se pudo registrar portfolio.fragment_visible para %s",
            component,
            exc_info=True,
        )
