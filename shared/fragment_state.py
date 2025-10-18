"""State guardian for Streamlit lazy fragments.

This module persists the activation state of lazy fragments between reruns
so that components such as the portfolio table do not disappear after an
unexpected rerun.  It tracks the fragment toggles stored in ``st.session_state``
 and rehydrates them when necessary while keeping the behaviour compatible with
explicit user toggles.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, Tuple

try:  # pragma: no cover - optional dependency during tests
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - defensive import fallback
    st = None  # type: ignore

from shared.user_actions import log_user_action

logger = logging.getLogger(__name__)

_STATE_KEY = "__fragment_state_guardian__"
_SENTINEL = object()


@dataclass(frozen=True)
class FragmentGuardResult:
    """Outcome of the guardian evaluation for a lazy block."""

    rehydrated: bool
    explicit_hide: bool


class FragmentStateGuardian:
    """Persisted registry protecting fragment toggles between reruns."""

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Any]] = self._ensure_registry()
        self._cycle_dataset: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def begin_cycle(self, dataset_hash: str | None) -> None:
        """Record the dataset hash associated with the current render cycle."""

        self._cycle_dataset = str(dataset_hash or "")

    def maybe_rehydrate(
        self,
        *,
        key: str,
        session_key: str | None,
        dataset_hash: str | None,
        component: str | None,
        scope: str | None,
        was_loaded: bool,
        fallback_key: str | None = None,
    ) -> FragmentGuardResult:
        """Ensure that a previously loaded fragment is still visible."""

        entry = self._ensure_entry(
            key,
            session_key=session_key,
            dataset_hash=dataset_hash,
            component=component,
            scope=scope,
            fallback_key=fallback_key,
        )
        entry["last_seen"] = time.time()
        entry["was_loaded"] = bool(was_loaded)

        key_exists, key_value = self._read_flag(key)
        session_exists, session_value = self._read_flag(session_key)
        explicit_hide = bool(session_exists and not session_value)

        entry.setdefault("last_value", bool(key_value or session_value))

        dataset_token = entry.get("dataset_hash", "")
        missing_key = not key_exists
        missing_session = not session_exists
        should_rehydrate = (
            entry.get("active")
            and entry.get("was_loaded")
            and (missing_key or missing_session)
            and not entry.get("dismissed")
            and not explicit_hide
        )

        rehydrated = False
        if should_rehydrate:
            logger.debug(
                "Fragment guardian rehydrating %s (component=%s, scope=%s)",
                key,
                entry.get("component"),
                entry.get("scope"),
            )
            dataset_token = str(dataset_hash or dataset_token or "")
            self._set_flag(key, True)
            if session_key and session_key != key:
                self._set_flag(session_key, True)
            if fallback_key and fallback_key not in (key, session_key):
                self._set_flag(fallback_key, True)
            entry["last_value"] = True
            entry["rehydrated_at"] = time.time()
            detail = {
                "key": key,
                "component": entry.get("component", ""),
                "session_key": session_key or key,
            }
            if fallback_key:
                detail["fallback_key"] = fallback_key
            log_user_action("lazy_block_rehydrated", detail, dataset_hash=dataset_token)
            rehydrated = True

        return FragmentGuardResult(rehydrated=rehydrated, explicit_hide=explicit_hide)

    def mark_ready(
        self,
        *,
        key: str,
        session_key: str | None,
        dataset_hash: str | None,
        component: str | None,
        scope: str | None,
        fallback_key: str | None = None,
    ) -> None:
        """Persist that a fragment is actively rendered."""

        entry = self._ensure_entry(
            key,
            session_key=session_key,
            dataset_hash=dataset_hash,
            component=component,
            scope=scope,
            fallback_key=fallback_key,
        )
        entry["active"] = True
        entry["dismissed"] = False
        entry["last_value"] = True
        entry["last_seen"] = time.time()

    def mark_not_ready(
        self,
        *,
        key: str,
        session_key: str | None,
        dataset_hash: str | None,
        explicit_hide: bool,
    ) -> None:
        """Update guardian state when the fragment is not rendered."""

        entry = self._registry.get(key)
        if entry is None or entry.get("dataset_hash") != str(dataset_hash or entry.get("dataset_hash") or ""):
            entry = self._ensure_entry(
                key,
                session_key=session_key,
                dataset_hash=dataset_hash,
                component=None,
                scope=None,
                fallback_key=None,
            )
        entry["last_seen"] = time.time()
        entry["last_value"] = False
        if explicit_hide:
            entry["active"] = False
            entry["dismissed"] = True
        else:
            # Keep tracking the fragment as active so we can rehydrate later.
            if entry.get("active"):
                entry.setdefault("pending_restore", True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_registry(self) -> Dict[str, Dict[str, Any]]:
        if st is None:
            return {}
        state = getattr(st, "session_state", None)
        if state is None:
            return {}
        try:
            registry = state.get(_STATE_KEY)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - defensive safeguard
            registry = None
        if not isinstance(registry, dict):
            registry = {}
            try:
                state[_STATE_KEY] = registry  # type: ignore[index]
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug("No se pudo inicializar el registro de fragmentos", exc_info=True)
        return registry

    def _ensure_entry(
        self,
        key: str,
        *,
        session_key: str | None,
        dataset_hash: str | None,
        component: str | None,
        scope: str | None,
        fallback_key: str | None,
    ) -> Dict[str, Any]:
        dataset_token = str(dataset_hash or "")
        entry = self._registry.get(key)
        if not isinstance(entry, dict) or entry.get("dataset_hash") != dataset_token:
            entry = {
                "dataset_hash": dataset_token,
                "active": False,
                "dismissed": False,
                "last_value": False,
            }
            self._registry[key] = entry
        entry["session_key"] = session_key
        entry["component"] = component or ""
        entry["scope"] = scope or "global"
        entry["fallback_key"] = fallback_key
        return entry

    def _read_flag(self, name: str | None) -> Tuple[bool, bool]:
        if not name or st is None:
            return False, False
        state = getattr(st, "session_state", None)
        if state is None:
            return False, False
        sentinel = _SENTINEL
        try:
            value = state.get(name, sentinel)  # type: ignore[attr-defined]
        except Exception:
            try:
                value = state[name]  # type: ignore[index]
            except Exception:
                value = sentinel
        if value is sentinel:
            return False, False
        return True, bool(value)

    def _set_flag(self, name: str | None, value: bool) -> None:
        if not name or st is None:
            return
        state = getattr(st, "session_state", None)
        if state is None:
            return
        try:
            state[name] = bool(value)  # type: ignore[index]
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug("No se pudo actualizar la bandera %s", name, exc_info=True)


def get_fragment_state_guardian() -> FragmentStateGuardian:
    """Return a guardian bound to the current Streamlit session."""

    return FragmentStateGuardian()


def reset_fragment_state_guardian() -> None:
    """Utility for tests to clear the guardian registry."""

    if st is None:
        return
    state = getattr(st, "session_state", None)
    if state is None:
        return
    try:
        state.pop(_STATE_KEY, None)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudo reiniciar el registro de fragmentos", exc_info=True)


__all__ = [
    "FragmentGuardResult",
    "FragmentStateGuardian",
    "get_fragment_state_guardian",
    "reset_fragment_state_guardian",
]
