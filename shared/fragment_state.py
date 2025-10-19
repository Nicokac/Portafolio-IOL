"""State guardian for Streamlit lazy fragments.

This module persists the activation state of lazy fragments between reruns
so that components such as the portfolio table do not disappear after an
unexpected rerun.  It tracks the fragment toggles stored in ``st.session_state``
 and rehydrates them when necessary while keeping the behaviour compatible with
explicit user toggles.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, Tuple

try:  # pragma: no cover - optional dependency during tests
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - defensive import fallback
    st = None  # type: ignore

from infrastructure.iol.auth import get_current_user_id
from shared.telemetry import is_hydration_locked
from shared.user_actions import log_user_action

logger = logging.getLogger(__name__)

_STATE_KEY = "__fragment_state_guardian__"
_PERSIST_PENDING_KEY = "__fragment_state_pending_restore__"
_SENTINEL = object()
_PERSIST_DIR = Path.home() / ".iol_state"
_PERSIST_PATH = _PERSIST_DIR / ".fragment_state.json"
_PERSIST_VERSION = 1
_PERSIST_LOCK = threading.Lock()


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
        self._pending_restore: Dict[str, Any] | None = self._load_pending_restore()
        self._pending_restore_consumed = False
        self._lazy_signature = _compute_lazy_modules_signature()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def begin_cycle(self, dataset_hash: str | None) -> None:
        """Record the dataset hash associated with the current render cycle."""

        self._cycle_dataset = str(dataset_hash or "")
        if is_hydration_locked():
            logger.debug("Hydration locked; deferring fragment restore for %s", self._cycle_dataset)
            return
        self._maybe_restore_from_persistence()

    def prepare_persistent_restore(self) -> None:
        """Load persisted state for the current user into session memory."""

        if is_hydration_locked():
            logger.debug("Hydration locked; skipping persistent fragment restore preload")
            self._clear_pending_restore()
            return
        payload = self._load_persisted_snapshot()
        if payload is None:
            self._clear_pending_restore()
            return
        self._store_pending_restore(payload)
        self._pending_restore = payload
        self._pending_restore_consumed = False

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
        if should_rehydrate and not is_hydration_locked():
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

    def soft_refresh(self, dataset_hash: str | None = None) -> None:
        """Record a dataset refresh that did not trigger a rerun."""

        dataset_token = str(dataset_hash or self._resolve_dataset_hash() or "")
        if not dataset_token:
            return
        now = time.time()
        refreshed: list[str] = []
        for key, entry in list(self._registry.items()):
            if not isinstance(entry, dict):
                continue
            if entry.get("dataset_hash") != dataset_token:
                continue
            entry.setdefault("active", True)
            entry.setdefault("was_loaded", True)
            entry["last_value"] = True
            entry["last_seen"] = now
            entry.pop("pending_restore", None)
            refreshed.append(key)
        if refreshed:
            detail = {"dataset_hash": dataset_token, "fragments": sorted(refreshed)}
            logger.info("[Guardian] Soft refresh: dataset revalidated without rerun", extra=detail)

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

    def persist_to_disk(self) -> bool:
        """Persist the current active fragments to a JSON snapshot."""

        user_id = _resolve_current_user_id()
        if not user_id:
            return False
        dataset_hash = self._resolve_dataset_hash()
        if not dataset_hash:
            return False
        fragments_payload, fragment_keys = self._collect_active_fragments(dataset_hash)
        with _PERSIST_LOCK:
            store = _read_persisted_store()
            users = store.setdefault("users", {})
            if not fragments_payload:
                if user_id in users:
                    users.pop(user_id, None)
                    if not _write_persisted_store(store):
                        return False
                return False
            users[user_id] = {
                "lazy_signature": self._lazy_signature,
                "datasets": {
                    dataset_hash: {
                        "updated_at": time.time(),
                        "fragments": fragments_payload,
                    }
                },
            }
            if not _write_persisted_store(store):
                return False
        detail = {
            "user_id": user_id,
            "dataset_hash": dataset_hash,
            "fragments": fragment_keys,
        }
        log_user_action("fragment_state_saved", detail, dataset_hash=dataset_hash)
        return True

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load_pending_restore(self) -> Dict[str, Any] | None:
        if st is None:
            return None
        state = getattr(st, "session_state", None)
        if state is None:
            return None
        try:
            pending = state.get(_PERSIST_PENDING_KEY)  # type: ignore[attr-defined]
        except Exception:
            pending = None
        return pending if isinstance(pending, dict) else None

    def _store_pending_restore(self, payload: Dict[str, Any]) -> None:
        state = self._get_session_state()
        if state is None:
            return
        try:
            state[_PERSIST_PENDING_KEY] = json.loads(json.dumps(payload))  # shallow copy
        except Exception:
            logger.debug("No se pudo almacenar el estado persistente pendiente", exc_info=True)

    def _clear_pending_restore(self) -> None:
        state = self._get_session_state()
        if state is not None:
            try:
                state.pop(_PERSIST_PENDING_KEY, None)
            except Exception:
                logger.debug("No se pudo limpiar el estado persistente pendiente", exc_info=True)
        self._pending_restore = None
        self._pending_restore_consumed = True

    def _get_session_state(self) -> Dict[str, Any] | None:
        if st is None:
            return None
        state = getattr(st, "session_state", None)
        if state is None:
            return None
        return state

    def _load_persisted_snapshot(self) -> Dict[str, Any] | None:
        user_id = _resolve_current_user_id()
        if not user_id:
            return None
        store = _read_persisted_store()
        users = store.get("users")
        if not isinstance(users, dict):
            return None
        entry = users.get(user_id)
        if not isinstance(entry, dict):
            return None
        signature = str(entry.get("lazy_signature") or "")
        if signature and signature != self._lazy_signature:
            _purge_user_persisted_state(user_id)
            return None
        datasets = entry.get("datasets")
        if not isinstance(datasets, dict):
            return None
        normalized: Dict[str, Any] = {}
        for dataset_hash, dataset_payload in datasets.items():
            if not isinstance(dataset_hash, str) or not dataset_hash:
                continue
            if not isinstance(dataset_payload, dict):
                continue
            fragments = dataset_payload.get("fragments")
            if not isinstance(fragments, dict) or not fragments:
                continue
            normalized[dataset_hash] = {
                "fragments": {
                    str(key): value
                    for key, value in fragments.items()
                    if isinstance(value, dict)
                }
            }
        if not normalized:
            return None
        return {
            "user_id": user_id,
            "lazy_signature": signature or self._lazy_signature,
            "datasets": normalized,
        }

    def _maybe_restore_from_persistence(self) -> None:
        if self._pending_restore_consumed:
            return
        dataset_hash = self._resolve_dataset_hash()
        if not dataset_hash:
            return
        if self._pending_restore is None:
            self._pending_restore = self._load_pending_restore()
            if self._pending_restore is None:
                return
        pending = self._pending_restore
        user_id = str(pending.get("user_id") or "")
        signature = str(pending.get("lazy_signature") or "")
        if signature and signature != self._lazy_signature:
            _purge_user_persisted_state(user_id)
            self._clear_pending_restore()
            return
        datasets = pending.get("datasets")
        if not isinstance(datasets, dict):
            self._clear_pending_restore()
            return
        dataset_payload = datasets.get(dataset_hash)
        if not isinstance(dataset_payload, dict):
            if user_id:
                _purge_user_persisted_state(user_id)
            self._clear_pending_restore()
            return
        fragments = dataset_payload.get("fragments")
        if not isinstance(fragments, dict) or not fragments:
            if user_id:
                _purge_user_persisted_state(user_id)
            self._clear_pending_restore()
            return
        restored: list[str] = []
        for key, fragment_state in fragments.items():
            if not isinstance(fragment_state, dict):
                continue
            entry = self._registry.get(key)
            current = entry if isinstance(entry, dict) else {}
            if entry is None or entry.get("dataset_hash") != dataset_hash:
                entry = self._ensure_entry(
                    key,
                    session_key=current.get("session_key"),
                    dataset_hash=dataset_hash,
                    component=current.get("component"),
                    scope=current.get("scope"),
                    fallback_key=current.get("fallback_key"),
                )
            entry["active"] = True
            entry["dismissed"] = False
            entry["was_loaded"] = bool(fragment_state.get("was_loaded", True))
            entry["last_value"] = bool(fragment_state.get("last_value", True))
            entry.setdefault("pending_restore", True)
            entry.setdefault("restored_from_persistence", True)
            entry.setdefault("last_seen", time.time())
            restored.append(key)
        if restored:
            detail = {
                "user_id": user_id or _resolve_current_user_id() or "",
                "dataset_hash": dataset_hash,
                "fragments": sorted(restored),
            }
            log_user_action("fragment_state_restored", detail, dataset_hash=dataset_hash)
        self._clear_pending_restore()

    def _collect_active_fragments(self, dataset_hash: str) -> Tuple[Dict[str, Dict[str, Any]], list[str]]:
        fragments: Dict[str, Dict[str, Any]] = {}
        restored_keys: list[str] = []
        for key, entry in list(self._registry.items()):
            if not isinstance(entry, dict):
                continue
            if entry.get("dataset_hash") != dataset_hash:
                continue
            if not entry.get("active") or entry.get("dismissed"):
                continue
            fragments[key] = {
                "was_loaded": bool(entry.get("was_loaded") or entry.get("last_value")),
                "last_value": bool(entry.get("last_value", True)),
                "last_seen": float(entry.get("last_seen") or time.time()),
            }
            restored_keys.append(key)
        return fragments, sorted(restored_keys)

    def _resolve_dataset_hash(self) -> str:
        if self._cycle_dataset:
            return self._cycle_dataset
        state = self._get_session_state()
        if state is None:
            return ""
        try:
            candidate = state.get("dataset_hash")  # type: ignore[attr-defined]
        except Exception:
            candidate = None
        if isinstance(candidate, str) and candidate:
            return candidate
        return ""


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
        state.pop(_PERSIST_PENDING_KEY, None)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudo reiniciar el registro de fragmentos", exc_info=True)


def fragment_state_soft_refresh(dataset_hash: str | None = None) -> None:
    """Public helper to record a soft refresh in the active guardian."""

    guardian = get_fragment_state_guardian()
    try:
        guardian.soft_refresh(dataset_hash)
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudo registrar el soft refresh del guardian", exc_info=True)


def prepare_persistent_fragment_restore() -> None:
    """Best-effort helper to queue a persisted restore for the active user."""

    if is_hydration_locked():
        logger.debug("Hydration locked; skipping top-level fragment restore preparation")
        return
    guardian = get_fragment_state_guardian()
    guardian.prepare_persistent_restore()


def persist_fragment_state_snapshot() -> bool:
    """Persist the current guardian state to disk if possible."""

    guardian = get_fragment_state_guardian()
    try:
        return guardian.persist_to_disk()
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudo persistir el estado de fragmentos", exc_info=True)
        return False


def _compute_lazy_modules_signature() -> str:
    modules = ["ui.lazy.table_fragment", "ui.lazy.charts_fragment"]
    signature_parts: list[str] = []
    for name in modules:
        try:
            module = sys.modules.get(name) or importlib.import_module(name)
        except Exception:
            continue
        path = getattr(module, "__file__", None)
        mtime = 0.0
        if path:
            try:
                mtime = Path(path).stat().st_mtime
            except OSError:
                mtime = 0.0
        signature_parts.append(f"{name}:{mtime:.0f}")
    return "|".join(sorted(signature_parts))


def _resolve_current_user_id() -> str | None:
    try:
        user_id = get_current_user_id()
    except Exception:  # pragma: no cover - defensive fallback
        logger.debug("No se pudo resolver el usuario actual", exc_info=True)
        user_id = None
    if user_id:
        return str(user_id)
    state = getattr(st, "session_state", None)
    if state is None:
        return None
    try:
        candidate = state.get("last_user_id")  # type: ignore[attr-defined]
    except Exception:
        candidate = None
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return None


def _read_persisted_store() -> Dict[str, Any]:
    if not _PERSIST_PATH.exists():
        return {"version": _PERSIST_VERSION, "users": {}}
    try:
        raw = _PERSIST_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("No se pudo leer el estado persistente de fragmentos: %s", exc)
        return {"version": _PERSIST_VERSION, "users": {}}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Estado de fragmentos corrupto; se descartará: %s", exc)
        return {"version": _PERSIST_VERSION, "users": {}}
    if not isinstance(data, dict):
        return {"version": _PERSIST_VERSION, "users": {}}
    users = data.get("users")
    if not isinstance(users, dict):
        users = {}
    data["users"] = users
    data["version"] = _PERSIST_VERSION
    return data


def _write_persisted_store(data: Dict[str, Any]) -> bool:
    try:
        _PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("No se pudo crear el directorio de estado persistente: %s", exc)
        return False
    tmp_path = _PERSIST_PATH.with_suffix(".tmp")
    try:
        serialized = json.dumps(data, ensure_ascii=False, sort_keys=True)
        tmp_path.write_text(serialized, encoding="utf-8")
        tmp_path.replace(_PERSIST_PATH)
    except OSError as exc:
        logger.warning("No se pudo guardar el estado persistente de fragmentos: %s", exc)
        return False
    return True


def _purge_user_persisted_state(user_id: str) -> None:
    if not user_id:
        return
    with _PERSIST_LOCK:
        store = _read_persisted_store()
        users = store.get("users")
        if not isinstance(users, dict) or user_id not in users:
            return
        users.pop(user_id, None)
        _write_persisted_store(store)


__all__ = [
    "FragmentGuardResult",
    "FragmentStateGuardian",
    "fragment_state_soft_refresh",
    "get_fragment_state_guardian",
    "persist_fragment_state_snapshot",
    "prepare_persistent_fragment_restore",
    "reset_fragment_state_guardian",
]
