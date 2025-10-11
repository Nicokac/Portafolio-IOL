"""Helpers to check for remote updates and trigger manual upgrades."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import requests
from shared.version import __version__

REMOTE_VERSION_URL = (
    "https://raw.githubusercontent.com/Nicokac/portafolio-iol/main/shared/version.json"
)

logger = logging.getLogger(__name__)

_CHECK_FILE = os.path.join(
    tempfile.gettempdir(), "portafolio_iol_version_check.json"
)

LOG_FILE = os.path.join(tempfile.gettempdir(), "portafolio_iol_update_log.json")


def _log_event(event: str, version: str, status: str) -> None:
    entry = {
        "event": event,
        "version": version,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        logs: list[dict[str, Any]] = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        logs.append(entry)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(logs[-20:], f, indent=2)
    except Exception as exc:  # pragma: no cover - best effort persistence
        logging.warning("No se pudo registrar evento de actualización: %s", exc)


def get_update_history() -> list[dict[str, Any]]:
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def safe_restart_app() -> bool:
    """Reinicia la aplicación tras una actualización exitosa.

    Returns ``True`` si se inició el proceso de reinicio, ``False`` en caso
    contrario (por ejemplo, cuando está deshabilitado por configuración).
    """

    if os.environ.get("DISABLE_AUTO_RESTART") == "1":
        _log_event("restart", __version__, "skipped: disabled")
        return False

    try:
        python = sys.executable
        script = os.path.abspath(sys.argv[0])
        _log_event("restart", __version__, "initiated")
        time.sleep(1)
        subprocess.Popen([python, script], close_fds=True)
        _log_event("restart", __version__, "done")
        sys.exit(0)
    except Exception as exc:  # pragma: no cover - defensive logging
        _log_event("restart", __version__, f"failed: {exc}")
    return False


def save_last_check_time() -> None:
    try:
        with open(_CHECK_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_check": time.time()}, f)
    except Exception:  # pragma: no cover - best effort persistence
        pass


def get_last_check_time() -> float | None:
    try:
        with open(_CHECK_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("last_check")
    except Exception:
        return None


def format_last_check(ts: float | None) -> str:
    if not ts:
        return "Nunca"
    delta = time.time() - ts
    mins = int(delta // 60)
    if mins < 60:
        return f"hace {mins} min"
    hrs = mins // 60
    return f"hace {hrs} h {mins % 60} min"


def _get_local_version() -> str:
    from shared.version import __version__  # Imported lazily for testability

    return __version__


def check_for_update() -> str | None:
    """Return the remote version if a newer release is available."""

    latest: str | None = None
    try:
        local_version = _get_local_version()
        response = requests.get(REMOTE_VERSION_URL, timeout=3)
        response.raise_for_status()
        data: Any = response.json()
        remote_version = data.get("version") if isinstance(data, dict) else None
        if isinstance(remote_version, str):
            remote_version = remote_version.strip()
            if remote_version and remote_version != local_version:
                latest = remote_version
        _log_event("check", local_version, "ok")
    except Exception:  # pragma: no cover - best-effort network call
        logger.debug("No se pudo obtener la versión remota", exc_info=True)
    finally:
        save_last_check_time()
    return latest


def _is_streamlit_cloud() -> bool:
    runtime = (os.environ.get("STREAMLIT_RUNTIME") or "").lower()
    if runtime == "cloud":
        return True
    if os.environ.get("STREAMLIT_SHARING_MODE"):
        return True
    if os.environ.get("STREAMLIT_CLOUD"):
        return True
    if (os.environ.get("STREAMLIT_ENV") or "").lower() == "cloud":
        return True
    return False


def _has_shell_support() -> bool:
    if _is_streamlit_cloud():
        return False
    if shutil.which("git") is None:
        return False
    return True


def _run_update_script(latest_version: str) -> bool:
    """Execute the local update sequence when shell access is available."""

    import streamlit as st

    if not _has_shell_support():
        st.info(
            "Las actualizaciones automáticas no están disponibles en este entorno."
        )
        st.link_button(
            "Ver cambios", "https://github.com/Nicokac/portafolio-iol/blob/main/CHANGELOG.md"
        )
        return False

    _log_event("update", latest_version, "started")
    try:
        subprocess.run(["git", "pull"], check=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                "requirements.txt",
                "--upgrade",
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        st.error(
            "❌ Error al actualizar. Por favor, actualice manualmente desde el repositorio."
        )
        return False

    _log_event("update", latest_version, "done")
    return True


__all__ = [
    "check_for_update",
    "_run_update_script",
    "save_last_check_time",
    "get_last_check_time",
    "format_last_check",
    "get_update_history",
    "safe_restart_app",
]
