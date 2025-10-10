"""Helpers to check for remote updates and trigger manual upgrades."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from typing import Any

import requests

REMOTE_VERSION_URL = (
    "https://raw.githubusercontent.com/Nicokac/portafolio-iol/main/shared/version.json"
)

logger = logging.getLogger(__name__)


def _get_local_version() -> str:
    from shared.version import __version__  # Imported lazily for testability

    return __version__


def check_for_update() -> str | None:
    """Return the remote version if a newer release is available."""

    try:
        local_version = _get_local_version()
        response = requests.get(REMOTE_VERSION_URL, timeout=3)
        response.raise_for_status()
        data: Any = response.json()
        remote_version = data.get("version") if isinstance(data, dict) else None
        if isinstance(remote_version, str):
            remote_version = remote_version.strip()
            if remote_version and remote_version != local_version:
                return remote_version
    except Exception:  # pragma: no cover - best-effort network call
        logger.debug("No se pudo obtener la versión remota", exc_info=True)
    return None


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

    return True


__all__ = ["check_for_update", "_run_update_script"]
