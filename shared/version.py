"""Project version and release metadata."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

VERSION = "0.8.9.1"
RELEASE_NAME = "Portafolio IOL v0.8.9.1"
RELEASE_DATE = "2026-05-08"
CHANGELOG_REF = ("Hotfix 6.1.1 — Fix TypeError deepcopy (attrs)",)


def _resolve_build_signature() -> str:
    env_signature = os.getenv("PORTAFOLIO_BUILD_SIGNATURE")
    if env_signature:
        return env_signature

    project_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        result = ""

    if result:
        return result

    timestamp = datetime.now(timezone.utc).strftime("ts-%Y%m%d%H%M%S")
    return timestamp


BUILD_SIGNATURE = _resolve_build_signature()

__version__ = VERSION
__codename__ = RELEASE_NAME
__release_date__ = RELEASE_DATE
__build_signature__ = BUILD_SIGNATURE
__changelog_ref__ = CHANGELOG_REF
__stability__ = "stable"
# Keep in sync with ``pyproject.toml``'s ``project.version``.
DEFAULT_VERSION = __version__


def get_version_info() -> dict[str, str]:
    """Return the version metadata for consumers."""

    return {
        "version": __version__,
        "codename": __codename__,
        "release_date": __release_date__,
        "build_signature": __build_signature__,
        "changelog_ref": __changelog_ref__,
        "stability": __stability__,
    }


__all__ = [
    "VERSION",
    "RELEASE_NAME",
    "RELEASE_DATE",
    "BUILD_SIGNATURE",
    "CHANGELOG_REF",
    "__version__",
    "__codename__",
    "__release_date__",
    "__build_signature__",
    "__changelog_ref__",
    "__stability__",
    "DEFAULT_VERSION",
    "get_version_info",
]
