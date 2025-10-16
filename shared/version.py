"""Project version and release metadata."""
from __future__ import annotations

VERSION = "0.6.11"
RELEASE_NAME = "Portafolio IOL v0.6.11"
RELEASE_DATE = "2025-10-29"
CHANGELOG_REF = (
    "Optimización de rendimiento y diagnóstico avanzado con nuevas métricas de caché, telemetría y cobertura de pruebas"
)

__version__ = VERSION
__codename__ = RELEASE_NAME
__release_date__ = RELEASE_DATE
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
        "changelog_ref": __changelog_ref__,
        "stability": __stability__,
    }


__all__ = [
    "VERSION",
    "RELEASE_NAME",
    "RELEASE_DATE",
    "CHANGELOG_REF",
    "__version__",
    "__codename__",
    "__release_date__",
    "__changelog_ref__",
    "__stability__",
    "DEFAULT_VERSION",
    "get_version_info",
]
