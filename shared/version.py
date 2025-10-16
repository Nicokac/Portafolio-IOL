"""Project version and release metadata."""
from __future__ import annotations

VERSION = "0.6.8"
RELEASE_NAME = "Streamlit 1.50 + Predictive optimization"
BUILD_DATE = "2025-10-17"
CHANGELOG_REF = (
    "Streamlit 1.50 sparklines, adaptive predictive batching with lock timeouts, and dashboard/stability improvements"
)

__version__ = VERSION
__codename__ = RELEASE_NAME
__release_date__ = BUILD_DATE
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
    "BUILD_DATE",
    "CHANGELOG_REF",
    "__version__",
    "__codename__",
    "__release_date__",
    "__changelog_ref__",
    "__stability__",
    "DEFAULT_VERSION",
    "get_version_info",
]
