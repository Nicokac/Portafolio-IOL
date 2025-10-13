"""Project version and release metadata."""
from __future__ import annotations

__version__ = "v0.6.6-patch9b2"
__codename__ = "Optimization Nexus"
__release_date__ = "2025-10-13"
__changelog_ref__ = "Reduced predictive and quotes latency under 10s total render time"
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
    "__version__",
    "__codename__",
    "__release_date__",
    "__changelog_ref__",
    "__stability__",
    "DEFAULT_VERSION",
    "get_version_info",
]
