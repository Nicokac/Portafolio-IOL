"""FastAPI entrypoint package for Portafolio IOL."""

from shared.version import __build_signature__ as _APP_BUILD_SIGNATURE
from shared.version import __version__ as _APP_VERSION

__version__ = _APP_VERSION
__build_signature__ = _APP_BUILD_SIGNATURE

__all__ = [
    "main",
    "__version__",
    "__build_signature__",
]
