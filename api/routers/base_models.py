"""Shared Pydantic compatibility helpers for API routers."""

from __future__ import annotations

from pydantic import BaseModel

try:  # pragma: no cover - compatibility across Pydantic versions
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover
    ConfigDict = None  # type: ignore[assignment]


class _BaseModel(BaseModel):
    """Base model tolerant to extra fields across Pydantic versions."""

    if ConfigDict is not None:  # pragma: no branch
        model_config = ConfigDict(extra="ignore")  # type: ignore[assignment]
    else:  # pragma: no cover - fallback for Pydantic v1
        class Config:  # type: ignore[override]
            extra = "ignore"


__all__ = ["_BaseModel"]
