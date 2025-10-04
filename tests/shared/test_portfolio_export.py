"""Tests for shared export helpers."""
from __future__ import annotations

import base64
from pathlib import Path
import sys

import plotly.graph_objects as go
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared.export import fig_to_png_bytes


_DUMMY_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
)


def test_fig_to_png_bytes_uses_plotly_without_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    fig = go.Figure()
    called: dict[str, object] = {}

    def fake_get_scope() -> object:
        called["scope_checked"] = True
        return object()

    def fake_to_image(fig_arg: go.Figure, *, format: str = "png", **kwargs) -> bytes:
        called["fig"] = fig_arg
        called["kwargs"] = kwargs
        assert format == "png"
        if "scope" in kwargs:
            raise TypeError("scope argument is not supported")
        return _DUMMY_PNG

    monkeypatch.setattr("shared.export._get_kaleido_scope", fake_get_scope)
    monkeypatch.setattr("shared.export.pio.to_image", fake_to_image)

    result = fig_to_png_bytes(fig)

    assert result == _DUMMY_PNG
    assert called["fig"] is fig
    assert called.get("scope_checked") is True
    assert "scope" not in called.get("kwargs", {})


def test_fig_to_png_bytes_raises_value_error_on_engine_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    fig = go.Figure()

    def fake_to_image(*_args, **_kwargs) -> bytes:
        raise RuntimeError("engine missing")

    monkeypatch.setattr("shared.export._get_kaleido_scope", lambda: None)
    monkeypatch.setattr("shared.export.pio.to_image", fake_to_image)

    with pytest.raises(ValueError):
        fig_to_png_bytes(fig)
