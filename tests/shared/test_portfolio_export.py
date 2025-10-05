"""Tests for shared export helpers."""
from __future__ import annotations

import base64
import logging
from pathlib import Path
import sys

import plotly.graph_objects as go
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from shared import export


_DUMMY_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
)


def _reset_runtime_state() -> None:
    export._KALEIDO_RUNTIME_AVAILABLE = None
    export._KALEIDO_WARNING_EMITTED = False


def test_fig_to_png_bytes_returns_png_when_runtime_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_runtime_state()

    fig = go.Figure()
    called: dict[str, object] = {}

    class _Scope:
        def ensure_chrome(self) -> None:
            called["ensure_chrome"] = True

    def fake_get_scope() -> object:
        called["scope_checked"] = True
        return _Scope()

    def fake_to_image(fig_arg: go.Figure, *, format: str = "png", **kwargs) -> bytes:
        called["fig"] = fig_arg
        assert format == "png"
        return _DUMMY_PNG

    monkeypatch.setattr(export, "_get_kaleido_scope", fake_get_scope)
    monkeypatch.setattr(export.pio, "to_image", fake_to_image)

    result = export.fig_to_png_bytes(fig)

    assert result == _DUMMY_PNG
    assert called["fig"] is fig
    assert called.get("scope_checked") is True
    assert called.get("ensure_chrome") is True
    assert export._KALEIDO_RUNTIME_AVAILABLE is True


def test_fig_to_png_bytes_returns_none_and_warns_when_runtime_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _reset_runtime_state()
    caplog.set_level(logging.WARNING, logger=export.logger.name)

    class _Scope:
        def ensure_chrome(self) -> None:
            raise OSError("chromium not found")

    monkeypatch.setattr(export, "_get_kaleido_scope", lambda: _Scope())
    monkeypatch.setattr(export.pio, "to_image", lambda *_args, **_kwargs: _DUMMY_PNG)

    result = export.fig_to_png_bytes(go.Figure())

    assert result is None
    assert export._KALEIDO_RUNTIME_AVAILABLE is False
    assert any("Exportación a PNG deshabilitada" in message for message in caplog.messages)


def test_fig_to_png_bytes_short_circuits_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_runtime_state()

    class _Scope:
        def ensure_chrome(self) -> None:
            raise RuntimeError("unexpected failure")

    monkeypatch.setattr(export, "_get_kaleido_scope", lambda: _Scope())
    monkeypatch.setattr(export.pio, "to_image", lambda *_args, **_kwargs: _DUMMY_PNG)

    assert export.fig_to_png_bytes(go.Figure()) is None

    # Con el runtime marcado como no disponible, no se debe intentar exportar nuevamente.
    monkeypatch.setattr(export.pio, "to_image", pytest.fail)
    assert export.fig_to_png_bytes(go.Figure()) is None
