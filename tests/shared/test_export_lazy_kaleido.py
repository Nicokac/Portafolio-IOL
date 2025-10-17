"""Tests for the deferred Kaleido loading behaviour in ``shared.export``."""

from __future__ import annotations

from types import SimpleNamespace

import plotly.graph_objects as go
import pytest

from shared import export


@pytest.fixture(autouse=True)
def _reset_kaleido_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure each test starts from a clean Kaleido runtime state."""

    monkeypatch.setattr(export, "_KALEIDO_IMPORTED", False)
    monkeypatch.setattr(export, "_KALEIDO_RUNTIME_AVAILABLE", None)
    monkeypatch.setattr(export, "_KALEIDO_WARNING_LAST_TS", None)


def test_fig_to_png_bytes_triggers_lazy_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first export lazily imports Kaleido and records telemetry."""

    calls: list[str] = []
    metrics: dict[str, float] = {}

    original_import = export.importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name == "kaleido":
            calls.append(name)
            return SimpleNamespace(__name__="kaleido")
        return original_import(name, package)

    class _Scope:
        def ensure_chrome(self) -> None:
            return None

    monkeypatch.setattr(export, "_KALEIDO_AVAILABLE", True)
    monkeypatch.setattr(export.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(export, "_get_kaleido_scope", lambda: _Scope())
    monkeypatch.setattr(export.pio, "to_image", lambda *_args, **_kwargs: b"png-bytes")
    monkeypatch.setattr(
        export,
        "record_kaleido_lazy_load",
        lambda duration_ms, completed_at=None: metrics.setdefault(
            "duration", float(duration_ms)
        ),
    )

    result = export.fig_to_png_bytes(go.Figure())

    assert result == b"png-bytes"
    assert calls == ["kaleido"]
    assert metrics.get("duration", 0.0) >= 0.0
    assert export._KALEIDO_IMPORTED is True
    assert export._KALEIDO_RUNTIME_AVAILABLE is True

    # Subsequent exports reuse the cached runtime without importing again.
    second = export.fig_to_png_bytes(go.Figure())
    assert second == b"png-bytes"
    assert calls == ["kaleido"]


def test_fig_to_png_bytes_returns_none_when_kaleido_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No import is attempted when Kaleido is not installed."""

    original_import = export.importlib.import_module

    def failing_import(name: str, package: str | None = None):  # pragma: no cover - defensive
        if name == "kaleido":
            pytest.fail("kaleido import should not be attempted when unavailable")
        return original_import(name, package)

    monkeypatch.setattr(export, "_KALEIDO_AVAILABLE", False)
    monkeypatch.setattr(export.importlib, "import_module", failing_import)

    result = export.fig_to_png_bytes(go.Figure())

    assert result is None
    assert export._KALEIDO_RUNTIME_AVAILABLE is False
