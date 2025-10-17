"""Performance-oriented regression tests for the Kaleido fallback renderer."""

from __future__ import annotations

import time

from shared import export


def test_browser_renderer_fallback_avoids_kaleido_delay(monkeypatch) -> None:
    """When Kaleido is unavailable, the fallback should return quickly using the browser renderer."""

    monkeypatch.setattr(export, "_KALEIDO_AVAILABLE", False)
    monkeypatch.setattr(export, "_KALEIDO_IMPORTED", False)
    monkeypatch.setattr(export, "_KALEIDO_RUNTIME_AVAILABLE", None)

    start = time.perf_counter()
    runtime_available = export.ensure_kaleido_runtime()
    elapsed = time.perf_counter() - start

    assert export.pio.renderers.default == "browser"
    assert runtime_available is False
    assert elapsed < 10.0
