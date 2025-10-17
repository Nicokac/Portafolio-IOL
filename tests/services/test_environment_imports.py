"""Regression tests for lazy Kaleido handling inside ``services.environment``."""

from __future__ import annotations

import csv
import logging
import threading
from pathlib import Path

import pytest

from services import environment


@pytest.fixture(autouse=True)
def _reset_environment_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset lazy-load flags between tests to avoid interference."""

    monkeypatch.setattr(environment, "_PORTFOLIO_RENDER_COMPLETED_AT", None)
    monkeypatch.setattr(environment, "_KALEIDO_LAZY_RECORDED", False)


def test_record_kaleido_lazy_load_writes_metrics(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Recording the lazy load should persist metrics and emit an info log."""

    metrics_path = tmp_path / "kaleido.csv"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(environment, "_KALEIDO_METRICS_PATH", metrics_path)
    monkeypatch.setattr(environment, "_KALEIDO_METRICS_FIELDS", ("kaleido_load_ms",))
    monkeypatch.setattr(environment, "_KALEIDO_LOCK", threading.Lock())

    try:
        environment.mark_portfolio_ui_render_complete(timestamp=100.0)
        caplog.set_level(logging.INFO, logger=environment.logger.name)
        environment.record_kaleido_lazy_load(1234.0, completed_at=105.0)
    finally:
        monkeypatch.undo()

    assert metrics_path.exists()
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["kaleido_load_ms"]
    assert rows[1] == ["1234.00"]
    assert "Kaleido initialized lazily after UI render" in " ".join(caplog.messages)


def test_mark_portfolio_render_complete_is_idempotent() -> None:
    """Only the first invocation should update the stored timestamp."""

    environment.mark_portfolio_ui_render_complete(timestamp=200.0)
    first = environment._PORTFOLIO_RENDER_COMPLETED_AT
    environment.mark_portfolio_ui_render_complete(timestamp=150.0)
    assert environment._PORTFOLIO_RENDER_COMPLETED_AT == first
