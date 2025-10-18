import csv

import pytest

import controllers.portfolio.portfolio as portfolio_mod
from shared.cache import visual_cache_registry
from tests.ui.test_portfolio_ui import FakeStreamlit


def test_visual_cache_debug_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake_st = FakeStreamlit(radio_sequence=[0])
    monkeypatch.setattr(portfolio_mod, "st", fake_st)
    monkeypatch.setenv("ST_DEBUG_CACHE", "1")

    visual_cache_registry.reset()
    log_path = tmp_path / "visual_cache_invalidations.csv"
    monkeypatch.setattr(visual_cache_registry, "_log_path", log_path, raising=False)

    visual_cache_registry.record("summary", dataset_hash="alpha", reused=False)
    visual_cache_registry.record("summary", dataset_hash="alpha", reused=True)

    portfolio_mod._publish_visual_cache_debug()

    debug_state = fake_st.session_state.get("portfolio_visual_cache_debug")
    assert isinstance(debug_state, dict)
    summary_entry = debug_state["entries"].get("summary")
    assert summary_entry is not None
    assert summary_entry["hits"] == 1
    assert summary_entry["misses"] == 1

    visual_cache_registry.invalidate_dataset("alpha", reason="dataset_hash_changed")
    assert log_path.exists()
    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["reason"] == "dataset_hash_changed" for row in rows)

    visual_cache_registry.reset()
