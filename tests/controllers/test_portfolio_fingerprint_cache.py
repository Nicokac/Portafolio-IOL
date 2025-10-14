from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.portfolio import portfolio


@pytest.fixture(autouse=True)
def setup_session(monkeypatch: pytest.MonkeyPatch):
    """Provide an isolated Streamlit session state for fingerprint tests."""

    state: dict[str, object] = {}
    fake_st = SimpleNamespace(session_state=state)
    monkeypatch.setattr(portfolio, "st", fake_st)
    monkeypatch.setattr(portfolio, "log_performance_stage", lambda *args, **kwargs: None)

    cache = portfolio._get_dataset_fingerprint_cache()
    cache.clear()
    stats = portfolio._get_dataset_fingerprint_stats()
    stats.clear()
    stats.update(
        {
            "hits": 0,
            "misses": 0,
            "hit_ratio": 0.0,
            "last_status": None,
            "last_latency_ms": 0.0,
            "last_key": None,
        }
    )
    state.pop("portfolio_fingerprint_cache_stats", None)
    yield state
    cache.clear()
    stats.clear()
    state.clear()


def _build_viewmodel(*, controls: object | None = None) -> SimpleNamespace:
    controls = controls or SimpleNamespace(
        hide_cash=False,
        selected_syms=[],
        selected_types=[],
        symbol_query="",
        order_by="pl",
        desc=True,
        show_usd=True,
        top_n=10,
    )
    return SimpleNamespace(
        snapshot_id=None,
        controls=controls,
        totals=SimpleNamespace(total_value=0, total_cost=0, total_pl=0, total_pl_pct=0),
        historical_total=pd.DataFrame(),
        contributions=None,
    )


def test_dataset_fingerprint_memoized_per_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = {"calls": 0}

    def fake_compute(viewmodel, df_view):
        counter["calls"] += 1
        return f"fingerprint-{counter['calls']}"

    monkeypatch.setattr(portfolio, "_compute_portfolio_dataset_key", fake_compute)

    df_view = pd.DataFrame({"valor": range(5000)})
    viewmodel = _build_viewmodel()

    first = portfolio._portfolio_dataset_key(viewmodel, df_view)
    second = portfolio._portfolio_dataset_key(viewmodel, df_view)

    assert first == second == "fingerprint-1"
    assert counter["calls"] == 1

    stats = portfolio.st.session_state.get("portfolio_fingerprint_cache_stats", {})
    assert stats.get("misses") == 1
    assert stats.get("hits") == 1


def test_dataset_fingerprint_changes_with_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    counter = {"calls": 0}

    def fake_compute(viewmodel, df_view):
        counter["calls"] += 1
        return f"fingerprint-{counter['calls']}"

    monkeypatch.setattr(portfolio, "_compute_portfolio_dataset_key", fake_compute)

    df_view = pd.DataFrame({"valor": range(6000)})
    controls = SimpleNamespace(
        hide_cash=False,
        selected_syms=[],
        selected_types=[],
        symbol_query="",
        order_by="pl",
        desc=True,
        show_usd=True,
        top_n=10,
    )
    viewmodel = _build_viewmodel(controls=controls)

    key_a = portfolio._portfolio_dataset_key(viewmodel, df_view)
    controls.hide_cash = True
    key_b = portfolio._portfolio_dataset_key(viewmodel, df_view)

    assert key_a != key_b
    assert counter["calls"] == 2


def test_dataset_fingerprint_cache_improves_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def slow_compute(viewmodel, df_view):
        calls["count"] += 1
        time.sleep(0.01)
        return "slow-fingerprint"

    monkeypatch.setattr(portfolio, "_compute_portfolio_dataset_key", slow_compute)

    df_view = pd.DataFrame({"valor": range(8000)})
    viewmodel = _build_viewmodel()

    start = time.perf_counter()
    first = portfolio._portfolio_dataset_key(viewmodel, df_view)
    first_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    second = portfolio._portfolio_dataset_key(viewmodel, df_view)
    second_elapsed = time.perf_counter() - start

    assert first == second == "slow-fingerprint"
    assert calls["count"] == 1
    assert first_elapsed > 0.009
    assert second_elapsed < first_elapsed
    assert (first_elapsed - second_elapsed) > 0.005

