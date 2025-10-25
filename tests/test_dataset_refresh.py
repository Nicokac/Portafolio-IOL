import sys
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from controllers.auth import login_flow
from services.data_fetch_service import DatasetMetadata, PortfolioDataset


class DummyFetchService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get_dataset(self, cli: Any, psvc: Any, *, force_refresh: bool = False):
        payload = {
            "cli": cli,
            "psvc": psvc,
            "force_refresh": force_refresh,
        }
        self.calls.append(payload)
        dataset = PortfolioDataset(
            positions=pd.DataFrame(),
            quotes={},
            all_symbols=(),
            available_types=(),
            dataset_hash="positions",
            raw_payload={"_cached": False},
            quotes_hash="quotes",
        )
        metadata = DatasetMetadata(
            source="live_endpoint",
            updated_at=0.0,
            stale=False,
            refresh_in_progress=False,
            refresh_count=1,
            duration=0.0,
            cache_hit=False,
            error=None,
            skip_invalidation=False,
        )
        return dataset, metadata


from types import SimpleNamespace
import sys


def test_force_refresh_after_login(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch_service = DummyFetchService()
    monkeypatch.setattr(login_flow, "get_portfolio_data_fetch_service", lambda: fetch_service)

    stub_module = SimpleNamespace(get_portfolio_service=lambda: object())
    monkeypatch.setitem(sys.modules, "controllers.portfolio.portfolio", stub_module)

    dummy_client = object()
    login_flow.force_portfolio_refresh_after_login(dummy_client)

    assert fetch_service.calls, "the fetch service should have been invoked"
    call = fetch_service.calls[-1]
    assert call["force_refresh"] is True
    assert call["cli"] is dummy_client


def test_quotes_hash_invalidation(monkeypatch: pytest.MonkeyPatch) -> None:
    from services import portfolio_view

    forced_value = 19_966_960.0
    df_positions = pd.DataFrame(
        {
            "simbolo": ["BPOC7"],
            "moneda": ["ARS"],
            "moneda_origen": ["ARS"],
            "cantidad": [146.0],
            "ultimoPrecio": [1_367.6],
            "valor_actual": [forced_value],
            "pricing_source": ["override_bopreal_forced"],
        }
    )
    controls = SimpleNamespace(
        selected_syms=[],
        selected_types=[],
        symbol_query="",
        refresh_secs=0,
    )

    apply_calls = {"count": 0}

    def _fake_apply_filters(df_pos, controls_unused, cli, psvc, *, dataset_hash=None, skip_invalidation=False):
        apply_calls["count"] += 1
        return df_pos.copy()

    monkeypatch.setattr(portfolio_view, "_apply_filters", _fake_apply_filters)
    monkeypatch.setattr(
        portfolio_view,
        "_compute_contribution_metrics",
        lambda df: portfolio_view.PortfolioContributionMetrics.empty(),
    )
    monkeypatch.setattr(
        portfolio_view,
        "calculate_totals",
        lambda df: portfolio_view.PortfolioTotals(100.0, 50.0, 50.0, 0.0, 0.0),
    )
    monkeypatch.setattr(
        portfolio_view.PortfolioViewModelService,
        "_update_history",
        lambda self, totals: pd.DataFrame(),
        raising=False,
    )

    service = portfolio_view.PortfolioViewModelService()
    base_hash = service._hash_dataset(df_positions)

    fingerprint_a = f"{base_hash}|quotes:hashA"
    fingerprint_b = f"{base_hash}|quotes:hashB"

    service.get_portfolio_view(
        df_positions, controls, cli=None, psvc=None, dataset_hash=fingerprint_a
    )
    snapshot_b = service.get_portfolio_view(
        df_positions, controls, cli=None, psvc=None, dataset_hash=fingerprint_b
    )

    assert apply_calls["count"] == 2
    assert service._dataset_key is not None
    assert "hashB" in service._dataset_key

    row = snapshot_b.df_view.iloc[0]
    assert row["valor_actual"] == pytest.approx(forced_value)
    assert row["pricing_source"] == "override_bopreal_forced"
