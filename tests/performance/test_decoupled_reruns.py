from types import SimpleNamespace

import pandas as pd

from controllers.portfolio import filters as filters_mod
from services.data_fetch_service import (
    PortfolioDataFetchService,
    PortfolioDataset,
    _compute_dataset_hash,
)


class _DummyPSvc:
    def calc_rows(self, lookup, df_pos, exclude_syms):  # noqa: ANN001
        df = df_pos.copy()
        df["valor_actual"] = 100.0
        df["costo"] = 80.0
        df["pl"] = 20.0
        df["pl_d"] = 20.0
        return df

    def classify_asset_cached(self, symbol):  # noqa: ANN001
        return "accion"


def test_apply_filters_reuses_preloaded_quotes(monkeypatch):
    df_pos = pd.DataFrame({"simbolo": ["GGAL"], "mercado": ["bcba"]})
    dataset_hash = _compute_dataset_hash(df_pos)
    quotes_map = {("bcba", "GGAL"): {"last": 100.0}}
    builder_calls: list[int] = []

    def _builder(cli, psvc):
        builder_calls.append(1)
        return PortfolioDataset(
            positions=df_pos.copy(),
            quotes=quotes_map.copy(),
            all_symbols=("GGAL",),
            available_types=("accion",),
            dataset_hash=dataset_hash,
            raw_payload={},
        )

    service = PortfolioDataFetchService(ttl_seconds=60.0, builder=_builder)
    service.get_dataset(SimpleNamespace(), _DummyPSvc(), force_refresh=True)
    assert len(builder_calls) == 1

    import services.data_fetch_service as dfs

    monkeypatch.setattr(dfs, "_SERVICE_SINGLETON", service)
    monkeypatch.setattr(dfs, "get_portfolio_data_fetch_service", lambda: service)

    fetch_calls: list[list[tuple[str, str]]] = []

    def _fake_fetch_quotes(cli, pairs):
        fetch_calls.append(list(pairs))
        return quotes_map

    monkeypatch.setattr(filters_mod, "fetch_quotes_bulk", _fake_fetch_quotes)

    controls = SimpleNamespace(
        hide_cash=False,
        selected_syms=[],
        selected_types=[],
        symbol_query="",
    )

    first = filters_mod.apply_filters(
        df_pos.copy(),
        controls,
        cli=SimpleNamespace(),
        psvc=_DummyPSvc(),
        dataset_hash=dataset_hash,
    )

    second = filters_mod.apply_filters(
        df_pos.copy(),
        controls,
        cli=SimpleNamespace(),
        psvc=_DummyPSvc(),
        dataset_hash=dataset_hash,
    )

    assert len(fetch_calls) == 0, "quotes should come from the preloaded dataset"
    assert not first.empty
    assert not second.empty
