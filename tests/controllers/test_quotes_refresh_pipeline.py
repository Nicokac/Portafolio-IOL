from typing import Dict, Tuple
import sys
import time
import types
from pathlib import Path

import pandas as pd

# Avoid importing the heavy controllers.portfolio package during tests.
if "controllers.portfolio" not in sys.modules:
    stub = types.ModuleType("controllers.portfolio")
    stub.__path__ = [str(Path(__file__).resolve().parents[2] / "controllers" / "portfolio")]
    sys.modules["controllers.portfolio"] = stub

from controllers.portfolio.load_data import build_quote_batches, refresh_quotes_pipeline
from services.cache.core import CacheService
from services.cache.market_data_cache import StaleWhileRevalidateCache


class DummyPortfolioService:
    def __init__(self, mapping: Dict[str, str]) -> None:
        self._mapping = mapping

    def classify_asset_cached(self, sym: str) -> str:
        return self._mapping.get(sym, "otros")


class DummyFetcher:
    def __init__(self) -> None:
        self._versions: Dict[Tuple[str, str], int] = {}

    def __call__(self, _cli, batch):
        payload: Dict[Tuple[str, str], Dict[str, int]] = {}
        for pair in batch:
            self._versions[pair] = self._versions.get(pair, 0) + 1
            payload[pair] = {"version": self._versions[pair]}
        return payload


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"mercado": "BCBA", "simbolo": "GGAL", "tipo": "Finanzas"},
            {"mercado": "BCBA", "simbolo": "BMA", "tipo": "Finanzas"},
            {"mercado": "NASDAQ", "simbolo": "AAPL", "tipo": "Tecnologia"},
            {"mercado": "NASDAQ", "simbolo": "MSFT", "tipo": "Tecnologia"},
        ]
    )


def test_build_quote_batches_groups_by_type() -> None:
    df = _sample_df()
    svc = DummyPortfolioService({})

    batches = build_quote_batches(df, svc, batch_size=2)

    assert len(batches) == 2
    assert batches[0].group == "Finanzas"
    assert len(batches[0].pairs) == 2
    assert {sym for _, sym in batches[0].pairs} == {"GGAL", "BMA"}


def test_refresh_quotes_pipeline_uses_swr() -> None:
    df = _sample_df()
    svc = DummyPortfolioService({})
    fetcher = DummyFetcher()
    cache = CacheService()
    swr_cache = StaleWhileRevalidateCache(cache, default_ttl=0.1, grace_ttl=0.4, max_workers=2)

    try:
        data1, diag1 = refresh_quotes_pipeline(
            object(),
            df,
            svc,
            fetcher=fetcher,
            swr_cache=swr_cache,
            ttl=0.1,
            grace=0.4,
            batch_size=2,
            max_workers_override=2,
        )

        assert all(entry["served_mode"] == "refresh" for entry in diag1)
        assert all(value["version"] == 1 for value in data1.values())

        time.sleep(0.12)

        data2, diag2 = refresh_quotes_pipeline(
            object(),
            df,
            svc,
            fetcher=fetcher,
            swr_cache=swr_cache,
            ttl=0.1,
            grace=0.4,
            batch_size=2,
            max_workers_override=2,
        )

        assert data2 == data1
        assert all(entry["served_mode"] == "stale" for entry in diag2)

        swr_cache.wait()
        time.sleep(0.02)

        data3, diag3 = refresh_quotes_pipeline(
            object(),
            df,
            svc,
            fetcher=fetcher,
            swr_cache=swr_cache,
            ttl=0.1,
            grace=0.4,
            batch_size=2,
            max_workers_override=2,
        )

        assert all(not entry["stale"] for entry in diag3)
        assert all(value["version"] >= 2 for value in data3.values())
    finally:
        swr_cache.shutdown()
