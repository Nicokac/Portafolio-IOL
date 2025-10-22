from types import SimpleNamespace
from typing import Callable

import pandas as pd

from services.data_fetch_service import PortfolioDataFetchService, PortfolioDataset


class _TimeController:
    def __init__(self) -> None:
        self.value = 0.0

    def advance(self, delta: float) -> None:
        self.value += float(delta)

    def now(self) -> float:
        return self.value


def _dataset_frame() -> pd.DataFrame:
    return pd.DataFrame({"simbolo": ["GGAL"], "mercado": ["bcba"]})


def _builder_factory(
    counter: list[int],
) -> Callable[[object, object], PortfolioDataset]:
    def _builder(_cli, _psvc):
        counter.append(1)
        frame = _dataset_frame()
        return PortfolioDataset(
            positions=frame.copy(),
            quotes={("bcba", "GGAL"): {"last": 100.0}},
            all_symbols=("GGAL",),
            available_types=("accion",),
            dataset_hash="hash",  # deterministic payload for tests
            raw_payload={},
        )

    return _builder


def test_get_dataset_uses_cached_snapshot() -> None:
    counter: list[int] = []
    clock = _TimeController()
    service = PortfolioDataFetchService(
        ttl_seconds=60.0,
        builder=_builder_factory(counter),
        time_provider=clock.now,
    )

    dataset, meta = service.get_dataset(SimpleNamespace(), SimpleNamespace(), force_refresh=True)

    assert dataset.positions.equals(_dataset_frame())
    assert meta.cache_hit is False
    assert len(counter) == 1

    dataset_cached, meta_cached = service.get_dataset(SimpleNamespace(), SimpleNamespace())
    assert dataset_cached.positions.equals(_dataset_frame())
    assert meta_cached.cache_hit is True
    assert len(counter) == 1, "builder should not run again within TTL"


def test_stale_dataset_triggers_background_refresh(monkeypatch) -> None:
    counter: list[int] = []
    clock = _TimeController()

    def _immediate_thread(target):
        class _Immediate:
            def start(self_nonlocal) -> None:
                target()

        return _Immediate()

    service = PortfolioDataFetchService(
        ttl_seconds=1.0,
        builder=_builder_factory(counter),
        time_provider=clock.now,
        background_factory=_immediate_thread,
    )

    service.get_dataset(SimpleNamespace(), SimpleNamespace(), force_refresh=True)
    assert len(counter) == 1

    clock.advance(2.0)
    dataset, meta = service.get_dataset(SimpleNamespace(), SimpleNamespace())
    assert dataset.positions.equals(_dataset_frame())
    assert meta.cache_hit is True
    assert meta.stale is True
    assert len(counter) == 2, "background refresh should run immediately"


def test_update_quotes_persists_changes() -> None:
    counter: list[int] = []
    clock = _TimeController()
    service = PortfolioDataFetchService(
        ttl_seconds=60.0,
        builder=_builder_factory(counter),
        time_provider=clock.now,
    )

    dataset, _ = service.get_dataset(SimpleNamespace(), SimpleNamespace(), force_refresh=True)
    quotes_update = {("bcba", "GGAL"): {"last": 123.45, "bid": 122.0}}
    service.update_quotes(dataset.dataset_hash, quotes_update)
    cached, _ = service.peek_dataset()
    assert cached is not None
    assert cached.quotes[("bcba", "GGAL")]["last"] == 123.45
