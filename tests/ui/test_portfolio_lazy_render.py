import sys
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest

from application.portfolio_service import PortfolioTotals
from controllers.portfolio.portfolio import render_portfolio_section
from services.portfolio_view import (
    PortfolioContributionMetrics,
    PortfolioViewModelService,
    PortfolioViewSnapshot,
)
from tests.ui.test_portfolio_ui import FakeStreamlit

# Stub heavy optional dependencies used by portfolio controllers during import time.
statsmodels_mod = types.ModuleType("statsmodels")
statsmodels_api = types.ModuleType("statsmodels.api")
statsmodels_mod.api = statsmodels_api
sys.modules.setdefault("statsmodels", statsmodels_mod)
sys.modules.setdefault("statsmodels.api", statsmodels_api)

scipy_mod = types.ModuleType("scipy")
scipy_stats = types.ModuleType("scipy.stats")
scipy_sparse = types.ModuleType("scipy.sparse")
scipy_stats_qmc = types.ModuleType("scipy.stats._qmc")
scipy_stats_multicomp = types.ModuleType("scipy.stats._multicomp")
scipy_sparse_csgraph = types.ModuleType("scipy.sparse.csgraph")
scipy_sparse_shortest = types.ModuleType("scipy.sparse.csgraph._shortest_path")
scipy_mod.stats = scipy_stats
scipy_mod.sparse = scipy_sparse
scipy_sparse.csgraph = scipy_sparse_csgraph  # type: ignore[attr-defined]
sys.modules.setdefault("scipy", scipy_mod)
sys.modules.setdefault("scipy.stats", scipy_stats)
sys.modules.setdefault("scipy.stats._qmc", scipy_stats_qmc)
sys.modules.setdefault("scipy.stats._multicomp", scipy_stats_multicomp)
sys.modules.setdefault("scipy.sparse", scipy_sparse)
sys.modules.setdefault("scipy.sparse.csgraph", scipy_sparse_csgraph)
sys.modules.setdefault("scipy.sparse.csgraph._shortest_path", scipy_sparse_shortest)


@pytest.fixture
def _portfolio_setup(monkeypatch: pytest.MonkeyPatch):
    from tests.ui import test_portfolio_ui as portfolio_ui

    return portfolio_ui._portfolio_setup._fixture_function(monkeypatch)


class _ImmediateThread:
    def __init__(self, target, *args, **kwargs):
        self._target = target

    def start(self) -> None:
        self._target()

    def is_alive(self) -> bool:  # pragma: no cover - compatibility shim
        return False


class _LazySnapshotService:
    def __init__(self) -> None:
        self.pending = True

    def _hash_dataset(self, df):  # noqa: D401 - mimic service helper
        return PortfolioViewModelService._hash_dataset(df)

    def get_portfolio_view(
        self,
        df_pos,
        controls,
        cli,
        psvc,
        *,
        lazy_metrics: bool = False,
        dataset_hash: str | None = None,
        skip_invalidation: bool = False,
    ):  # noqa: ANN001,D417 - compatibility shim
        dataset_key = dataset_hash or self._hash_dataset(df_pos)
        self._dataset_key = dataset_key
        pending_metrics = ("extended_metrics",) if lazy_metrics and self.pending else ()
        history = (
            pd.DataFrame()
            if pending_metrics
            else pd.DataFrame(
                {
                    "timestamp": [0.0],
                    "total_value": [200.0],
                    "total_cost": [150.0],
                    "total_pl": [50.0],
                }
            )
        )
        return PortfolioViewSnapshot(
            df_view=pd.DataFrame({"simbolo": ["GGAL"], "valor_actual": [1200.0]}),
            totals=PortfolioTotals(200.0, 150.0, 50.0, 0.0, 20.0),
            apply_elapsed=0.05,
            totals_elapsed=0.02,
            generated_at=0.0,
            historical_total=history,
            contribution_metrics=PortfolioContributionMetrics.empty(),
            pending_metrics=pending_metrics,
        )

    def compute_extended_metrics(
        self,
        df_pos,
        controls,
        cli,
        psvc,
        *,
        dataset_hash: str | None = None,
        skip_invalidation: bool = False,
    ):  # noqa: ANN001,D417 - compatibility shim
        self.pending = False
        return self.get_portfolio_view(
            df_pos,
            controls,
            cli,
            psvc,
            lazy_metrics=False,
            dataset_hash=dataset_hash,
            skip_invalidation=skip_invalidation,
        )


def _render_portfolio(
    fake_st: FakeStreamlit,
    service: _LazySnapshotService,
    *,
    lazy_metrics: bool,
    setup_factory,
) -> None:
    (_portfolio_mod, *_rest, _vm_factory, notifications_factory) = setup_factory(fake_st)

    render_portfolio_section(
        fake_st.container(),
        cli=object(),
        fx_rates={"ccl": 0.0},
        view_model_service_factory=lambda: service,
        notifications_service_factory=notifications_factory,
        lazy_metrics=lazy_metrics,
    )


def test_lazy_metrics_initial_spinner_and_rerun(
    monkeypatch: pytest.MonkeyPatch,
    _portfolio_setup,  # pytest fixture from test_portfolio_ui
) -> None:
    fake_st = FakeStreamlit(radio_sequence=[0, 0])
    fake_st.experimental_rerun = MagicMock()
    monkeypatch.setattr("controllers.portfolio.portfolio.threading.Thread", _ImmediateThread)

    service = _LazySnapshotService()

    # Primera ejecución: muestra datos mínimos y deja pendientes las métricas
    _render_portfolio(fake_st, service, lazy_metrics=True, setup_factory=_portfolio_setup)

    spinner_messages = [text for text in fake_st.captions if "Calculando métricas extendidas" in text]
    assert spinner_messages, "Debe mostrarse un mensaje de carga para métricas extendidas"
    fake_st.experimental_rerun.assert_called_once()

    # Simula la finalización de la fase extendida y re-renderiza
    _render_portfolio(fake_st, service, lazy_metrics=True, setup_factory=_portfolio_setup)

    final_spinner_messages = [text for text in fake_st.captions if "Calculando métricas extendidas" in text]
    assert len(final_spinner_messages) == len(spinner_messages), (
        "El spinner no debe repetirse tras completar las métricas"
    )
    assert service.pending is False
