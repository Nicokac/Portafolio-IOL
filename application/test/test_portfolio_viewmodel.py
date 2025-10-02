import pandas as pd
import pandas.testing as pdt
import pytest

from application.portfolio_viewmodel import (
    build_portfolio_viewmodel,
    get_portfolio_tabs,
)
from domain.models import Controls


def test_build_portfolio_viewmodel_computes_totals_and_metrics():
    df_pos = pd.DataFrame({'simbolo': ['AAA']})
    controls = Controls(refresh_secs=60)
    filtered = pd.DataFrame({
        'simbolo': ['AAA'],
        'valor_actual': [120.0],
        'costo': [100.0],
    })

    vm = build_portfolio_viewmodel(
        df_pos=df_pos,
        controls=controls,
        cli=object(),
        portfolio_service=object(),
        fx_rates={'ccl': 900.0},
        all_symbols=['AAA'],
        apply_filters_fn=lambda *a, **k: filtered,
    )

    pdt.assert_frame_equal(vm.positions, filtered)
    assert vm.controls is controls
    assert vm.totals.total_value == pytest.approx(120.0)
    assert vm.totals.total_cost == pytest.approx(100.0)
    assert vm.totals.total_pl == pytest.approx(20.0)
    assert vm.metrics.refresh_secs == 60
    assert vm.metrics.ccl_rate == 900.0
    assert vm.metrics.all_symbols == ('AAA',)
    assert vm.metrics.has_positions is True
    assert vm.tab_options == get_portfolio_tabs()


def test_build_portfolio_viewmodel_handles_missing_data():
    controls = Controls()

    vm = build_portfolio_viewmodel(
        df_pos=pd.DataFrame(),
        controls=controls,
        cli=None,
        portfolio_service=None,
        fx_rates=None,
        all_symbols=None,
        apply_filters_fn=lambda *a, **k: None,
    )

    assert isinstance(vm.positions, pd.DataFrame)
    assert vm.positions.empty
    assert vm.metrics.has_positions is False
    assert vm.metrics.ccl_rate is None
    assert vm.metrics.all_symbols == ()
    assert vm.controls is controls
