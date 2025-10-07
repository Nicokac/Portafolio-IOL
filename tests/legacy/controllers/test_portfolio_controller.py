import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, ANY

import pandas as pd
import pytest

# NOTE: Casos heredados previos a la refactorización de `tests/controllers/`.
# Se conservan aquí para comparaciones manuales hasta completar la migración
# definitiva del controlador de portafolio.
from controllers.portfolio import (
    PortfolioMetrics,
    PortfolioViewModel,
    get_portfolio_tabs,
    render_portfolio_section,
)
from application.portfolio_service import calculate_totals
from domain.models import Controls
from services.portfolio_view import PortfolioContributionMetrics


def _make_viewmodel(df: pd.DataFrame, controls: Controls, all_symbols=None, ccl_rate=None):
    totals = calculate_totals(df)
    metrics = PortfolioMetrics(
        refresh_secs=controls.refresh_secs,
        ccl_rate=ccl_rate,
        all_symbols=tuple(all_symbols or ()),
        has_positions=not df.empty,
    )
    return PortfolioViewModel(
        positions=df,
        totals=totals,
        controls=controls,
        metrics=metrics,
        tab_options=get_portfolio_tabs(),
        historical_total=pd.DataFrame(),
        contributions=PortfolioContributionMetrics.empty(),
    )


def _make_snapshot(vm: PortfolioViewModel, *, apply_elapsed: float = 0.0, totals_elapsed: float = 0.0):
    return SimpleNamespace(
        df_view=vm.positions,
        totals=vm.totals,
        apply_elapsed=apply_elapsed,
        totals_elapsed=totals_elapsed,
        generated_at=0.0,
    )


def _notifications_stub() -> SimpleNamespace:
    return SimpleNamespace(
        get_flags=lambda: SimpleNamespace(
            risk_alert=False, technical_signal=False, upcoming_earnings=False
        )
    )


def test_render_portfolio_section_returns_refresh_secs_and_handles_empty():
    container = contextlib.nullcontext()
    mock_cli = MagicMock()
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.radio.return_value = 0

    empty_df = pd.DataFrame(columns=['simbolo'])
    controls = Controls(refresh_secs=55)
    vm = _make_viewmodel(empty_df, controls, all_symbols=[])

    snapshot = _make_snapshot(vm)

    with patch('controllers.portfolio.portfolio.st', mock_st), \
         patch('controllers.portfolio.charts.st', mock_st), \
         patch('controllers.portfolio.portfolio.PortfolioService'), \
         patch('controllers.portfolio.portfolio.TAService'), \
         patch('controllers.portfolio.portfolio.load_portfolio_data', return_value=(pd.DataFrame(), [], [])), \
         patch('controllers.portfolio.portfolio.render_sidebar', return_value=controls), \
         patch(
             'controllers.portfolio.portfolio.get_portfolio_view_service',
             return_value=SimpleNamespace(get_portfolio_view=lambda **_: snapshot),
         ), \
         patch(
             'controllers.portfolio.portfolio.get_notifications_service',
             return_value=_notifications_stub(),
         ), \
         patch('controllers.portfolio.portfolio.build_portfolio_viewmodel', return_value=vm), \
         patch('controllers.portfolio.portfolio.render_advanced_analysis'), \
         patch('controllers.portfolio.portfolio.render_risk_analysis'), \
         patch('controllers.portfolio.portfolio.render_fundamental_analysis'):
        refresh = render_portfolio_section(container, cli=mock_cli, fx_rates={})

    assert refresh == 55
    mock_st.info.assert_any_call("No hay datos del portafolio para mostrar.")


def test_ta_section_without_symbols_shows_message():
    container = contextlib.nullcontext()
    mock_cli = MagicMock()
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.radio.return_value = 4
    controls = Controls(refresh_secs=0)
    vm = _make_viewmodel(pd.DataFrame(), controls, all_symbols=[])
    snapshot = _make_snapshot(vm)

    with patch('controllers.portfolio.portfolio.st', mock_st), \
         patch('controllers.portfolio.portfolio.PortfolioService'), \
         patch('controllers.portfolio.portfolio.TAService'), \
         patch('controllers.portfolio.portfolio.load_portfolio_data', return_value=(pd.DataFrame(), [], [])), \
         patch('controllers.portfolio.portfolio.render_sidebar', return_value=controls), \
         patch(
             'controllers.portfolio.portfolio.get_portfolio_view_service',
             return_value=SimpleNamespace(get_portfolio_view=lambda **_: snapshot),
         ), \
         patch(
             'controllers.portfolio.portfolio.get_notifications_service',
             return_value=_notifications_stub(),
         ), \
         patch('controllers.portfolio.portfolio.build_portfolio_viewmodel', return_value=vm), \
         patch('controllers.portfolio.portfolio.render_basic_section'), \
         patch('controllers.portfolio.portfolio.render_advanced_analysis'), \
         patch('controllers.portfolio.portfolio.render_risk_analysis'), \
         patch('controllers.portfolio.portfolio.render_fundamental_analysis'):
        render_portfolio_section(container, cli=mock_cli, fx_rates={})

    mock_st.info.assert_any_call("No hay símbolos en el portafolio para analizar.")


@pytest.mark.parametrize(
    'tab_idx,func_name',
    [
        (0, 'render_basic_section'),
        (1, 'render_advanced_analysis'),
        (2, 'render_risk_analysis'),
        (3, 'render_fundamental_analysis'),
    ],
)
def test_tabs_render_expected_sections(tab_idx, func_name):
    container = contextlib.nullcontext()
    mock_cli = MagicMock()
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.radio.return_value = tab_idx

    df = pd.DataFrame({'simbolo': ['AAA']})
    controls = Controls(refresh_secs=0)
    vm = _make_viewmodel(df, controls, all_symbols=['AAA'])

    snapshot = _make_snapshot(vm)

    with patch('controllers.portfolio.portfolio.st', mock_st), \
         patch('controllers.portfolio.portfolio.PortfolioService'), \
         patch('controllers.portfolio.portfolio.TAService'), \
         patch('controllers.portfolio.portfolio.load_portfolio_data', return_value=(df, ['AAA'], [])), \
         patch('controllers.portfolio.portfolio.render_sidebar', return_value=controls), \
         patch(
             'controllers.portfolio.portfolio.get_portfolio_view_service',
             return_value=SimpleNamespace(get_portfolio_view=lambda **_: snapshot),
         ), \
         patch(
             'controllers.portfolio.portfolio.get_notifications_service',
             return_value=_notifications_stub(),
         ), \
         patch('controllers.portfolio.portfolio.build_portfolio_viewmodel', return_value=vm), \
         patch('controllers.portfolio.portfolio.render_basic_section') as basic, \
         patch('controllers.portfolio.portfolio.render_advanced_analysis') as adv, \
         patch('controllers.portfolio.portfolio.render_risk_analysis') as risk, \
         patch('controllers.portfolio.portfolio.render_fundamental_analysis') as fund:
        render_portfolio_section(container, cli=mock_cli, fx_rates={})

    calls = {
        'render_basic_section': basic,
        'render_advanced_analysis': adv,
        'render_risk_analysis': risk,
        'render_fundamental_analysis': fund,
    }
    for name, fn in calls.items():
        if name == func_name:
            fn.assert_called_once()
        else:
            fn.assert_not_called()


def test_ta_section_symbol_without_us_ticker():
    container = contextlib.nullcontext()
    mock_cli = MagicMock()
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.radio.return_value = 4
    mock_st.selectbox.return_value = 'AAA'
    mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]

    df = pd.DataFrame({'simbolo': ['AAA']})
    controls = Controls(refresh_secs=0)
    vm = _make_viewmodel(df, controls, all_symbols=['AAA'])

    snapshot = _make_snapshot(vm)

    with patch('controllers.portfolio.portfolio.st', mock_st), \
         patch('controllers.portfolio.portfolio.PortfolioService'), \
         patch('controllers.portfolio.portfolio.TAService'), \
         patch('controllers.portfolio.portfolio.load_portfolio_data', return_value=(df, ['AAA'], [])), \
         patch('controllers.portfolio.portfolio.render_sidebar', return_value=controls), \
         patch(
             'controllers.portfolio.portfolio.get_portfolio_view_service',
             return_value=SimpleNamespace(get_portfolio_view=lambda **_: snapshot),
         ), \
         patch('controllers.portfolio.portfolio.build_portfolio_viewmodel', return_value=vm), \
         patch('controllers.portfolio.portfolio.map_to_us_ticker', side_effect=ValueError('invalid')), \
         patch('controllers.portfolio.portfolio.render_basic_section'), \
         patch('controllers.portfolio.portfolio.render_advanced_analysis'), \
         patch('controllers.portfolio.portfolio.render_risk_analysis'), \
         patch('controllers.portfolio.portfolio.render_fundamental_analysis'), \
         patch('controllers.portfolio.portfolio.render_favorite_toggle'), \
         patch('controllers.portfolio.portfolio.render_favorite_badges'):
        render_portfolio_section(container, cli=mock_cli, fx_rates={})

    mock_st.info.assert_any_call("No se encontró ticker US para este activo.")


def _mock_columns(seq):
    cols = []
    for n in seq:
        col = MagicMock()
        col.number_input.return_value = 1
        col.selectbox.return_value = 'X'
        cols.append(col)
    return cols


def test_ta_section_symbol_with_empty_df():
    container = contextlib.nullcontext()
    mock_cli = MagicMock()
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.radio.return_value = 4
    mock_st.selectbox.side_effect = ['AAA', '3mo', '1d', 'SMA']
    mock_st.number_input.return_value = 1
    mock_st.columns.side_effect = [
        _mock_columns(range(4)),
        _mock_columns(range(3)),
        _mock_columns(range(3)),
        _mock_columns(range(3)),
    ]
    mock_st.expander.return_value.__enter__.return_value = MagicMock()

    df = pd.DataFrame({'simbolo': ['AAA']})
    mock_tasvc = MagicMock()
    mock_tasvc.fundamentals.return_value = {}
    mock_tasvc.indicators_for.return_value = pd.DataFrame()
    controls = Controls(refresh_secs=0)
    vm = _make_viewmodel(df, controls, all_symbols=['AAA'])
    snapshot = _make_snapshot(vm)

    with patch('controllers.portfolio.portfolio.st', mock_st), \
         patch('controllers.portfolio.portfolio.PortfolioService'), \
         patch('controllers.portfolio.portfolio.TAService', return_value=mock_tasvc), \
         patch('controllers.portfolio.portfolio.load_portfolio_data', return_value=(df, ['AAA'], [])), \
         patch('controllers.portfolio.portfolio.render_sidebar', return_value=controls), \
         patch(
             'controllers.portfolio.portfolio.get_portfolio_view_service',
             return_value=SimpleNamespace(get_portfolio_view=lambda **_: snapshot),
         ), \
         patch('controllers.portfolio.portfolio.build_portfolio_viewmodel', return_value=vm), \
         patch('controllers.portfolio.portfolio.map_to_us_ticker', return_value='AA'), \
         patch('controllers.portfolio.portfolio.render_basic_section'), \
         patch('controllers.portfolio.portfolio.render_advanced_analysis'), \
         patch('controllers.portfolio.portfolio.render_risk_analysis'), \
         patch('controllers.portfolio.portfolio.render_fundamental_analysis'), \
         patch('controllers.portfolio.portfolio.render_fundamental_data'), \
         patch('controllers.portfolio.portfolio.plot_technical_analysis_chart'), \
         patch('controllers.portfolio.portfolio.render_favorite_toggle'), \
         patch('controllers.portfolio.portfolio.render_favorite_badges'):
        render_portfolio_section(container, cli=mock_cli, fx_rates={})

    mock_st.info.assert_any_call(
        "No se pudo descargar histórico para ese símbolo/periodo/intervalo."
    )


def test_ta_section_symbol_with_data():
    container = contextlib.nullcontext()
    mock_cli = MagicMock()
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.radio.return_value = 4
    mock_st.selectbox.side_effect = ['AAA', '3mo', '1d', 'SMA']
    mock_st.number_input.return_value = 1
    mock_st.columns.side_effect = [
        _mock_columns(range(4)),
        _mock_columns(range(3)),
        _mock_columns(range(3)),
        _mock_columns(range(3)),
    ]
    mock_st.expander.return_value.__enter__.return_value = MagicMock()

    df = pd.DataFrame({'simbolo': ['AAA']})
    df_ind = pd.DataFrame({'close': [1, 2]})
    mock_tasvc = MagicMock()
    mock_tasvc.fundamentals.return_value = {}
    mock_tasvc.indicators_for.return_value = df_ind
    mock_tasvc.alerts_for.return_value = []
    mock_tasvc.backtest.return_value = pd.DataFrame({'equity': [1.0, 1.1]})
    controls = Controls(refresh_secs=0)
    vm = _make_viewmodel(df, controls, all_symbols=['AAA'])
    snapshot = _make_snapshot(vm)

    with patch('controllers.portfolio.portfolio.st', mock_st), \
         patch('controllers.portfolio.portfolio.PortfolioService'), \
         patch('controllers.portfolio.portfolio.TAService', return_value=mock_tasvc), \
         patch('controllers.portfolio.portfolio.load_portfolio_data', return_value=(df, ['AAA'], [])), \
         patch('controllers.portfolio.portfolio.render_sidebar', return_value=controls), \
         patch(
             'controllers.portfolio.portfolio.get_portfolio_view_service',
             return_value=SimpleNamespace(get_portfolio_view=lambda **_: snapshot),
         ), \
         patch('controllers.portfolio.portfolio.build_portfolio_viewmodel', return_value=vm), \
         patch('controllers.portfolio.portfolio.map_to_us_ticker', return_value='AA'), \
         patch('controllers.portfolio.portfolio.render_basic_section'), \
         patch('controllers.portfolio.portfolio.render_advanced_analysis'), \
         patch('controllers.portfolio.portfolio.render_risk_analysis'), \
         patch('controllers.portfolio.portfolio.render_fundamental_analysis'), \
         patch('controllers.portfolio.portfolio.render_fundamental_data'), \
         patch('controllers.portfolio.portfolio.plot_technical_analysis_chart', return_value='fig'), \
         patch('controllers.portfolio.portfolio.render_favorite_toggle'), \
         patch('controllers.portfolio.portfolio.render_favorite_badges'):
        render_portfolio_section(container, cli=mock_cli, fx_rates={})

    mock_st.plotly_chart.assert_called_once_with(
        'fig', width="stretch", key='ta_chart', config=ANY
    )
