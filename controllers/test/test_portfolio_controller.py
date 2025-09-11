import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from controllers.portfolio import render_portfolio_section
from domain.models import Controls


def test_render_portfolio_section_returns_refresh_secs_and_handles_empty():
    container = contextlib.nullcontext()
    mock_cli = MagicMock()
    mock_st = MagicMock()
    mock_cache = SimpleNamespace(session_state={})
    mock_st.radio.return_value = 0

    empty_df = pd.DataFrame(columns=['simbolo'])

    with patch('controllers.portfolio.portfolio.st', mock_st), \
         patch('controllers.portfolio.portfolio.cache', mock_cache), \
         patch('controllers.portfolio.charts.st', mock_st), \
         patch('controllers.portfolio.portfolio.PortfolioService'), \
         patch('controllers.portfolio.portfolio.TAService'), \
         patch('controllers.portfolio.portfolio.load_portfolio_data', return_value=(pd.DataFrame(), [], [])), \
         patch('controllers.portfolio.portfolio.render_sidebar', return_value=Controls(refresh_secs=55)), \
         patch('controllers.portfolio.portfolio.render_ui_controls'), \
         patch('controllers.portfolio.portfolio.apply_filters', return_value=empty_df), \
         patch('controllers.portfolio.portfolio.render_advanced_analysis'), \
         patch('controllers.portfolio.portfolio.render_risk_analysis'), \
         patch('controllers.portfolio.portfolio.render_fundamental_analysis'):
        refresh = render_portfolio_section(container, cli=mock_cli, fx_rates={})

    assert refresh == 55
    mock_st.info.assert_any_call("No hay datos del portafolio para mostrar.")
