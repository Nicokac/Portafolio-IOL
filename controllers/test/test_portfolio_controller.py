import contextlib
from unittest.mock import MagicMock, patch

import pandas as pd

from controllers.portfolio import render_portfolio_section
from domain.models import Controls


def test_render_portfolio_section_returns_refresh_secs_and_handles_empty():
    container = contextlib.nullcontext()
    mock_cli = MagicMock()
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.tabs.return_value = [contextlib.nullcontext() for _ in range(5)]

    empty_df = pd.DataFrame(columns=['simbolo'])

    with patch('controllers.portfolio.st', mock_st), \
         patch('controllers.portfolio.PortfolioService'), \
         patch('controllers.portfolio.TAService'), \
         patch('controllers.portfolio._load_portfolio_data', return_value=(pd.DataFrame(), [], [])), \
         patch('controllers.portfolio.render_sidebar', return_value=Controls(refresh_secs=55)), \
         patch('controllers.portfolio.render_ui_controls'), \
         patch('controllers.portfolio._apply_filters', return_value=empty_df):
        refresh = render_portfolio_section(container, cli=mock_cli, fx_rates={})

    assert refresh == 55
    mock_st.info.assert_any_call("No hay datos del portafolio para mostrar.")
