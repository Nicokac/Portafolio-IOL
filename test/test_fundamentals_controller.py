import pandas as pd
from unittest.mock import MagicMock
from contextlib import nullcontext

import streamlit as st

from controllers.portfolio.fundamentals import render_fundamental_analysis


def test_render_fundamental_analysis_empty_df(monkeypatch):
    monkeypatch.setattr(st, "subheader", MagicMock())
    info_mock = MagicMock()
    monkeypatch.setattr(st, "info", info_mock)

    df_view = pd.DataFrame(columns=["simbolo"])

    render_fundamental_analysis(df_view, MagicMock())

    info_mock.assert_called_once_with("No hay símbolos en el portafolio para analizar.")


def test_render_fundamental_analysis_with_symbols(monkeypatch):
    monkeypatch.setattr(st, "subheader", MagicMock())
    info_mock = MagicMock()
    monkeypatch.setattr(st, "info", info_mock)
    monkeypatch.setattr(st, "spinner", lambda *_args, **_kwargs: nullcontext())

    df_view = pd.DataFrame([{"simbolo": "AAPL"}])
    fund_df = pd.DataFrame([{"simbolo": "AAPL"}])
    tasvc = MagicMock()
    tasvc.portfolio_fundamentals.return_value = fund_df

    rfr = MagicMock()
    rsc = MagicMock()
    monkeypatch.setattr("controllers.portfolio.fundamentals.render_fundamental_ranking", rfr)
    monkeypatch.setattr("controllers.portfolio.fundamentals.render_sector_comparison", rsc)

    render_fundamental_analysis(df_view, tasvc)

    tasvc.portfolio_fundamentals.assert_called_once_with(["AAPL"])
    rfr.assert_called_once_with(fund_df)
    rsc.assert_called_once_with(fund_df)
    info_mock.assert_not_called()
