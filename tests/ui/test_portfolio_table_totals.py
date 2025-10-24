"""Tests for totals row in the portfolio table."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from application.portfolio_service import calculate_totals
import ui.tables as tables_mod
from tests.fixtures.streamlit import UIFakeStreamlit


class _FavoritesStub:
    def is_favorite(self, _symbol: str) -> bool:
        return False


@pytest.fixture()
def _setup_streamlit(monkeypatch: pytest.MonkeyPatch) -> UIFakeStreamlit:
    fake_st = UIFakeStreamlit()
    fake_st.text_input = lambda *_, **__: ""
    fake_st.session_state["dataset_hash"] = "test-dataset"
    fake_st.subheader = lambda *_, **__: None

    class _ColumnConfigStub:
        class Column:  # pragma: no cover - simple stub
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class NumberColumn:  # pragma: no cover - simple stub
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class LineChartColumn:  # pragma: no cover - simple stub
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

    fake_st.column_config = _ColumnConfigStub()

    monkeypatch.setattr(tables_mod, "st", fake_st)
    monkeypatch.setattr(tables_mod, "ensure_fragment_ready_script", lambda *a, **k: None)
    monkeypatch.setattr(tables_mod, "register_fragment_ready", lambda *a, **k: None)
    monkeypatch.setattr(tables_mod, "mark_fragment_ready", lambda *a, **k: None)
    monkeypatch.setattr(tables_mod, "emit_fragment_ready", lambda *a, **k: None)
    monkeypatch.setattr(tables_mod, "download_csv", lambda *a, **k: None)
    palette = SimpleNamespace(positive="#16a34a", negative="#dc2626")
    monkeypatch.setattr(tables_mod, "get_active_palette", lambda: palette)

    return fake_st


def test_portfolio_table_includes_totals_row(_setup_streamlit: UIFakeStreamlit) -> None:
    df_view = pd.DataFrame(
        {
            "simbolo": ["GGAL", "YPFD", "AL30"],
            "tipo": ["ACCION", "ACCION", "BONO"],
            "cantidad": [10, 5, 20],
            "valor_actual": [1200.0, 1500.0, 800.0],
            "costo": [1000.0, 1300.0, 900.0],
            "pl": [200.0, 200.0, -100.0],
            "pl_%": [20.0, 15.3846153846, -11.1111111111],
            "pl_d": [10.0, 5.0, -2.0],
            "chg_%": [1.5, 0.8, -0.2],
        }
    )

    totals = calculate_totals(df_view)

    tables_mod.render_table(
        df_view=df_view,
        order_by="valor_actual",
        desc=True,
        ccl_rate=None,
        show_usd=False,
        favorites=_FavoritesStub(),
    )

    assert _setup_streamlit.dataframes, "Se esperaba que la tabla principal se renderice"
    styled = _setup_streamlit.dataframes[0][0]
    assert hasattr(styled, "data"), "Se esperaba un Styler con los datos renderizados"
    rendered_df = styled.data
    assert not rendered_df.empty

    totals_row = rendered_df.iloc[-1]
    assert totals_row["Símbolo"] == "Totales (suma de activos)"

    assert totals_row["valor_actual_num"] == pytest.approx(totals.total_value, rel=0.005)
    assert totals_row["costo_num"] == pytest.approx(totals.total_cost, rel=0.005)
    assert totals_row["pl_num"] == pytest.approx(totals.total_pl, rel=0.005)

    if np.isfinite(totals.total_pl_pct):
        assert totals_row["pl_pct_num"] == pytest.approx(totals.total_pl_pct, rel=0.005)
    else:
        assert np.isnan(totals_row["pl_pct_num"])
