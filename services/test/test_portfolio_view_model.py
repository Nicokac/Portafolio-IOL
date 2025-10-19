import pandas as pd

from domain.models import Controls
from services.portfolio_view import PortfolioViewModelService


class DummyPSvc:
    pass


def test_get_portfolio_view_caches_apply_filters(monkeypatch):
    svc = PortfolioViewModelService()

    df_pos = pd.DataFrame({"simbolo": ["AL30"], "mercado": ["BCBA"]})
    df_view = pd.DataFrame(
        {
            "simbolo": ["AL30", "IOLPORA"],
            "mercado": ["BCBA", "BCBA"],
            "valor_actual": [100.0, 50.0],
            "costo": [80.0, 50.0],
        }
    )

    calls = {"count": 0}

    def fake_apply(df, controls, cli, psvc, *, dataset_hash=None, skip_invalidation=False):
        calls["count"] += 1
        return df_view

    monkeypatch.setattr("services.portfolio_view._apply_filters", fake_apply)

    controls = Controls()
    snapshot1 = svc.get_portfolio_view(df_pos, controls, cli=None, psvc=DummyPSvc())
    assert snapshot1.totals.total_value == 150.0
    assert snapshot1.totals.total_cash == 50.0

    snapshot2 = svc.get_portfolio_view(df_pos, controls, cli=None, psvc=DummyPSvc())
    assert snapshot2 is snapshot1
    assert calls["count"] == 1


def test_dataset_change_invalidates_cache(monkeypatch):
    svc = PortfolioViewModelService()

    base_df = pd.DataFrame({"simbolo": ["AL30"], "mercado": ["BCBA"]})
    new_df = pd.DataFrame({"simbolo": ["AL35"], "mercado": ["BCBA"]})

    outputs = [
        pd.DataFrame({
            "simbolo": ["AL30"],
            "mercado": ["BCBA"],
            "valor_actual": [100.0],
            "costo": [80.0],
        }),
        pd.DataFrame({
            "simbolo": ["AL35"],
            "mercado": ["BCBA"],
            "valor_actual": [200.0],
            "costo": [150.0],
        }),
    ]

    def fake_apply(df, controls, cli, psvc, *, dataset_hash=None, skip_invalidation=False):
        return outputs.pop(0)

    monkeypatch.setattr("services.portfolio_view._apply_filters", fake_apply)

    controls = Controls()
    snap_a = svc.get_portfolio_view(base_df, controls, cli=None, psvc=DummyPSvc())
    snap_b = svc.get_portfolio_view(new_df, controls, cli=None, psvc=DummyPSvc())

    assert snap_a is not snap_b
    assert snap_a.totals.total_value == 100.0
    assert snap_b.totals.total_value == 200.0


def test_filter_change_triggers_recomputation(monkeypatch):
    svc = PortfolioViewModelService()

    df_pos = pd.DataFrame({"simbolo": ["AL30"], "mercado": ["BCBA"]})
    df_view = pd.DataFrame({
        "simbolo": ["AL30"],
        "mercado": ["BCBA"],
        "valor_actual": [100.0],
        "costo": [90.0],
    })

    calls = {"count": 0}

    def fake_apply(df, controls, cli, psvc, *, dataset_hash=None, skip_invalidation=False):
        calls["count"] += 1
        return df_view

    monkeypatch.setattr("services.portfolio_view._apply_filters", fake_apply)

    controls_a = Controls(selected_syms=["AL30"])
    controls_b = Controls(selected_syms=["AL30", "AL35"])

    svc.get_portfolio_view(df_pos, controls_a, cli=None, psvc=DummyPSvc())
    svc.get_portfolio_view(df_pos, controls_b, cli=None, psvc=DummyPSvc())

    assert calls["count"] == 2
