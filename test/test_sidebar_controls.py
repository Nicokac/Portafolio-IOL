from dataclasses import asdict
from types import SimpleNamespace

from ui import sidebar_controls
from domain.models import Controls


class FakeForm:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class FakeColumn:
    def __init__(self, result):
        self.result = result

    def form_submit_button(self, label):
        return self.result


def make_st(return_vals, session_state=None, apply=False, reset=False):
    if session_state is None:
        session_state = {}

    class MockSt:
        def __init__(self):
            self.session_state = session_state
            self._rerun_called = False
            self.sidebar = SimpleNamespace(
                header=lambda *a, **k: None,
                form=lambda name: FakeForm(),
            )

        def markdown(self, *a, **k):
            pass

        def caption(self, *a, **k):
            pass

        def slider(self, label, *a, **k):
            return return_vals[label]

        def checkbox(self, label, value=False):
            return return_vals[label]

        def text_input(self, label, value="", placeholder=""):
            return return_vals[label]

        def multiselect(self, label, options, default=None):
            return return_vals[label]

        def toggle(self, label, value=False):
            return return_vals[label]

        def selectbox(self, label, options, index=0):
            return return_vals[label]

        def columns(self, n):
            return (FakeColumn(apply), FakeColumn(reset))

        def rerun(self):
            self._rerun_called = True

    return MockSt()


def test_reset_btn_clears_session_and_reruns(monkeypatch):
    return_vals = {
        "Intervalo (seg)": 10,
        "Ocultar IOLPORA / PARKING": False,
        "Buscar símbolo": "",
        "Filtrar por símbolo": ["A"],
        "Filtrar por tipo": ["T"],
        "Mostrar valores en USD CCL": False,
        "Ordenar por": "valor_actual",
        "Descendente": True,
        "Top N": 20,
    }
    initial_state = {
        "refresh_secs": 99,
        "hide_cash": True,
        "show_usd": True,
        "order_by": "pl",
        "desc": False,
        "top_n": 5,
        "selected_syms": ["X"],
        "selected_types": ["Y"],
        "symbol_query": "X",
        "controls_snapshot": {"foo": "bar"},
    }
    fake_st = make_st(return_vals, session_state=initial_state, reset=True)
    monkeypatch.setattr(sidebar_controls, "st", fake_st)

    sidebar_controls.render_sidebar(["A"], ["T"])

    keys = [
        "refresh_secs",
        "hide_cash",
        "show_usd",
        "order_by",
        "desc",
        "top_n",
        "selected_syms",
        "selected_types",
        "symbol_query",
    ]
    for key in keys:
        assert key not in fake_st.session_state
    assert fake_st.session_state.get("controls_snapshot") is None
    assert fake_st._rerun_called


def test_apply_btn_updates_session_with_controls(monkeypatch):
    return_vals = {
        "Intervalo (seg)": 15,
        "Ocultar IOLPORA / PARKING": True,
        "Buscar símbolo": "nvda",
        "Filtrar por símbolo": ["NVDA"],
        "Filtrar por tipo": ["CEDEAR"],
        "Mostrar valores en USD CCL": True,
        "Ordenar por": "pl",
        "Descendente": False,
        "Top N": 10,
    }
    fake_st = make_st(return_vals, apply=True)
    monkeypatch.setattr(sidebar_controls, "st", fake_st)

    controls = sidebar_controls.render_sidebar(["NVDA", "MSFT"], ["CEDEAR", "ACCION"])

    expected = Controls(
        refresh_secs=15,
        hide_cash=True,
        show_usd=True,
        order_by="pl",
        desc=False,
        top_n=10,
        selected_syms=["NVDA"],
        selected_types=["CEDEAR"],
        symbol_query="NVDA",
    )

    assert controls == expected
    for k, v in asdict(expected).items():
        assert fake_st.session_state[k] == v
    assert fake_st.session_state["controls_snapshot"] == asdict(expected)
    assert not fake_st._rerun_called
