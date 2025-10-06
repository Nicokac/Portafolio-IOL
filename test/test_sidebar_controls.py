from dataclasses import asdict

from ui import sidebar_controls
from domain.models import Controls


class FakeColumn:
    def __init__(self, api, submit_result=False):
        self._api = api
        self._submit_result = submit_result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def markdown(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def slider(self, label, *args, **kwargs):
        return self._api.slider(label, *args, **kwargs)

    def checkbox(self, label, *args, **kwargs):
        return self._api.checkbox(label, *args, **kwargs)

    def text_input(self, label, *args, **kwargs):
        return self._api.text_input(label, *args, **kwargs)

    def multiselect(self, label, options, *args, **kwargs):
        return self._api.multiselect(label, options, *args, **kwargs)

    def toggle(self, label, *args, **kwargs):
        return self._api.toggle(label, *args, **kwargs)

    def selectbox(self, label, options, *args, **kwargs):
        return self._api.selectbox(label, options, *args, **kwargs)

    def form_submit_button(self, label):
        return self._submit_result


class FakeForm:
    def __init__(self, api, *, apply=False, reset=False):
        self._api = api
        self._apply = apply
        self._reset = reset

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def columns(self, spec):
        if isinstance(spec, int):
            count = spec
        else:
            count = len(spec)
        cols = []
        for idx in range(count):
            submit = False
            if count == 2:
                submit = self._apply if idx == 0 else self._reset
            cols.append(FakeColumn(self._api, submit_result=submit))
        return tuple(cols)


class FakeContainer:
    def __init__(self, api, *, apply=False, reset=False):
        self._api = api
        self._apply = apply
        self._reset = reset

    def markdown(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def form(self, name):
        return FakeForm(self._api, apply=self._apply, reset=self._reset)


def make_st(return_vals, session_state=None, apply=False, reset=False):
    if session_state is None:
        session_state = {}

    class MockSt:
        def __init__(self):
            self.session_state = session_state
            self._rerun_called = False

        def markdown(self, *a, **k):
            pass

        def caption(self, *a, **k):
            pass

        def slider(self, label, *a, **k):
            return return_vals[label]

        def checkbox(self, label, *a, **k):
            return return_vals[label]

        def text_input(self, label, *a, **k):
            return return_vals[label]

        def multiselect(self, label, *a, **k):
            return return_vals[label]

        def toggle(self, label, *a, **k):
            return return_vals[label]

        def selectbox(self, label, *a, **k):
            return return_vals[label]

        def rerun(self):
            self._rerun_called = True

    api = MockSt()
    container = FakeContainer(api, apply=apply, reset=reset)
    return api, container


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
    fake_st, container = make_st(return_vals, session_state=initial_state, reset=True)
    monkeypatch.setattr(sidebar_controls, "st", fake_st)

    sidebar_controls.render_sidebar(["A"], ["T"], container=container)

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
    fake_st, container = make_st(return_vals, apply=True)
    monkeypatch.setattr(sidebar_controls, "st", fake_st)

    controls = sidebar_controls.render_sidebar(
        ["NVDA", "MSFT"], ["CEDEAR", "ACCION"], container=container
    )

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
