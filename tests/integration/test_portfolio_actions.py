import csv

import pytest

from controllers.portfolio import portfolio as portfolio_mod
from shared import user_actions


class _DummyPlaceholder:
    def __init__(self, host: "_StubStreamlit") -> None:
        self._host = host

    def write(self, *_args, **_kwargs):
        return None

    def button(self, *args, **kwargs):
        return self._host.button(*args, **kwargs)

    def toggle(self, *args, **kwargs):
        return self._host.toggle(*args, **kwargs)

    def checkbox(self, *args, **kwargs):
        return self._host.checkbox(*args, **kwargs)


class _StubStreamlit:
    def __init__(self):
        self.session_state: dict[str, object] = {}
        self.button_calls: list[tuple] = []

    def button(self, label, *, key=None, **_kwargs):  # pragma: no cover - fallback guard
        if key is not None:
            self.session_state[key] = True
        self.button_calls.append((label, key))
        return True

    def toggle(self, label, *, key=None, value=False, **_kwargs):  # pragma: no cover - fallback guard
        if key is not None:
            self.session_state[key] = True
        return True

    def checkbox(self, label, *, key=None, value=False, **_kwargs):  # pragma: no cover - fallback guard
        if key is not None:
            self.session_state[key] = True
        return True


@pytest.fixture(autouse=True)
def patch_user_actions(monkeypatch, tmp_path):
    log_path = tmp_path / "user_actions.csv"
    monkeypatch.setattr(user_actions, "_LOG_PATH", log_path)
    monkeypatch.setattr(user_actions, "_resolve_user_id", lambda: "test-user")
    user_actions._reset_for_tests()
    yield log_path
    user_actions._reset_for_tests()


def test_portfolio_actions_are_logged(monkeypatch, patch_user_actions):
    log_path = patch_user_actions
    stub_st = _StubStreamlit()
    stub_st.session_state["dataset_hash"] = "hash123"
    monkeypatch.setattr(portfolio_mod, "st", stub_st)
    monkeypatch.setattr(user_actions, "st", stub_st, raising=False)
    monkeypatch.setattr("ui.lazy.runtime.current_scope", lambda: "fragment")

    block: dict[str, object] = {}
    placeholder = _DummyPlaceholder(stub_st)
    ready = portfolio_mod._prompt_lazy_block(
        block,
        placeholder=placeholder,
        button_label="📊 Cargar tabla del portafolio",
        info_message="",
        key="portafolio_load_table",
        dataset_token="hash123",
    )
    assert ready is True

    portfolio_mod._set_active_tab("resumen")
    portfolio_mod._log_quotes_refresh_event("hash123", source="manual_test")

    assert user_actions.wait_for_flush(2.0)
    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    actions = {(row["action"], row["dataset_hash"]) for row in rows}
    assert ("load_portfolio_table", "hash123") in actions
    assert any(row["action"] == "tab_change" and row["dataset_hash"] == "hash123" for row in rows)
    assert ("quotes_refresh", "hash123") in actions
