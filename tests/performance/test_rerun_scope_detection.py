from __future__ import annotations

import ui.lazy.runtime as lazy_runtime


class _GuardianStub:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    def wait_for_hydration(self, dataset_hash: str | None = None, *, timeout: float = 0.25):
        self.calls.append(dataset_hash)
        return True


class _ContextManagerStub:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _StubStreamlit:
    def __init__(self, *, provide_fragment: bool, provide_form: bool) -> None:
        self.session_state: dict[str, object] = {}
        self.fragment_calls: list[str] = []
        self.form_calls: list[str] = []
        self.container_calls: int = 0

        if provide_fragment:
            self.fragment = self._fragment_impl  # type: ignore[attr-defined]
            self.experimental_fragment = self._fragment_impl  # type: ignore[attr-defined]
        if provide_form:
            self.form = self._form_impl  # type: ignore[attr-defined]

    def _fragment_impl(self, name: str):
        self.fragment_calls.append(name)
        return _ContextManagerStub()

    def _form_impl(self, key: str):
        self.form_calls.append(key)
        return _ContextManagerStub()

    def container(self):
        self.container_calls += 1
        return _ContextManagerStub()

    def stop(self):  # pragma: no cover - not used in these tests
        return None


def test_lazy_fragment_logs_dataset_and_scope(monkeypatch):
    events: list[dict[str, object]] = []
    fake_st = _StubStreamlit(provide_fragment=True, provide_form=False)
    monkeypatch.setattr(lazy_runtime, "st", fake_st)
    monkeypatch.setattr(lazy_runtime, "log_default_telemetry", lambda **kwargs: events.append(kwargs))
    guardian = _GuardianStub()
    monkeypatch.setattr(lazy_runtime, "get_fragment_state_guardian", lambda: guardian)

    with lazy_runtime.lazy_fragment("portfolio_table", component="table", dataset_token="abc") as context:
        assert context.scope == "fragment"

    assert fake_st.fragment_calls == ["portfolio_table"]
    assert events
    payload = events[-1]
    assert payload.get("dataset_hash") == "abc"
    extra = payload.get("extra", {})
    assert extra.get("ui_rerun_scope") == "fragment"
    assert extra.get("lazy_loaded_component") == "table"
    assert guardian.calls == ["abc"]


def test_lazy_fragment_falls_back_to_global_scope(monkeypatch):
    events: list[dict[str, object]] = []
    fake_st = _StubStreamlit(provide_fragment=False, provide_form=False)
    monkeypatch.setattr(lazy_runtime, "st", fake_st)
    monkeypatch.setattr(lazy_runtime, "log_default_telemetry", lambda **kwargs: events.append(kwargs))
    guardian = _GuardianStub()
    monkeypatch.setattr(lazy_runtime, "get_fragment_state_guardian", lambda: guardian)

    with lazy_runtime.lazy_fragment("portfolio_table", component="table", dataset_token=None) as context:
        assert context.scope == "global"

    assert fake_st.container_calls == 1
    payload = events[-1]
    extra = payload.get("extra", {})
    assert extra.get("ui_rerun_scope") == "global"
    assert guardian.calls == [None]
