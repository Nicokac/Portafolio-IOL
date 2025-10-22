import importlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


@contextmanager
def _noop_context():
    yield


def test_portfolio_visibility_restored_when_unlocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("ui.controllers.portfolio_ui")

    calls: list[tuple[str, object | None]] = []

    class Guardian:
        def prepare_persistent_restore(self) -> None:
            calls.append(("prepare", None))

        def begin_cycle(self, dataset_hash: str) -> None:
            calls.append(("begin", dataset_hash))

    guardian = Guardian()

    state = {
        "_hydration_lock": False,
        "dataset_hash": "hash-123",
        "tab_loaded": {},
        "cached_render": {},
    }

    st_stub = SimpleNamespace(session_state=state)
    monkeypatch.setattr(module, "st", st_stub, raising=False)

    snapshot_flags = {"busy": 0, "idle": 0, "completed": 0}
    monkeypatch.setattr(
        module,
        "snapshot_defer",
        SimpleNamespace(
            mark_ui_busy=lambda: snapshot_flags.__setitem__("busy", snapshot_flags["busy"] + 1),
            mark_ui_idle=lambda: snapshot_flags.__setitem__("idle", snapshot_flags["idle"] + 1),
        ),
    )
    monkeypatch.setattr(
        module,
        "mark_portfolio_ui_render_complete",
        lambda: snapshot_flags.__setitem__("completed", snapshot_flags["completed"] + 1),
    )
    monkeypatch.setattr(module, "get_fragment_state_guardian", lambda: guardian)
    monkeypatch.setattr(module, "log_user_action", lambda *a, **k: None)

    @contextmanager
    def _performance_timer(*_args, **_kwargs):
        yield

    monkeypatch.setattr(module, "_get_performance_timer", lambda: _performance_timer)
    monkeypatch.setattr(module, "measure_execution", lambda *a, **k: _noop_context())
    monkeypatch.setattr(module, "_get_portfolio_section", lambda: lambda *a, **k: 45)

    result = module.render_portfolio_ui(container=None, cli=object(), fx_rates={})

    assert result == 45
    assert ("prepare", None) in calls
    assert ("begin", "hash-123") in calls
    assert state["ui_idle"] is True
    assert snapshot_flags == {"busy": 1, "idle": 1, "completed": 1}
