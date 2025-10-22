from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any

import pytest

import shared.fragment_state as fragment_state


@dataclass
class _FakeStreamlit:
    session_state: dict[str, Any]


@pytest.fixture()
def persistence_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    fake_st = _FakeStreamlit(session_state={})
    events: list[tuple[str, Any, Any]] = []

    def _capture(
        action: str,
        detail: Any,
        *,
        dataset_hash: str | None = None,
        latency_ms: Any | None = None,
    ) -> None:
        events.append((action, detail, dataset_hash))

    monkeypatch.setattr(fragment_state, "st", fake_st)
    monkeypatch.setattr(fragment_state, "log_user_action", _capture)
    monkeypatch.setattr(fragment_state, "_PERSIST_DIR", tmp_path)
    monkeypatch.setattr(fragment_state, "_PERSIST_PATH", tmp_path / ".fragment_state.json")
    monkeypatch.setattr(fragment_state, "_PERSIST_LOCK", threading.Lock())
    monkeypatch.setattr(fragment_state, "_compute_lazy_modules_signature", lambda: "sig")
    monkeypatch.setattr(fragment_state, "get_current_user_id", lambda: "userA")
    fragment_state.reset_fragment_state_guardian()
    return fake_st, events


def _activate_fragment(fake_st: _FakeStreamlit, dataset_hash: str) -> None:
    fragment_state.reset_fragment_state_guardian()
    fake_st.session_state.clear()
    fake_st.session_state["dataset_hash"] = dataset_hash
    guardian = fragment_state.get_fragment_state_guardian()
    guardian.begin_cycle(dataset_hash)
    guardian.mark_ready(
        key="positions_load_table",
        session_key="positions_load_table",
        dataset_hash=dataset_hash,
        component="table",
        scope="portfolio",
        fallback_key="load_table",
    )
    fake_st.session_state["positions_load_table"] = True
    guardian.maybe_rehydrate(
        key="positions_load_table",
        session_key="positions_load_table",
        dataset_hash=dataset_hash,
        component="table",
        scope="portfolio",
        was_loaded=True,
        fallback_key="load_table",
    )
    assert fragment_state.persist_fragment_state_snapshot() is True


def _start_new_session(fake_st: _FakeStreamlit) -> None:
    fragment_state.reset_fragment_state_guardian()
    fake_st.session_state = {}
    fragment_state.reset_fragment_state_guardian()


def test_fragment_state_roundtrip_between_sessions(persistence_env):
    fake_st, events = persistence_env
    dataset_hash = "dataset-1"
    _activate_fragment(fake_st, dataset_hash)
    saved_events = [evt for evt in events if evt[0] == "fragment_state_saved"]
    assert saved_events
    assert saved_events[-1][1]["fragments"] == ["positions_load_table"]
    store = json.loads(fragment_state._PERSIST_PATH.read_text(encoding="utf-8"))
    assert store["users"]["userA"]["datasets"][dataset_hash]["fragments"]

    events.clear()
    _start_new_session(fake_st)
    fragment_state.prepare_persistent_fragment_restore()
    fake_st.session_state["dataset_hash"] = dataset_hash
    guardian = fragment_state.get_fragment_state_guardian()
    guardian.begin_cycle(dataset_hash)
    result = guardian.maybe_rehydrate(
        key="positions_load_table",
        session_key="positions_load_table",
        dataset_hash=dataset_hash,
        component="table",
        scope="portfolio",
        was_loaded=True,
        fallback_key="load_table",
    )
    assert result.rehydrated is True
    assert fake_st.session_state.get("positions_load_table") is True
    restored_events = [evt for evt in events if evt[0] == "fragment_state_restored"]
    assert restored_events
    assert restored_events[-1][1]["fragments"] == ["positions_load_table"]


def test_fragment_state_isolation_between_users(persistence_env, monkeypatch):
    fake_st, events = persistence_env
    dataset_hash = "dataset-2"
    _activate_fragment(fake_st, dataset_hash)

    events.clear()
    _start_new_session(fake_st)
    monkeypatch.setattr(fragment_state, "get_current_user_id", lambda: "userB")
    fragment_state.prepare_persistent_fragment_restore()
    fake_st.session_state["dataset_hash"] = dataset_hash
    guardian = fragment_state.get_fragment_state_guardian()
    guardian.begin_cycle(dataset_hash)
    result = guardian.maybe_rehydrate(
        key="positions_load_table",
        session_key="positions_load_table",
        dataset_hash=dataset_hash,
        component="table",
        scope="portfolio",
        was_loaded=True,
        fallback_key="load_table",
    )
    assert result.rehydrated is False
    assert fake_st.session_state.get("positions_load_table") is not True
    assert not [evt for evt in events if evt[0] == "fragment_state_restored"]


def test_fragment_state_clears_on_dataset_mismatch(persistence_env):
    fake_st, events = persistence_env
    _activate_fragment(fake_st, "dataset-3")

    events.clear()
    _start_new_session(fake_st)
    fragment_state.prepare_persistent_fragment_restore()
    fake_st.session_state["dataset_hash"] = "dataset-4"
    guardian = fragment_state.get_fragment_state_guardian()
    guardian.begin_cycle("dataset-4")
    result = guardian.maybe_rehydrate(
        key="positions_load_table",
        session_key="positions_load_table",
        dataset_hash="dataset-4",
        component="table",
        scope="portfolio",
        was_loaded=True,
        fallback_key="load_table",
    )
    assert result.rehydrated is False
    assert not [evt for evt in events if evt[0] == "fragment_state_restored"]
    store = json.loads(fragment_state._PERSIST_PATH.read_text(encoding="utf-8"))
    assert "userA" not in store.get("users", {})
