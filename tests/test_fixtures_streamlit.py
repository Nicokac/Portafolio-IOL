from __future__ import annotations

from tests.fixtures.streamlit import BaseFakeStreamlit, FakeStreamlit, UIFakeStreamlit


def test_base_fake_streamlit_session_state_roundtrip() -> None:
    fake = BaseFakeStreamlit()
    fake.session_state["user"] = "alice"
    fake.session_state.setdefault("counter", 0)
    fake.session_state["counter"] += 1

    assert fake.session_state == {"user": "alice", "counter": 1}

    fake.clear_session_state()
    assert fake.session_state == {}


def test_logging_mixin_records_spinner_and_messages() -> None:
    fake = FakeStreamlit()

    with fake.spinner("loading portfolio"):
        fake.info("warming up")
        fake.warning("almost ready")

    fake.error("boom")

    assert fake.spinner_messages == ["loading portfolio"]
    assert fake.spinner_events == [
        ("start", "loading portfolio"),
        ("stop", "loading portfolio"),
    ]
    assert fake.messages == [
        ("info", "warming up"),
        ("warning", "almost ready"),
        ("error", "boom"),
    ]


def test_ui_fake_streamlit_extends_widget_contract() -> None:
    fake = UIFakeStreamlit(radio_sequence=[2], checkbox_values={"accept": True})

    value = fake.radio("tabs", options=[0, 1, 2], format_func=str, index=0, horizontal=True)
    assert value == 2

    fake.checkbox("accept", value=False, key="accept")
    fake.download_button("Export", b"data", file_name="export.csv")
    fake.caption("caption text")

    columns = fake.columns(2)
    columns[0].metric("Metric", 10, 2)

    assert fake.checkbox_calls
    assert fake.download_buttons[0]["file_name"] == "export.csv"
    assert fake.captions == ["caption text"]
    assert fake.metrics[0][0:3] == ("Metric", 10, 2)
