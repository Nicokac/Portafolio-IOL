import logging
from cryptography.fernet import Fernet

import pytest

from services import auth as auth_module
from services import diagnostics


@pytest.mark.parametrize(
    "state",
    [
        {
            "session_id": "abc",
            "auth_token": "secret-token",
            "IOL_USERNAME": "alice",
            "password": "hunter2",
            "nested": {"refresh_token": "refresh", "other": "ok"},
        }
    ],
)
def test_snapshot_session_state_redacts_sensitive_values(state):
    snapshot = diagnostics._snapshot_session_state(state)
    values = snapshot.get("values", {})
    assert values.get("auth_token") == "[REDACTED]"
    assert values.get("IOL_USERNAME") == "[REDACTED]"
    assert values.get("password") == "[REDACTED]"
    assert values.get("nested", {}).get("refresh_token") == "[REDACTED]"
    assert values.get("nested", {}).get("other") == "ok"


def test_analysis_logger_receives_redacted_snapshot(caplog):
    state = {
        "session_id": "abc",
        "IOL_PASSWORD": "super-secret",
        "tokens_file": "/tmp/tokens.json",
        "flags": True,
    }
    snapshot = diagnostics._snapshot_session_state(state)

    with caplog.at_level(logging.INFO, logger="analysis"):
        logging.getLogger("analysis").info("session_snapshot", extra={"state": snapshot})

    assert caplog.records
    record = caplog.records[0]
    stored_state = getattr(record, "state", {})
    assert stored_state.get("values", {}).get("IOL_PASSWORD") == "[REDACTED]"
    assert stored_state.get("values", {}).get("tokens_file") == "[REDACTED]"
    assert "super-secret" not in repr(stored_state)
    assert "/tmp/tokens.json" not in repr(stored_state)


def test_fastapi_and_iol_keys_must_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("FASTAPI_TOKENS_KEY", key)
    monkeypatch.setenv("IOL_TOKENS_KEY", key)

    with pytest.raises(RuntimeError):
        auth_module._get_tokens_key()

    monkeypatch.setenv("IOL_TOKENS_KEY", Fernet.generate_key().decode())
    assert auth_module._get_tokens_key()
