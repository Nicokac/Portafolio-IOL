from __future__ import annotations

import base64
import os

import pytest

from shared.security_env_validator import (
    SecurityValidationError,
    validate_security_environment,
)


def _encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii")


@pytest.fixture()
def valid_env() -> dict[str, str]:
    return {
        "FASTAPI_TOKENS_KEY": _encode(os.urandom(32)),
        "IOL_TOKENS_KEY": _encode(os.urandom(32)),
        "APP_ENV": "dev",
    }


def test_validate_security_environment_accepts_valid_keys(
    valid_env: dict[str, str],
) -> None:
    validate_security_environment(valid_env)


def test_missing_key_raises_error(valid_env: dict[str, str]) -> None:
    valid_env["APP_ENV"] = "prod"
    valid_env.pop("FASTAPI_TOKENS_KEY")

    with pytest.raises(SecurityValidationError) as exc_info:
        validate_security_environment(valid_env)

    assert "Falta la variable obligatoria FASTAPI_TOKENS_KEY" in str(exc_info.value)


def test_equal_keys_raise_error(valid_env: dict[str, str]) -> None:
    valid_env["APP_ENV"] = "prod"
    valid_env["FASTAPI_TOKENS_KEY"] = valid_env["IOL_TOKENS_KEY"]

    with pytest.raises(SecurityValidationError) as exc_info:
        validate_security_environment(valid_env)

    assert "no pueden ser iguales" in str(exc_info.value)


def test_weak_keys_warn_in_prod(valid_env: dict[str, str], caplog: pytest.LogCaptureFixture) -> None:
    weak_key = _encode(b"\x00" * 32)
    valid_env.update(
        {
            "FASTAPI_TOKENS_KEY": weak_key,
            "APP_ENV": "prod",
        }
    )

    with caplog.at_level("WARNING"):
        validate_security_environment(valid_env)

    assert any("parece débil" in message for message in caplog.messages)


def test_relaxed_mode_in_dev(valid_env: dict[str, str], caplog: pytest.LogCaptureFixture) -> None:
    valid_env.pop("FASTAPI_TOKENS_KEY")

    with caplog.at_level("WARNING"):
        report = validate_security_environment(valid_env)

    assert report.relaxed is True
    assert any("modo relajado" in message.lower() for message in caplog.messages)
