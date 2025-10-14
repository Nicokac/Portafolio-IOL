"""Validation helpers for security-related environment variables."""

from __future__ import annotations

import base64
import binascii
import logging
import os
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping

try:  # pragma: no cover - import guarded for lightweight runtimes
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - streamlit not available during some tests
    st = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class SecurityValidationError(RuntimeError):
    """Raised when mandatory security secrets are invalid or missing."""


@dataclass
class _KeyValidationResult:
    name: str
    value: str
    decoded: bytes
    is_weak: bool


_REQUIRED_KEYS: tuple[str, str] = ("FASTAPI_TOKENS_KEY", "IOL_TOKENS_KEY")
_EXPECTED_KEY_BYTES = 32


def _decode_key(name: str, raw_value: str) -> bytes:
    normalized_value = raw_value.strip()
    if not normalized_value:
        raise SecurityValidationError(f"La variable {name} está definida pero vacía.")

    padding = (-len(normalized_value)) % 4
    padded_value = normalized_value + ("=" * padding)
    try:
        decoded = base64.urlsafe_b64decode(padded_value.encode("ascii"))
    except (binascii.Error, ValueError) as exc:
        raise SecurityValidationError(
            f"La variable {name} no contiene una clave base64 válida."
        ) from exc

    if len(decoded) != _EXPECTED_KEY_BYTES:
        raise SecurityValidationError(
            f"La variable {name} debe representar {_EXPECTED_KEY_BYTES} bytes tras decodificar base64."
        )

    return decoded


def _is_weak(decoded: bytes) -> bool:
    unique_bytes = {b for b in decoded}
    return len(unique_bytes) <= 4


def _log_warnings(app_env: str | None, warnings: Iterable[str]) -> None:
    if app_env and app_env.lower() == "prod":
        for warning in warnings:
            logger.warning(warning)


EnvMapping = Mapping[str, str] | MutableMapping[str, str]


def _get_session_state(
    explicit_state: MutableMapping[str, object] | None = None,
):
    if explicit_state is not None:
        return explicit_state
    if st is None:
        return None
    try:
        return st.session_state
    except Exception:  # pragma: no cover - depends on Streamlit runtime
        return None


def validate_security_environment(
    environ: EnvMapping | None = None,
    *,
    session_state: MutableMapping[str, object] | None = None,
) -> None:
    """Validate mandatory secrets before the application starts once per session."""

    state = _get_session_state(session_state)
    if state is not None:
        try:
            if state.get("_security_validated"):
                return
        except Exception:  # pragma: no cover - custom state implementations
            pass

    env = environ or os.environ
    app_env = env.get("APP_ENV")

    results: list[_KeyValidationResult] = []
    errors: list[str] = []
    warnings: list[str] = []

    for key_name in _REQUIRED_KEYS:
        raw_value = env.get(key_name)
        if raw_value is None:
            errors.append(f"Falta la variable obligatoria {key_name} en el entorno.")
            continue

        try:
            decoded = _decode_key(key_name, raw_value)
        except SecurityValidationError as exc:
            errors.append(str(exc))
            continue

        is_weak = _is_weak(decoded)
        if is_weak:
            warnings.append(f"La clave {key_name} parece débil (baja entropía detectada).")

        results.append(
            _KeyValidationResult(
                name=key_name,
                value=raw_value.strip(),
                decoded=decoded,
                is_weak=is_weak,
            )
        )

    if len(results) == len(_REQUIRED_KEYS):
        fastapi_key, iol_key = results
        if fastapi_key.value == iol_key.value:
            warnings.append(
                "FASTAPI_TOKENS_KEY e IOL_TOKENS_KEY reutilizan la misma clave: genere valores distintos."
            )
            error_msg = "Las claves FASTAPI_TOKENS_KEY e IOL_TOKENS_KEY no pueden ser iguales."
            errors.append(error_msg)

    if errors:
        error_text = "\n".join(errors)
        _log_warnings(app_env, warnings)
        raise SecurityValidationError(
            "Configuración de seguridad inválida:\n" f"{error_text}"
        )

    _log_warnings(app_env, warnings)

    if state is not None:
        try:
            state["_security_validated"] = True
        except Exception:  # pragma: no cover - custom state implementations
            pass


def main() -> None:
    """CLI entry-point for CI validation."""

    validate_security_environment()


if __name__ == "__main__":
    main()
