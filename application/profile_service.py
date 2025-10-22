"""Utilities to persist and expose the investor profile preferences."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, MutableMapping

from cryptography.fernet import Fernet, InvalidToken

try:  # pragma: no cover - streamlit may be absent in certain contexts
    import streamlit as st
except Exception:  # pragma: no cover - streamlit not installed during tests
    st = None  # type: ignore[assignment]


DEFAULT_PROFILE: dict[str, str] = {
    "risk_tolerance": "medio",
    "investment_horizon": "mediano",
    "preferred_mode": "diversify",
}


class ProfileService:
    """Persist lightweight investor preferences across sessions."""

    SESSION_KEY = "investor_profile"
    STORAGE_KEY = "investor_profile_encrypted"
    VALID_RISK = {"bajo", "medio", "alto"}
    VALID_HORIZON = {"corto", "mediano", "largo"}
    VALID_MODES = {"diversify", "max_return", "low_risk"}

    def __init__(
        self,
        *,
        storage_path: str | Path | None = None,
        session_state: MutableMapping[str, object] | None = None,
        secrets: Mapping[str, object] | None = None,
        encryption_key: str | None = None,
    ) -> None:
        self._storage_path = Path(storage_path or "config.json")
        self._session_state = session_state if session_state is not None else self._resolve_session_state()
        self._secrets = secrets if secrets is not None else self._resolve_secrets()
        self._encryption_key = encryption_key or self._resolve_key_material()
        self._cipher = Fernet(self._normalise_key(self._encryption_key))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_profile(self) -> dict[str, str]:
        """Return the active investor profile and hydrate session state."""

        profile = self._session_state.get(self.SESSION_KEY)
        if isinstance(profile, Mapping):
            normalized = self._normalise_profile(profile)
        else:
            normalized = None

        if not normalized:
            normalized = self._load_profile_from_sources()

        if not normalized:
            normalized = DEFAULT_PROFILE.copy()

        self._session_state[self.SESSION_KEY] = normalized
        return normalized.copy()

    def update_profile(
        self,
        *,
        risk_tolerance: str,
        investment_horizon: str,
        preferred_mode: str,
    ) -> dict[str, str]:
        """Persist the provided profile if it differs from the stored one."""

        current = self.get_profile()
        candidate = self._normalise_profile(
            {
                "risk_tolerance": risk_tolerance,
                "investment_horizon": investment_horizon,
                "preferred_mode": preferred_mode,
            }
        )
        if not candidate:
            return current

        if candidate == current:
            return current

        self._session_state[self.SESSION_KEY] = candidate
        self._persist_profile(candidate)
        return candidate.copy()

    def badge_label(self, profile: Mapping[str, str] | None = None) -> str:
        """Return a short label describing the active profile."""

        profile_data = self._normalise_profile(profile or self.get_profile())
        risk_map = {
            "bajo": "Conservador",
            "medio": "Moderado",
            "alto": "Dinámico",
        }
        horizon_map = {"corto": "3 m", "mediano": "12 m", "largo": "24 m+"}
        risk_text = risk_map.get(profile_data.get("risk_tolerance", ""), "Moderado")
        horizon_text = horizon_map.get(profile_data.get("investment_horizon", ""), "12 m")
        return f"🧩 Perfil: {risk_text} / {horizon_text}"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_session_state() -> MutableMapping[str, object]:  # pragma: no cover - UI glue
        if st is not None:
            session_state = getattr(st, "session_state", None)
            if isinstance(session_state, MutableMapping):
                return session_state
        return {}

    @staticmethod
    def _resolve_secrets() -> Mapping[str, object]:  # pragma: no cover - UI glue
        if st is not None:
            try:
                secrets = getattr(st, "secrets")
            except Exception:
                return {}
            if isinstance(secrets, Mapping):
                return secrets
        return {}

    def _resolve_key_material(self) -> str:
        env_key = os.getenv("PORTFOLIO_PROFILE_KEY")
        if env_key:
            return env_key
        try:
            secret_key = self._secrets.get("PORTFOLIO_PROFILE_KEY")
        except Exception:
            secret_key = None
        if isinstance(secret_key, str) and secret_key:
            return secret_key
        return "portfolio-iol-profile-v1"

    @staticmethod
    def _normalise_key(material: str) -> bytes:
        candidate = material.encode("utf-8")
        try:
            Fernet(candidate)
        except Exception:
            digest = hashlib.sha256(material.encode("utf-8")).digest()
            return base64.urlsafe_b64encode(digest)
        return candidate

    def _normalise_profile(self, profile: Mapping[str, object] | None) -> dict[str, str]:
        if not isinstance(profile, Mapping):
            return {}
        normalized: dict[str, str] = DEFAULT_PROFILE.copy()
        risk = str(profile.get("risk_tolerance", "")).strip().lower()
        horizon = str(profile.get("investment_horizon", "")).strip().lower()
        mode = str(profile.get("preferred_mode", "")).strip().lower()
        if risk in self.VALID_RISK:
            normalized["risk_tolerance"] = risk
        if horizon in self.VALID_HORIZON:
            normalized["investment_horizon"] = horizon
        if mode in self.VALID_MODES:
            normalized["preferred_mode"] = mode
        return normalized

    def _load_profile_from_sources(self) -> dict[str, str]:
        for source in (
            self._session_profile,
            self._secrets_profile,
            self._file_profile,
        ):
            profile = source()
            if profile:
                return profile
        return {}

    def _session_profile(self) -> dict[str, str]:
        profile = self._session_state.get(self.SESSION_KEY)
        if isinstance(profile, Mapping):
            return self._normalise_profile(profile)
        return {}

    def _secrets_profile(self) -> dict[str, str]:
        encrypted = self._secrets.get(self.STORAGE_KEY)
        if isinstance(encrypted, str) and encrypted:
            try:
                decoded = self._cipher.decrypt(encrypted.encode("utf-8"))
            except InvalidToken:
                return {}
            return self._normalise_profile(json.loads(decoded.decode("utf-8")))
        profile = self._secrets.get(self.SESSION_KEY)
        if isinstance(profile, Mapping):
            return self._normalise_profile(profile)
        return {}

    def _file_profile(self) -> dict[str, str]:
        path = self._storage_path
        try:
            if not path.exists():
                return {}
            content = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        encrypted = content.get(self.STORAGE_KEY)
        if not isinstance(encrypted, str) or not encrypted:
            return {}
        try:
            payload = self._cipher.decrypt(encrypted.encode("utf-8"))
        except InvalidToken:
            return {}
        try:
            profile = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return self._normalise_profile(profile)

    def _persist_profile(self, profile: Mapping[str, str]) -> None:
        path = self._storage_path
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
            else:
                raw = {}
        except (OSError, json.JSONDecodeError):
            raw = {}
        encrypted = self._cipher.encrypt(json.dumps(profile).encode("utf-8")).decode("utf-8")
        raw[self.STORAGE_KEY] = encrypted
        try:
            path.write_text(
                json.dumps(raw, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:  # pragma: no cover - IO failure is logged upstream
            return


__all__ = ["ProfileService", "DEFAULT_PROFILE"]
