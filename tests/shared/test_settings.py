"""Regression coverage for ``shared.config.Settings`` TTL defaults and overrides."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import shared.config as config


def _fresh_settings(monkeypatch, env: dict[str, str] | None = None) -> config.Settings:
    """Instantiate ``Settings`` with a clean configuration context."""

    for key in ("YAHOO_FUNDAMENTALS_TTL", "YAHOO_QUOTES_TTL"):
        monkeypatch.delenv(key, raising=False)

    if env:
        for key, value in env.items():
            monkeypatch.setenv(key, value)

    monkeypatch.setattr(config, "st", SimpleNamespace(secrets={}))
    monkeypatch.setattr(config, "_load_cfg", lambda: {})

    importlib.reload(config)
    monkeypatch.setattr(config, "st", SimpleNamespace(secrets={}))
    monkeypatch.setattr(config, "_load_cfg", lambda: {})

    return config.Settings()


def test_settings_use_expected_default_ttls(monkeypatch):
    """Defaults should fall back to 3600s (fundamentals) and 300s (quotes)."""

    settings = _fresh_settings(monkeypatch)

    assert settings.YAHOO_FUNDAMENTALS_TTL == 3600
    assert settings.YAHOO_QUOTES_TTL == 300


def test_settings_respect_environment_overrides(monkeypatch):
    """Environment variables should override the default TTL values."""

    settings = _fresh_settings(
        monkeypatch,
        {
            "YAHOO_FUNDAMENTALS_TTL": "42",
            "YAHOO_QUOTES_TTL": "99",
        },
    )

    assert settings.YAHOO_FUNDAMENTALS_TTL == 42
    assert settings.YAHOO_QUOTES_TTL == 99
