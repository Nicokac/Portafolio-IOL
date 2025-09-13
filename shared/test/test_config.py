import importlib
import json
import sys
import types

import pytest
import streamlit as st

# Stub dotenv to avoid external dependency
sys.modules.setdefault(
    "dotenv", types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)
)


def test_get_config_sanitizes_types(monkeypatch, tmp_path):
    config = importlib.import_module("shared.config")
    bad_cfg = {
        "cedear_to_us": None,
        "etfs": None,
        "acciones_ar": None,
        "fci_symbols": None,
        "scale_overrides": None,
        "classification_patterns": None,
    }
    cfg_file = tmp_path / "conf.json"
    cfg_file.write_text(json.dumps(bad_cfg), encoding="utf-8")
    monkeypatch.setenv("PORTFOLIO_CONFIG_PATH", str(cfg_file))
    config.get_config.cache_clear()
    data = config.get_config()
    assert data["cedear_to_us"] == {}
    assert data["etfs"] == []
    assert data["acciones_ar"] == []
    assert data["fci_symbols"] == []
    assert data["scale_overrides"] == {}
    assert data["classification_patterns"] == {}


def test_secrets_take_priority(monkeypatch):
    monkeypatch.setattr(st, "secrets", {"IOL_USERNAME": "secret"})
    monkeypatch.setenv("IOL_USERNAME", "env")
    config = importlib.reload(importlib.import_module("shared.config"))
    assert config.settings.IOL_USERNAME == "secret"


def test_env_used_when_no_secret(monkeypatch):
    monkeypatch.setattr(st, "secrets", {})
    monkeypatch.setenv("IOL_USERNAME", "env")
    config = importlib.reload(importlib.import_module("shared.config"))
    assert config.settings.IOL_USERNAME == "env"


def test_env_used_when_secrets_missing(monkeypatch):
    # st.secrets.get should raise StreamlitSecretNotFoundError
    class Secrets:
        def get(self, key):
            config = importlib.import_module("shared.config")
            raise config.StreamlitSecretNotFoundError("no secrets")

    monkeypatch.setattr(st, "secrets", Secrets())
    monkeypatch.setenv("IOL_USERNAME", "env")
    config = importlib.reload(importlib.import_module("shared.config"))
    assert config.settings.IOL_USERNAME == "env"

