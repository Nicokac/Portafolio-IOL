import importlib
import json
import os
import sys
import types

import pytest

# Stub dotenv to avoid external dependency
sys.modules.setdefault('dotenv', types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None))

config = importlib.import_module('shared.config')


def test_get_config_sanitizes_types(monkeypatch, tmp_path):
    bad_cfg = {
        "cedear_to_us": None,
        "etfs": None,
        "acciones_ar": None,
        "fci_symbols": None,
        "scale_overrides": None,
        "classification_patterns": None,
    }
    cfg_file = tmp_path / 'conf.json'
    cfg_file.write_text(json.dumps(bad_cfg), encoding='utf-8')
    monkeypatch.setenv('PORTFOLIO_CONFIG_PATH', str(cfg_file))
    config.get_config.cache_clear()
    data = config.get_config()
    assert data["cedear_to_us"] == {}
    assert data["etfs"] == []
    assert data["acciones_ar"] == []
    assert data["fci_symbols"] == []
    assert data["scale_overrides"] == {}
    assert data["classification_patterns"] == {}
