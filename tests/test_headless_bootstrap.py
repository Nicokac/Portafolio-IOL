from __future__ import annotations

import importlib
import sys
import time
from types import SimpleNamespace


def test_headless_bootstrap_loads_fast(monkeypatch):
    monkeypatch.setenv("UNIT_TEST", "1")
    sys.modules.pop("bootstrap.config", None)
    start = time.perf_counter()
    config = importlib.import_module("bootstrap.config")
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0
    assert config._is_headless_mode()
    assert callable(config.init_app)
    assert hasattr(config, "TOTAL_LOAD_START")

    sys.modules.pop("ui.fundamentals", None)
    fundamentals = importlib.import_module("ui.fundamentals")
    px = fundamentals._get_plotly_express()
    assert isinstance(px, SimpleNamespace)
    figure = px.bar(None)
    assert hasattr(figure, "add_hline")
