import importlib.util
import logging
from pathlib import Path
import sys


def test_runtime_fallback_reuses_placeholder(monkeypatch, caplog):
    runtime_path = Path(__file__).resolve().parents[2] / "ui" / "lazy" / "runtime.py"
    spec = importlib.util.spec_from_file_location("ui.lazy.runtime_testcase", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    spec.loader.exec_module(runtime)

    caplog.set_level(logging.INFO)
    monkeypatch.setattr(runtime, "log_default_telemetry", lambda **_: None)
    monkeypatch.setattr(runtime, "_get_persistent_placeholder", lambda _fragment_id: "PERSISTENT")

    placeholder = runtime._handle_fragment_fallback(
        "frag123", context_ready=False, scope="global"
    )

    assert placeholder == "PERSISTENT"
    assert any(
        "fallback_reuse_persistent_container=True" in line
        for line in caplog.text.splitlines()
    )
