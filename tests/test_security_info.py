from pathlib import Path
import sys
import types
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.version import __version__


def test_security_info_contains_expected_text():
    streamlit_stub = types.SimpleNamespace(markdown=MagicMock())
    sys.modules["streamlit"] = streamlit_stub

    spec_path = Path(__file__).resolve().parents[1] / "ui" / "security_info.py"
    spec = types.ModuleType("security_info_spec")
    exec(spec_path.read_text(encoding="utf-8"), spec.__dict__)
    security_info = spec

    security_info.render_security_info()
    rendered = "\n".join(str(call.args[0]) for call in streamlit_stub.markdown.call_args_list)
    assert "Fernet" in rendered
    assert "Streamlit Secrets" in rendered
    assert __version__ in rendered
