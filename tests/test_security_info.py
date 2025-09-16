from pathlib import Path
import sys
from unittest.mock import MagicMock

sys.path.append(str(Path(__file__).resolve().parents[1]))

import ui.security_info as security_info


def test_security_info_contains_expected_text(monkeypatch):
    mock_markdown = MagicMock()
    monkeypatch.setattr(security_info.st, "markdown", mock_markdown)
    security_info.render_security_info()
    rendered = "\n".join(str(call.args[0]) for call in mock_markdown.call_args_list)
    assert "Fernet" in rendered
    assert "Streamlit Secrets" in rendered
