from __future__ import annotations

import pytest


def test_offline_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    with pytest.raises(RuntimeError):
        requests.get("https://api.invertironline.com")
