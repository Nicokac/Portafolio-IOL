from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from services import health  # noqa: E402


def test_quote_provider_summary_handles_mixed_data(monkeypatch):
    fake_state: dict[str, dict] = {}
    monkeypatch.setattr(health, "st", SimpleNamespace(session_state=fake_state))

    store = health._store()
    store["quote_providers"] = {
        "alpha": {
            "count": 3,
            "ok_count": 2,
            "stale_count": 1,
            "elapsed_history": [0.1, 0.2, 0.3],
        },
        "beta": {
            "count": 5,
            "stale_count": 0,
        },
        "gamma": {
            "count": 4,
            "ok_count": 0,
        },
    }
    store[health._QUOTE_RATE_LIMIT_KEY] = {
        "alpha": {
            "count": 1,
            "wait_total": 0.1,
            "wait_last": 0.05,
        }
    }

    summary = health.get_health_metrics()["quote_providers"]

    assert summary["total"] == 12
    assert summary["ok_total"] == 2

    providers = {entry["provider"]: entry for entry in summary["providers"]}
    alpha = providers["alpha"]
    assert alpha["ok_count"] == 2
    assert alpha["ok_ratio"] == pytest.approx(2 / 3)
    assert "rate_limit_count" in alpha

