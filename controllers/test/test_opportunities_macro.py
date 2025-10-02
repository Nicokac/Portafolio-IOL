from __future__ import annotations

from typing import Dict

import pandas as pd
import pytest

from controllers import opportunities
from infrastructure.macro import FredSeriesObservation


@pytest.fixture(autouse=True)
def _reset_macro_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    opportunities._macro_series_lookup.cache_clear()
    opportunities._macro_fallback_lookup.cache_clear()
    opportunities._get_macro_client.cache_clear()
    monkeypatch.setattr(opportunities, "record_macro_api_usage", lambda **_: None)


def test_enrich_with_macro_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"sector": ["Technology"]})

    monkeypatch.setattr(opportunities, "macro_api_provider", "fred")
    monkeypatch.setattr(opportunities, "fred_api_key", None)
    monkeypatch.setattr(opportunities, "fred_sector_series", {})
    monkeypatch.setattr(opportunities, "world_bank_sector_series", {})
    monkeypatch.setattr(
        opportunities,
        "macro_sector_fallback",
        {"Technology": {"value": 2.5, "as_of": "2023-08-01"}},
    )

    notes, metrics = opportunities._enrich_with_macro_context(df)

    assert metrics["macro_source"] == "fallback"
    assert "2.50" in df[opportunities._MACRO_COLUMN].iloc[0]
    assert any("fallback" in note for note in notes)
    attempts = metrics["macro_provider_attempts"]
    assert attempts[0]["provider"] == "fred"
    assert attempts[0]["status"] == "disabled"
    assert attempts[-1]["provider"] == "fallback"
    assert attempts[-1]["status"] == "success"


def test_enrich_with_macro_uses_fred(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"sector": ["Finance"], "ticker": ["FIN"]})

    class DummyClient:
        def get_latest_observations(self, mapping: Dict[str, str]) -> Dict[str, FredSeriesObservation]:
            assert mapping == {"Finance": "SERIES"}
            return {"Finance": FredSeriesObservation(series_id="SERIES", value=1.75, as_of="2023-09-01")}

    monkeypatch.setattr(opportunities, "macro_api_provider", "fred")
    monkeypatch.setattr(opportunities, "fred_api_key", "dummy")
    monkeypatch.setattr(opportunities, "fred_sector_series", {"Finance": "SERIES"})
    monkeypatch.setattr(opportunities, "macro_sector_fallback", {})
    monkeypatch.setattr(opportunities, "world_bank_sector_series", {})

    def _fake_client(provider: str):
        assert provider == "fred"
        return DummyClient()

    monkeypatch.setattr(opportunities, "_get_macro_client", _fake_client)

    notes, metrics = opportunities._enrich_with_macro_context(df)

    assert metrics["macro_source"] == "fred"
    assert metrics["macro_sector_coverage"] == 1
    assert "1.75" in df[opportunities._MACRO_COLUMN].iloc[0]
    assert any("FRED" in note for note in notes)
    attempts = metrics["macro_provider_attempts"]
    assert len(attempts) == 1
    assert attempts[0]["provider"] == "fred"
    assert attempts[0]["status"] == "success"


def test_enrich_with_macro_uses_secondary_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"sector": ["Energy"]})

    class DummyWorldBankClient:
        def get_latest_observations(self, mapping: Dict[str, str]) -> Dict[str, FredSeriesObservation]:
            assert mapping == {"Energy": "WB_SERIES"}
            return {
                "Energy": FredSeriesObservation(
                    series_id="WB_SERIES", value=3.4, as_of="2023-07-01"
                )
            }

    monkeypatch.setattr(opportunities, "macro_api_provider", "worldbank")
    monkeypatch.setattr(opportunities, "fred_api_key", None)
    monkeypatch.setattr(opportunities, "fred_sector_series", {})
    monkeypatch.setattr(opportunities, "world_bank_sector_series", {"Energy": "WB_SERIES"})
    monkeypatch.setattr(opportunities, "macro_sector_fallback", {})

    def _fake_client(provider: str):
        if provider == "fred":
            return None
        assert provider == "worldbank"
        return DummyWorldBankClient()

    monkeypatch.setattr(opportunities, "_get_macro_client", _fake_client)

    notes, metrics = opportunities._enrich_with_macro_context(df)

    assert metrics["macro_source"] == "worldbank"
    assert "3.40" in df[opportunities._MACRO_COLUMN].iloc[0]
    assert any("World Bank" in note for note in notes)
    attempts = metrics["macro_provider_attempts"]
    assert [attempt["status"] for attempt in attempts] == ["disabled", "success"]
    assert attempts[0]["provider"] == "fred"
    assert attempts[1]["provider"] == "worldbank"
