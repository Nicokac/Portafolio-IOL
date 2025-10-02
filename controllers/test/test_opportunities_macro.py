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
    monkeypatch.setattr(
        opportunities,
        "macro_sector_fallback",
        {"Technology": {"value": 2.5, "as_of": "2023-08-01"}},
    )

    notes, metrics = opportunities._enrich_with_macro_context(df)

    assert metrics["macro_source"] == "fallback"
    assert "2.50" in df[opportunities._MACRO_COLUMN].iloc[0]
    assert any("fallback" in note for note in notes)


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
    monkeypatch.setattr(opportunities, "_get_macro_client", lambda: DummyClient())

    notes, metrics = opportunities._enrich_with_macro_context(df)

    assert metrics["macro_source"] == "fred"
    assert metrics["macro_sector_coverage"] == 1
    assert "1.75" in df[opportunities._MACRO_COLUMN].iloc[0]
    assert any("FRED" in note for note in notes)
