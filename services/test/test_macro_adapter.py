from types import SimpleNamespace

import pytest

from infrastructure.macro import MacroAPIError, MacroSeriesObservation
from services.macro_adapter import MacroAdapter


def _build_settings(**overrides):
    base = {
        "macro_api_provider": "fred,worldbank",
        "fred_sector_series": {},
        "world_bank_sector_series": {},
        "macro_sector_fallback": {},
        "fred_api_key": "dummy",  # default valid key for factories
        "fred_api_base_url": "https://api.stlouisfed.org/fred",
        "fred_api_rate_limit_per_minute": 120,
        "world_bank_api_key": None,
        "world_bank_api_base_url": "https://api.worldbank.org/v2",
        "world_bank_api_rate_limit_per_minute": 60,
        "USER_AGENT": "TestAgent/1.0",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_macro_adapter_uses_primary_provider_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _build_settings(
        fred_sector_series={"Finance": "SERIES"},
        macro_sector_fallback={},
    )

    class DummyFredClient:
        def get_latest_observations(self, mapping):  # type: ignore[no-untyped-def]
            assert mapping == {"Finance": "SERIES"}
            return {"Finance": MacroSeriesObservation(series_id="SERIES", value=1.23, as_of="2023-09-01")}

    adapter = MacroAdapter(
        settings=settings,
        client_factories={"fred": lambda: DummyFredClient()},
        timer=lambda: 0.0,
        clock=lambda: 10.0,
    )

    result = adapter.fetch(["Finance"])

    assert result.provider == "fred"
    assert result.entries["Finance"]["value"] == pytest.approx(1.23)
    assert result.entries["Finance"]["as_of"] == "2023-09-01"
    assert len(result.attempts) == 1
    assert result.attempts[0]["status"] == "success"


def test_macro_adapter_falls_back_to_secondary_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _build_settings(
        fred_api_key=None,
        fred_sector_series={},
        world_bank_sector_series={"Energy": "WB"},
    )

    class DummyWorldBankClient:
        def get_latest_observations(self, mapping):  # type: ignore[no-untyped-def]
            assert mapping == {"Energy": "WB"}
            return {"Energy": MacroSeriesObservation(series_id="WB", value=3.4, as_of="2023-07-01")}

    adapter = MacroAdapter(
        settings=settings,
        client_factories={"worldbank": lambda: DummyWorldBankClient()},
        timer=lambda: 0.0,
        clock=lambda: 20.0,
    )

    result = adapter.fetch(["Energy"])

    assert result.provider == "worldbank"
    assert len(result.attempts) == 2
    assert result.attempts[0]["status"] == "disabled"
    assert result.attempts[1]["status"] == "success"
    assert result.entries["Energy"]["value"] == pytest.approx(3.4)
    assert "World Bank" in result.attempts[1]["provider_label"]


def test_macro_adapter_uses_static_fallback_when_all_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _build_settings(
        fred_sector_series={"Energy": "SERIES"},
        macro_sector_fallback={"Energy": {"value": 2.5, "as_of": "2023-01-01"}},
    )

    class FailingClient:
        def get_latest_observations(self, mapping):  # type: ignore[no-untyped-def]
            raise MacroAPIError("boom")

    adapter = MacroAdapter(
        settings=settings,
        client_factories={"fred": lambda: FailingClient()},
        timer=lambda: 0.0,
        clock=lambda: 30.0,
    )

    result = adapter.fetch(["Energy"])

    assert result.provider is None
    assert not result.entries
    assert result.fallback_entries["Energy"]["value"] == pytest.approx(2.5)
    assert result.attempts[-1]["provider"] == "fallback"
    assert result.attempts[-1]["status"] == "success"
    assert result.last_reason == "boom"
