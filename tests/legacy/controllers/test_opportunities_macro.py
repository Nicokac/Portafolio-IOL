from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.skip(reason="Escenarios cubiertos por tests/services/test_macro_adapter.py")

# NOTE: Estas pruebas viven en la suite legacy mientras migramos sus escenarios
# a `tests/controllers/test_opportunities_controller.py`. Mantenerlas permite
# auditar la lógica previa hasta que consolidemos los duplicados y quitemos el
# uso del helper `_enrich_with_macro_context` en tests nuevos.
from controllers import opportunities
from services.macro_adapter import MacroFetchResult


@pytest.fixture(autouse=True)
def _reset_macro_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(opportunities, "record_macro_api_usage", lambda **_: None)


def test_enrich_with_macro_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"sector": ["Technology"]})

    fallback_entries = {"Technology": {"value": 2.5, "as_of": "2023-08-01"}}

    def _adapter_factory():
        return type(
            "_DummyAdapter",
            (),
            {
                "fetch": staticmethod(
                    lambda sectors: MacroFetchResult(
                        provider=None,
                        provider_label=None,
                        entries={},
                        attempts=[
                            {
                                "provider": "fred",
                                "provider_label": "FRED",
                                "status": "disabled",
                                "detail": "FRED sin credenciales configuradas",
                                "ts": 1.0,
                            },
                            {
                                "provider": "fallback",
                                "provider_label": "Fallback",
                                "status": "success",
                                "fallback": True,
                                "ts": 2.0,
                            },
                        ],
                        notes=["FRED no disponible: FRED sin credenciales configuradas"],
                        missing_series=[],
                        fallback_entries=fallback_entries,
                        last_reason="FRED sin credenciales configuradas",
                    )
                )
            },
        )()

    monkeypatch.setattr(opportunities, "MacroAdapter", lambda: _adapter_factory())

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

    def _adapter_factory():
        return type(
            "_DummyAdapter",
            (),
            {
                "fetch": staticmethod(
                    lambda sectors: MacroFetchResult(
                        provider="fred",
                        provider_label="FRED",
                        entries={
                            "Finance": {"value": 1.75, "as_of": "2023-09-01"}
                        },
                        attempts=[
                            {
                                "provider": "fred",
                                "provider_label": "FRED",
                                "status": "success",
                                "elapsed_ms": 12.0,
                                "ts": 2.0,
                            }
                        ],
                        notes=[],
                        missing_series=[],
                        fallback_entries={},
                        last_reason=None,
                    )
                )
            },
        )()

    monkeypatch.setattr(opportunities, "MacroAdapter", lambda: _adapter_factory())

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

    def _adapter_factory():
        return type(
            "_DummyAdapter",
            (),
            {
                "fetch": staticmethod(
                    lambda sectors: MacroFetchResult(
                        provider="worldbank",
                        provider_label="World Bank",
                        entries={
                            "Energy": {"value": 3.4, "as_of": "2023-07-01"}
                        },
                        attempts=[
                            {
                                "provider": "fred",
                                "provider_label": "FRED",
                                "status": "disabled",
                                "detail": "FRED sin credenciales configuradas",
                                "ts": 3.0,
                            },
                            {
                                "provider": "worldbank",
                                "provider_label": "World Bank",
                                "status": "success",
                                "elapsed_ms": 15.0,
                                "ts": 4.0,
                            },
                        ],
                        notes=["FRED no disponible: FRED sin credenciales configuradas"],
                        missing_series=[],
                        fallback_entries={},
                        last_reason=None,
                    )
                )
            },
        )()

    monkeypatch.setattr(opportunities, "MacroAdapter", lambda: _adapter_factory())

    notes, metrics = opportunities._enrich_with_macro_context(df)

    assert metrics["macro_source"] == "worldbank"
    assert "3.40" in df[opportunities._MACRO_COLUMN].iloc[0]
    assert any("World Bank" in note for note in notes)
    attempts = metrics["macro_provider_attempts"]
    assert [attempt["status"] for attempt in attempts] == ["disabled", "success"]
    assert attempts[0]["provider"] == "fred"
    assert attempts[1]["provider"] == "worldbank"
