from __future__ import annotations

import pytest

from ui.utils.formatters import format_asset_type


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ACCIONES", "Acción"),
        ("accion", "Acción"),
        ("CEDEAR", "Cedear"),
        ("cedears", "Cedear"),
        ("LETRA", "Letra"),
        ("letras", "Letra"),
        ("BONO", "Bono"),
        ("titulospublicos", "Bono"),
        ("fondoComunDeInversion", "FCI"),
        ("fci", "FCI"),
    ],
)
def test_format_asset_type_known_aliases(raw: str, expected: str) -> None:
    assert format_asset_type(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", float("nan"), "<NA>"])
def test_format_asset_type_missing(raw: str | None) -> None:
    assert format_asset_type(raw) == "N/D"


def test_format_asset_type_unknown_returns_original() -> None:
    assert format_asset_type("OBLIGACION") == "OBLIGACION"
