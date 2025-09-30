from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from shared.ui import notes


@pytest.mark.parametrize(
    "note, expected",
    [
        ("⚠️ Datos simulados (Yahoo no disponible)", ":warning: **Datos simulados (Yahoo no disponible)**"),
        (
            "Se recortaron los resultados a los 10 mejores según el score compuesto.",
            "**Se recortaron los resultados a los 10 mejores según el score compuesto.**",
        ),
        (
            "ℹ️ Recuerda validar con fuentes oficiales",
            ":information_source: Recuerda validar con fuentes oficiales",
        ),
        ("✅ Operación exitosa", ":white_check_mark: **Operación exitosa**"),
        ("❌ Error en backend", ":x: **Error en backend**"),
        ("Considerar diversificación adicional.", "Considerar diversificación adicional."),
        ("", ""),
    ],
)
def test_format_note_returns_expected_rendering(note: str, expected: str) -> None:
    assert notes.format_note(note) == expected


def test_classify_note_detects_keyword_based_warning() -> None:
    note = "Se recortaron los resultados a los máximos solicitados."
    severity, content, matched_by_prefix = notes.classify_note(note)
    assert severity == "warning"
    assert content == note
    assert matched_by_prefix is False


def test_constants_exposed_for_reuse() -> None:
    assert "warning" in notes.NOTE_SEVERITIES
    warning_rendering = notes.NOTE_RENDERING["warning"]
    assert warning_rendering["icon"] == ":warning:"
    assert warning_rendering["emphasize"] is True
