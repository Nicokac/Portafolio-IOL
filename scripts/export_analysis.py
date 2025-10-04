"""Utilities to export enriched screening analyses.

This helper script runs the deterministic screener stub and produces
analysis exports ready to share with stakeholders. The resulting
payload includes both the tabular data and the summary/notes metadata
computed by the service, ensuring parity with the Streamlit UI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from application.screener.opportunities import run_screener_stub


def _write_csv(df: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)


def _write_json(df: pd.DataFrame, notes: Sequence[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": df.attrs.get("summary", {}),
        "notes": list(notes),
        "rows": df.to_dict(orient="records"),
    }
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera una exportación enriquecida del screening basado en el "
            "stub determinista, incluyendo notas y métricas agregadas."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exports/opportunities_analysis.csv"),
        help=(
            "Ruta base donde se guardará la exportación. Cuando se usa el "
            "formato 'both', se reutiliza la misma ruta para el CSV y se "
            "genera un JSON paralelo con el mismo nombre."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("csv", "json", "both"),
        default="csv",
        help="Formato de salida a generar.",
    )
    parser.add_argument(
        "--include-technicals",
        action="store_true",
        help="Incluye columnas de indicadores técnicos (RSI, SMAs, etc.).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Limita la cantidad de filas exportadas tras aplicar los filtros.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Filtra por puntaje compuesto mínimo antes de exportar.",
    )
    parser.add_argument(
        "--sectors",
        nargs="*",
        default=None,
        help="Lista opcional de sectores a priorizar (usa los nombres visibles en la UI).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)

    df, notes = run_screener_stub(
        include_technicals=args.include_technicals,
        max_results=args.max_results,
        min_score_threshold=args.min_score,
        sectors=args.sectors,
    )

    output = args.output

    if args.format in {"csv", "both"}:
        _write_csv(df, output if args.format == "csv" else output)

    if args.format in {"json", "both"}:
        json_path = output if args.format == "json" else output.with_suffix(".json")
        _write_json(df, notes, json_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry-point
    main()
