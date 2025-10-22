#!/usr/bin/env python3
"""Generador de reportes enriquecidos a partir de snapshots persistidos."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from shared.portfolio_export import (  # noqa: E402
    CHART_SPECS,
    METRIC_SPECS,
    PortfolioSnapshotExport,
    compute_kpis,
    create_csv_bundle,
    create_excel_workbook,
    write_tables_to_directory,
)

LOGGER = logging.getLogger(__name__)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exporta KPIs, rankings y gráficos del portafolio en CSV/Excel a partir de snapshots serializados (JSON)."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Directorio o archivo JSON con snapshots persistidos.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("exports"),
        help=("Directorio raíz donde se almacenarán los reportes (por defecto ./exports)."),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Lista de KPIs a incluir (usar --metrics help para ver opciones).",
    )
    parser.add_argument(
        "--charts",
        nargs="+",
        help=("Lista de gráficos a embeber en el Excel (usar --charts help para ver opciones)."),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Cantidad máxima de filas por ranking (default: 10).",
    )
    parser.add_argument(
        "--formats",
        "--format",
        choices=["csv", "excel", "both"],
        default="both",
        help="Formartos a generar (default: both).",
    )
    parser.add_argument(
        "--no-history",
        dest="include_history",
        action="store_false",
        help="Omitir la hoja/tables de historial de totales.",
    )
    parser.add_argument(
        "--no-rankings",
        dest="include_rankings",
        action="store_false",
        help="Omitir rankings en CSV/Excel.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Habilita logs informativos durante la ejecución.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _available(options: Sequence) -> dict[str, str]:
    return {spec.key: getattr(spec, "label", getattr(spec, "title", spec.key)) for spec in options}


def _resolve_metric_keys(raw: Sequence[str] | None) -> list[str]:
    lookup = _available(METRIC_SPECS)
    if raw is None:
        return [spec.key for spec in METRIC_SPECS[:5]]
    if len(raw) == 1 and raw[0].lower() == "help":
        msg = "\n".join(f"- {key}: {label}" for key, label in lookup.items())
        raise SystemExit(f"Métricas disponibles:\n{msg}")
    unknown = [key for key in raw if key not in lookup]
    if unknown:
        LOGGER.warning("Ignorando métricas desconocidas: %s", ", ".join(unknown))
    resolved = [key for key in raw if key in lookup]
    return resolved or [spec.key for spec in METRIC_SPECS[:5]]


def _resolve_chart_keys(raw: Sequence[str] | None) -> list[str]:
    lookup = _available(CHART_SPECS)
    if raw is None:
        return [spec.key for spec in CHART_SPECS]
    if len(raw) == 1 and raw[0].lower() == "help":
        msg = "\n".join(f"- {key}: {label}" for key, label in lookup.items())
        raise SystemExit(f"Gráficos disponibles:\n{msg}")
    unknown = [key for key in raw if key not in lookup]
    if unknown:
        LOGGER.warning("Ignorando gráficos desconocidos: %s", ", ".join(unknown))
    return [key for key in raw if key in lookup]


def _iter_snapshot_paths(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.glob("*.json") if p.is_file())
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"No se encontró {path}")


def _load_snapshot(path: Path) -> PortfolioSnapshotExport | None:
    try:
        snapshot = PortfolioSnapshotExport.from_path(path)
    except Exception as exc:  # pragma: no cover - robust frente a datos malformados
        LOGGER.error("No se pudo leer %s: %s", path, exc)
        return None
    return snapshot


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    metric_keys = _resolve_metric_keys(args.metrics)
    chart_keys = _resolve_chart_keys(args.charts)
    include_history = getattr(args, "include_history", True)
    include_rankings = getattr(args, "include_rankings", True)
    limit = max(1, args.limit)

    csv_enabled = args.formats in {"csv", "both"}
    excel_enabled = args.formats in {"excel", "both"}

    paths = _iter_snapshot_paths(args.input)
    if not paths:
        LOGGER.error("No se encontraron snapshots en %s", args.input)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []

    for path in paths:
        snapshot = _load_snapshot(path)
        if snapshot is None:
            continue
        name = snapshot.name or path.stem
        LOGGER.info("Procesando snapshot %s", name)
        target_dir = args.output / name
        target_dir.mkdir(parents=True, exist_ok=True)

        if csv_enabled:
            write_tables_to_directory(
                snapshot,
                target_dir,
                metric_keys=metric_keys,
                include_rankings=include_rankings,
                include_history=include_history,
                limit=limit,
            )
            zip_bytes = create_csv_bundle(
                snapshot,
                metric_keys=metric_keys,
                include_rankings=include_rankings,
                include_history=include_history,
                limit=limit,
            )
            (target_dir / "analysis.zip").write_bytes(zip_bytes)
        if excel_enabled:
            excel_bytes = create_excel_workbook(
                snapshot,
                metric_keys=metric_keys,
                chart_keys=chart_keys,
                include_rankings=include_rankings,
                include_history=include_history,
                limit=limit,
            )
            (target_dir / "analysis.xlsx").write_bytes(excel_bytes)

        kpis = compute_kpis(snapshot, metric_keys=metric_keys)
        if not kpis.empty:
            row = {
                "snapshot": name,
                "generated_at": (snapshot.generated_at.isoformat() if snapshot.generated_at else ""),
            }
            for _, metric in kpis.iterrows():
                row[metric["metric"]] = metric.get("raw_value")
            summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = args.output / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        LOGGER.info("Resumen guardado en %s", summary_path)

    LOGGER.info("Exportación completada (%d snapshots)", len(summary_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
