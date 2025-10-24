"""Herramienta de auditoría para validar escalas de bonos tras fixes BOPREAL."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from infrastructure.iol.client import IOLClient
from application.portfolio_service import PortfolioService, detect_bond_scale_anomalies
from services.data_fetch_service import PortfolioDataset, get_portfolio_data_fetch_service
from services.portfolio_view import PortfolioViewModelService
from domain.models import Controls


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Usar el dataset cacheado sin consultar a la API de InvertirOnline.",
    )
    parser.add_argument(
        "--baseline-view",
        type=Path,
        help="Ruta a la vista agregada del portafolio (v0.8.3) para calcular deltas.",
    )
    parser.add_argument(
        "--delta-output",
        type=Path,
        help="Archivo CSV donde persistir el reporte de diferencias por símbolo.",
    )
    return parser.parse_args()


@dataclass(slots=True)
class OfflineIOLClient:
    """Cliente mínimo que reutiliza un dataset cacheado para correr en modo offline."""

    dataset: PortfolioDataset

    def get_portfolio(self) -> Mapping[str, Any]:
        payload = self.dataset.raw_payload
        if isinstance(payload, Mapping):
            return payload
        return {"_cached": True, "positions": self.dataset.positions.to_dict(orient="records")}

    def get_quotes_bulk(self, items: Iterable[Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
        result: dict[tuple[str, str], Mapping[str, Any]] = {}
        quotes = getattr(self.dataset, "quotes", {}) or {}
        for raw in items or []:
            market: str | None = None
            symbol: str | None = None
            if isinstance(raw, Mapping):
                market = raw.get("market") or raw.get("mercado")
                symbol = raw.get("symbol") or raw.get("simbolo")
            elif isinstance(raw, (list, tuple)):
                if len(raw) >= 2:
                    market, symbol = raw[0], raw[1]
            else:
                market = getattr(raw, "market", getattr(raw, "mercado", None))
                symbol = getattr(raw, "symbol", getattr(raw, "simbolo", None))
            key = (str(market or "bcba").lower(), str(symbol or "").upper())
            quote = quotes.get(key)
            if isinstance(quote, Mapping):
                result[key] = dict(quote)
            else:
                result[key] = {}
        return result


def _load_cli(*, offline: bool, dataset: PortfolioDataset | None) -> Any:
    if offline:
        if dataset is None:
            raise RuntimeError("No hay dataset cacheado disponible para modo offline.")
        return OfflineIOLClient(dataset)

    user = os.environ["IOL_USERNAME"]
    password = os.environ["IOL_PASSWORD"]
    tokens_file = Path(os.environ.get("IOL_TOKENS_FILE", "tokens/iol_tokens.json"))
    return IOLClient(user=user, password=password, tokens_file=tokens_file)


def _load_baseline_view(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró la vista base en {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, Mapping):
            return pd.DataFrame(payload)
        raise ValueError("El archivo JSON no contiene un formato tabular soportado")
    raise ValueError(
        "Formato de archivo no soportado para baseline. Usa CSV, TSV, JSON o Parquet."
    )


def _build_symbol_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["simbolo", "valor_actual", "costo", "pl"])
    df_local = df.copy()
    if "simbolo" not in df_local.columns:
        raise KeyError("La vista del portafolio no contiene la columna 'simbolo'.")
    numeric_cols = [col for col in ("valor_actual", "costo", "pl") if col in df_local.columns]
    if not numeric_cols:
        raise KeyError("La vista del portafolio no incluye columnas numéricas para comparar.")
    aggregated = (
        df_local.groupby("simbolo", dropna=False)[numeric_cols]
        .sum(numeric_only=True)
        .reset_index()
        .fillna(0.0)
    )
    for column in numeric_cols:
        aggregated[column] = pd.to_numeric(aggregated[column], errors="coerce").fillna(0.0)
    return aggregated


def _diff_by_symbol(previous: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    previous_summary = _build_symbol_summary(previous).set_index("simbolo")
    current_summary = _build_symbol_summary(current).set_index("simbolo")
    merged = previous_summary.join(current_summary, how="outer", lsuffix="_prev", rsuffix="_curr")
    merged = merged.fillna(0.0)
    for column in ("valor_actual", "costo", "pl"):
        if f"{column}_curr" in merged.columns and f"{column}_prev" in merged.columns:
            merged[f"delta_{column}"] = merged[f"{column}_curr"] - merged[f"{column}_prev"]
    merged.index.name = "simbolo"
    return merged.reset_index()


def main() -> None:
    args = _parse_args()

    print("📦 Cargando servicios de portafolio...")
    psvc = PortfolioService()
    dataset_service = get_portfolio_data_fetch_service()

    dataset: PortfolioDataset | None
    _metadata: Any | None
    if args.offline:
        dataset, _metadata = dataset_service.peek_dataset()
        if dataset is None:
            raise RuntimeError(
                "No se encontró un dataset cacheado. Ejecutá el script online primero o recalculá las posiciones."
            )
        print("📁 Reutilizando dataset cacheado (modo offline).")
    else:
        print("🔑 Inicializando cliente IOL...")
        cli_temp = _load_cli(offline=False, dataset=None)
        print("⬇️ Descargando dataset desde IOL (o cache local)...")
        dataset, _metadata = dataset_service.get_dataset(cli_temp, psvc)
        dataset_hash = getattr(dataset, "dataset_hash", None)
        print(f"✅ Dataset actualizado (hash={dataset_hash})")
    cli = _load_cli(offline=args.offline, dataset=dataset)

    if dataset is None:
        raise RuntimeError("No fue posible obtener el dataset del portafolio.")

    print("🧮 Generando vista del portafolio...")
    controls = Controls()
    view_service = PortfolioViewModelService()
    snapshot = view_service.get_portfolio_view(
        df_pos=dataset.positions,
        controls=controls,
        cli=cli,
        psvc=psvc,
        dataset_hash=getattr(dataset, "dataset_hash", None),
        skip_invalidation=args.offline,
    )

    print("🔍 Ejecutando análisis de escala de bonos/letras...")
    report_df, total_impact = detect_bond_scale_anomalies(snapshot.df_view)

    print("\n📊 Resultado del análisis de escala:\n")
    if report_df.empty:
        print("Sin anomalías detectadas.")
    else:
        print(report_df.to_string(index=False))
    print(f"\n💰 Impacto total estimado: {total_impact:,.2f} ARS")

    if args.baseline_view:
        print("\n📈 Comparando contra baseline proporcionado...")
        baseline_df = _load_baseline_view(args.baseline_view)
        deltas = _diff_by_symbol(baseline_df, snapshot.df_view)
        if deltas.empty:
            print("No se pudieron calcular diferencias; revisá el dataset baseline.")
        else:
            ordered_cols = ["simbolo"]
            for column in ("valor_actual", "costo", "pl"):
                prev_col = f"{column}_prev"
                curr_col = f"{column}_curr"
                delta_col = f"delta_{column}"
                if prev_col in deltas.columns and curr_col in deltas.columns:
                    ordered_cols.extend([prev_col, curr_col])
                if delta_col in deltas.columns:
                    ordered_cols.append(delta_col)
            existing_cols = [col for col in ordered_cols if col in deltas.columns]
            ordered = deltas.loc[:, existing_cols]
            print(ordered.to_string(index=False))
            if args.delta_output:
                args.delta_output.parent.mkdir(parents=True, exist_ok=True)
                ordered.to_csv(args.delta_output, index=False)
                print(f"💾 Delta exportado en {args.delta_output}")


if __name__ == "__main__":
    main()
