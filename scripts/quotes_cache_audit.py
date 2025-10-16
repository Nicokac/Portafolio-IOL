"""Audit helper for quotes_refresh telemetry exported as JSONL logs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, List


@dataclass
class QuoteBatchEvent:
    group: str
    symbols: List[str]
    duration_s: float
    mode: str | None = None
    stale: bool | None = None
    refresh_scheduled: bool | None = None
    background: bool | None = None


@dataclass
class RefreshSummary:
    count: int
    fresh: int
    stale: int
    errors: int
    elapsed_s: float

    @property
    def hit_ratio(self) -> float:
        return (self.fresh / self.count) * 100 if self.count else 0.0

    @property
    def stale_ratio(self) -> float:
        return (self.stale / self.count) * 100 if self.count else 0.0


@dataclass
class AuditReport:
    summary: RefreshSummary
    batches: List[QuoteBatchEvent]

    def quotes_refresh_total_s(self) -> float:
        return self.summary.elapsed_s

    def avg_batch_time_ms(self) -> float:
        if not self.batches:
            return 0.0
        return mean(batch.duration_s for batch in self.batches) * 1000.0

    def max_batch_time_ms(self) -> float:
        if not self.batches:
            return 0.0
        return max(batch.duration_s for batch in self.batches) * 1000.0

    def quotes_hit_ratio(self) -> float:
        return self.summary.hit_ratio

    def stale_ratio(self) -> float:
        return self.summary.stale_ratio

    def anomalous_batches(self, threshold_s: float = 1.0) -> list[QuoteBatchEvent]:
        return [batch for batch in self.batches if batch.duration_s > threshold_s]

    def served_modes(self) -> dict[str, int]:
        modes: dict[str, int] = {}
        for batch in self.batches:
            if batch.mode:
                modes[batch.mode] = modes.get(batch.mode, 0) + 1
        return modes


@dataclass
class PortfolioCacheSnapshot:
    render_total_s: float
    hit_ratio: float
    miss_count: int
    hits: int
    render_invocations: int
    fingerprint_invalidations: dict[str, int]
    cache_miss_reasons: dict[str, int]
    recent_misses: list[dict]
    recent_invalidations: list[dict]

    def total_invalidations(self) -> int:
        return sum(self.fingerprint_invalidations.values())

    def invalidation_breakdown(self) -> str:
        if not self.fingerprint_invalidations:
            return "Sin invalidaciones registradas"
        return ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(self.fingerprint_invalidations.items())
        )

    def miss_breakdown(self) -> str:
        if not self.cache_miss_reasons:
            return "Sin misses registrados"
        return ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(self.cache_miss_reasons.items())
        )

    def unnecessary_misses(self) -> int:
        return int(self.cache_miss_reasons.get("unchanged_fingerprint", 0) or 0)

    def recent_unnecessary_misses(self) -> list[dict]:
        return [
            miss
            for miss in self.recent_misses
            if miss.get("reason") == "unchanged_fingerprint"
        ]


def _load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _parse_batches(events: Iterable[dict]) -> list[QuoteBatchEvent]:
    batches: list[QuoteBatchEvent] = []
    for event in events:
        if event.get("event") not in {"quotes_batch_refreshed", "quotes_batch_served"}:
            continue
        payload = event.get("quotes_batch") or {}
        group = str(payload.get("group", "")).strip()
        symbols = [str(sym) for sym in payload.get("symbols", [])]
        duration = float(payload.get("duration_s", 0.0))
        mode = payload.get("mode")
        stale = payload.get("stale")
        refresh_scheduled = payload.get("refresh_scheduled")
        background = payload.get("background")
        if event.get("event") == "quotes_batch_served":
            existing = next((b for b in batches if b.symbols == symbols and b.group == group), None)
            if existing:
                existing.mode = str(mode) if mode is not None else existing.mode
                existing.stale = bool(stale) if stale is not None else existing.stale
                existing.refresh_scheduled = (
                    bool(refresh_scheduled)
                    if refresh_scheduled is not None
                    else existing.refresh_scheduled
                )
                continue
        batches.append(
            QuoteBatchEvent(
                group=group,
                symbols=symbols,
                duration_s=duration,
                mode=str(mode) if mode is not None else None,
                stale=bool(stale) if stale is not None else None,
                refresh_scheduled=(bool(refresh_scheduled) if refresh_scheduled is not None else None),
                background=bool(background) if background is not None else None,
            )
        )
    return batches


def _parse_summary(events: Iterable[dict]) -> RefreshSummary:
    for event in events:
        if event.get("event") == "quotes_refresh_summary":
            stats = event.get("stats") or {}
            return RefreshSummary(
                count=int(stats.get("count", 0) or 0),
                fresh=int(stats.get("fresh", 0) or 0),
                stale=int(stats.get("stale", 0) or 0),
                errors=int(stats.get("errors", 0) or 0),
                elapsed_s=float(event.get("elapsed_s", 0.0)),
            )
    raise ValueError("No se encontró evento quotes_refresh_summary en los logs")


def load_report(path: Path) -> AuditReport:
    events = _load_events(path)
    summary = _parse_summary(events)
    batches = _parse_batches(events)
    return AuditReport(summary=summary, batches=batches)


def load_portfolio_metrics(path: Path) -> PortfolioCacheSnapshot:
    data = json.loads(path.read_text(encoding="utf-8"))
    invalidations_raw = data.get("fingerprint_invalidations") or {}
    miss_reasons_raw = data.get("cache_miss_reasons") or {}
    recent_misses = list(data.get("recent_misses") or [])
    recent_invalidations = list(data.get("recent_invalidations") or [])
    return PortfolioCacheSnapshot(
        render_total_s=float(data.get("portfolio_view_render_s", 0.0) or 0.0),
        hit_ratio=float(data.get("portfolio_cache_hit_ratio", 0.0) or 0.0),
        miss_count=int(data.get("portfolio_cache_miss_count", 0) or 0),
        hits=int(data.get("hits", 0) or 0),
        render_invocations=int(data.get("render_invocations", 0) or 0),
        fingerprint_invalidations={
            str(reason): int(count)
            for reason, count in invalidations_raw.items()
            if count is not None
        },
        cache_miss_reasons={
            str(reason): int(count)
            for reason, count in miss_reasons_raw.items()
            if count is not None
        },
        recent_misses=[
            miss for miss in recent_misses if isinstance(miss, dict)
        ],
        recent_invalidations=[
            entry for entry in recent_invalidations if isinstance(entry, dict)
        ],
    )


def export_metrics(
    report: AuditReport, portfolio: PortfolioCacheSnapshot, output: Path
) -> None:
    lines = [
        "metric,value,notes\n",
        f"quotes_refresh_total_s,{report.quotes_refresh_total_s():.2f},",
        "Duración total de quotes_refresh según telemetría\n",
        f"avg_batch_time_ms,{report.avg_batch_time_ms():.1f},",
        "Promedio de duración por sublote (ms)\n",
        f"max_batch_time_ms,{report.max_batch_time_ms():.1f},",
        "Sub-lote más lento observado (ms)\n",
        f"quotes_hit_ratio,{report.quotes_hit_ratio():.2f},",
        "Porcentaje de respuestas fresh sobre el total\n",
        f"stale_ratio,{report.stale_ratio():.2f},",
        "Porcentaje de respuestas servidas en modo stale\n",
        f"portfolio_view_render_s,{portfolio.render_total_s:.2f},",
        "Tiempo total invertido en portfolio_view.render (s)\n",
        f"portfolio_cache_hit_ratio,{portfolio.hit_ratio:.2f},",
        "Porcentaje de hits del memoizador del portafolio\n",
        f"portfolio_cache_miss_count,{portfolio.miss_count},",
        "Cantidad de recomputos del snapshot en el muestreo\n",
        f"fingerprint_invalidations,{portfolio.total_invalidations()},",
        f"Detalle: {portfolio.invalidation_breakdown()}\n",
    ]
    output.write_text("".join(lines), encoding="utf-8")


def render_quotes_cli(report: AuditReport) -> str:
    modes = report.served_modes()
    anomalous = report.anomalous_batches()
    lines = [
        "Resumen quotes_refresh:",
        f"- Duración total: {report.quotes_refresh_total_s():.2f} s",
        f"- Promedio sublotes: {report.avg_batch_time_ms():.1f} ms",
        f"- Máximo sublote: {report.max_batch_time_ms():.1f} ms",
        f"- Hit ratio: {report.quotes_hit_ratio():.2f}%",
        f"- Stale ratio: {report.stale_ratio():.2f}%",
        f"- Modos servidos: {modes}",
    ]
    if anomalous:
        lines.append("Sub-lotes >1s detectados:")
        for batch in anomalous:
            lines.append(
                f"  * {batch.group} ({', '.join(batch.symbols)}) -> {batch.duration_s:.2f}s"
            )
    else:
        lines.append("No se detectaron sub-lotes por encima de 1 segundo")
    return "\n".join(lines)


def render_portfolio_cli(metrics: PortfolioCacheSnapshot) -> str:
    lines = [
        "Resumen portfolio_view.render:",
        f"- Tiempo total: {metrics.render_total_s:.2f} s",
        f"- Invocaciones: {metrics.render_invocations} (hits={metrics.hits}, misses={metrics.miss_count})",
        f"- Hit ratio: {metrics.hit_ratio:.2f}%",
        f"- Razones de miss: {metrics.miss_breakdown()}",
        f"- Invalidaciones: {metrics.invalidation_breakdown()}",
    ]
    unnecessary = metrics.unnecessary_misses()
    if unnecessary:
        lines.append(
            f"Misses sin cambios de fingerprint detectados: {unnecessary}"
        )
        recent = metrics.recent_unnecessary_misses()
        if recent:
            lines.append("Últimos misses sin cambios registrados:")
            for miss in recent[:5]:
                apply_ms = float(miss.get("apply_elapsed") or 0.0)
                totals_ms = float(miss.get("totals_elapsed") or 0.0)
                render_ms = float(miss.get("render_elapsed") or 0.0)
                lines.append(
                    "  * "
                    f"apply={apply_ms:.3f}s totals={totals_ms:.3f}s render={render_ms:.3f}s"
                )
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audita logs de quotes_refresh")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/fixtures/telemetry/quotes_refresh_logs.jsonl"),
        help="Archivo JSONL con los eventos instrumentados",
    )
    parser.add_argument(
        "--portfolio-input",
        type=Path,
        default=Path("docs/fixtures/telemetry/portfolio_view_cache.json"),
        help="Archivo JSON con métricas de portfolio_view_cache",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("performance_metrics_9.csv"),
        help="Ruta de salida para el CSV de métricas",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = load_report(args.input)
    portfolio_metrics = load_portfolio_metrics(args.portfolio_input)
    export_metrics(report, portfolio_metrics, args.output)
    print(render_quotes_cli(report))
    print()
    print(render_portfolio_cli(portfolio_metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
