"""Recommendation service for portfolio extensions.

The service analyses the current portfolio composition (type, sector, currency
and risk buckets) and selects complementary assets from the opportunity
universe. It exposes a single entry point ``recommend`` that distributes a user
provided amount across the best five candidates according to the selected
mode.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

SUPPORTED_MODES: tuple[str, ...] = ("diversify", "max_return", "low_risk")
MIN_ALLOCATION: float = 0.10
MAX_ALLOCATION: float = 0.40
LOW_RISK_SECTORS: frozenset[str] = frozenset(
    {
        "Consumer Defensive",
        "Utilities",
        "Healthcare",
        "Real Estate",
    }
)
HIGH_RISK_SECTORS: frozenset[str] = frozenset(
    {
        "Technology",
        "Communication Services",
        "Consumer Cyclical",
        "Energy",
        "Materials",
    }
)


@dataclass(frozen=True)
class Recommendation:
    """Internal container used to build the final recommendation frame."""

    symbol: str
    score: float
    expected_return: float
    rationale: str
    sector: str
    currency: str
    tipo: str
    is_existing: bool
    max_allocation: float


def _normalise_amount(amount: float) -> float:
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(value):
        return 0.0
    return max(value, 0.0)


class RecommendationService:
    """Generate portfolio extension suggestions based on current exposure."""

    def __init__(
        self,
        *,
        portfolio_df: pd.DataFrame,
        opportunities_df: pd.DataFrame | None = None,
        risk_metrics_df: pd.DataFrame | None = None,
        fundamentals_df: pd.DataFrame | None = None,
    ) -> None:
        self._portfolio_df = portfolio_df.copy() if isinstance(portfolio_df, pd.DataFrame) else pd.DataFrame()
        self._opportunities_df = opportunities_df.copy() if isinstance(opportunities_df, pd.DataFrame) else pd.DataFrame()
        self._risk_metrics_df = (
            risk_metrics_df.copy() if isinstance(risk_metrics_df, pd.DataFrame) else pd.DataFrame()
        )
        self._fundamentals_df = fundamentals_df.copy() if isinstance(fundamentals_df, pd.DataFrame) else pd.DataFrame()
        self._analysis: dict[str, object] | None = None

        if not self._portfolio_df.empty:
            self._portfolio_df = self._portfolio_df.copy()
            self._portfolio_df.columns = [str(col) for col in self._portfolio_df.columns]
            if "valor_actual" not in self._portfolio_df.columns:
                # Fallback to costo when valuation column is missing.
                if "costo" in self._portfolio_df.columns:
                    self._portfolio_df["valor_actual"] = pd.to_numeric(
                        self._portfolio_df["costo"], errors="coerce"
                    )
                else:
                    self._portfolio_df["valor_actual"] = 0.0
            self._portfolio_df["valor_actual"] = pd.to_numeric(
                self._portfolio_df["valor_actual"], errors="coerce"
            ).fillna(0.0)

        if not self._risk_metrics_df.empty:
            self._risk_metrics_df = self._risk_metrics_df.copy()
            self._risk_metrics_df.columns = [str(col) for col in self._risk_metrics_df.columns]
            if "simbolo" in self._risk_metrics_df.columns:
                self._risk_metrics_df["simbolo"] = (
                    self._risk_metrics_df["simbolo"].astype("string").str.upper()
                )

        if not self._opportunities_df.empty:
            self._opportunities_df = self._opportunities_df.copy()
            self._opportunities_df.columns = [str(col) for col in self._opportunities_df.columns]

        if not self._fundamentals_df.empty:
            self._fundamentals_df = self._fundamentals_df.copy()
            self._fundamentals_df.columns = [str(col) for col in self._fundamentals_df.columns]

    # ------------------------------------------------------------------
    # Portfolio analysis helpers
    # ------------------------------------------------------------------
    def _weight_series(self) -> pd.Series:
        values = self._portfolio_df.get("valor_actual", pd.Series(dtype=float))
        weights = pd.to_numeric(values, errors="coerce").fillna(0.0)
        total = float(weights.sum())
        if total <= 0:
            if len(weights) == 0:
                return pd.Series(dtype=float)
            uniform = np.repeat(1 / len(weights), len(weights))
            return pd.Series(uniform, index=self._portfolio_df.index)
        return weights / total

    def _distribution(self, column: str) -> pd.Series:
        if column not in self._portfolio_df.columns:
            return pd.Series(dtype=float)
        weights = self._weight_series()
        if weights.empty:
            return pd.Series(dtype=float)
        enriched = self._portfolio_df[column].fillna("Sin dato").astype("string").str.strip()
        grouped = weights.groupby(enriched).sum().sort_values(ascending=False)
        return grouped

    def _merge_fundamentals(self) -> pd.DataFrame:
        if self._fundamentals_df.empty:
            return self._portfolio_df
        fundamentals = self._fundamentals_df.rename(columns={"symbol": "simbolo"})
        merged = self._portfolio_df.merge(fundamentals, on="simbolo", how="left", suffixes=("", "_fund"))
        if "sector_fund" in merged.columns:
            merged["sector"] = merged["sector"].fillna(merged["sector_fund"])
        return merged

    def _compute_beta_distribution(self) -> pd.Series:
        if self._risk_metrics_df.empty or self._portfolio_df.empty:
            return pd.Series(dtype=float)
        df = self._portfolio_df[["simbolo", "valor_actual"]].copy()
        risk = self._risk_metrics_df[["simbolo", "beta"]].copy()
        risk["beta"] = pd.to_numeric(risk["beta"], errors="coerce")
        merged = df.merge(risk, on="simbolo", how="left")
        if merged.empty:
            return pd.Series(dtype=float)

        def _bucket(value: float) -> str:
            if not np.isfinite(value):
                return "Sin dato"
            if value < 0.9:
                return "Beta baja (<0.9)"
            if value > 1.1:
                return "Beta alta (>1.1)"
            return "Beta media (0.9-1.1)"

        merged["beta_bucket"] = merged["beta"].apply(_bucket)
        total = float(merged["valor_actual"].sum())
        if total <= 0:
            return pd.Series(dtype=float)
        distribution = (
            merged.groupby("beta_bucket")["valor_actual"].sum().sort_values(ascending=False) / total
        )
        return distribution

    def analyze_portfolio(self) -> dict[str, object]:
        """Return cached composition analysis for the current portfolio."""

        if self._analysis is not None:
            return self._analysis

        portfolio = self._merge_fundamentals()
        type_dist = self._distribution("tipo")
        sector_dist = self._distribution("sector")
        currency_dist = self._distribution("moneda")
        beta_dist = self._compute_beta_distribution()

        def _bucket_categories(series: pd.Series, *, over: float, under: float) -> tuple[set[str], set[str]]:
            if series.empty:
                return set(), set()
            overexposed = {str(idx) for idx, value in series.items() if value >= over}
            underrepresented = {str(idx) for idx, value in series.items() if value <= under}
            return overexposed, underrepresented

        over_type, under_type = _bucket_categories(type_dist, over=0.45, under=0.15)
        over_sector, under_sector = _bucket_categories(sector_dist, over=0.35, under=0.10)
        over_currency, under_currency = _bucket_categories(currency_dist, over=0.70, under=0.20)
        over_beta, under_beta = _bucket_categories(beta_dist, over=0.55, under=0.20)

        analysis = {
            "total_value": float(self._portfolio_df.get("valor_actual", pd.Series(dtype=float)).sum()),
            "type_distribution": type_dist,
            "sector_distribution": sector_dist,
            "currency_distribution": currency_dist,
            "beta_distribution": beta_dist,
            "overexposed": {
                "tipo": over_type,
                "sector": over_sector,
                "moneda": over_currency,
                "beta": over_beta,
            },
            "underrepresented": {
                "tipo": under_type,
                "sector": under_sector,
                "moneda": under_currency,
                "beta": under_beta,
            },
        }

        self._analysis = analysis
        return analysis

    # ------------------------------------------------------------------
    # Opportunity scoring helpers
    # ------------------------------------------------------------------
    def _ensure_opportunities(self) -> pd.DataFrame:
        df = self._opportunities_df
        if df is None or df.empty:
            return pd.DataFrame()
        if "ticker" in df.columns:
            df = df.rename(columns={"ticker": "symbol"})
        df["symbol"] = df.get("symbol", pd.Series(dtype=str)).astype("string").str.upper()
        df["sector"] = df.get("sector", pd.Series(dtype=str)).fillna("Sin sector").astype("string")
        return df

    @staticmethod
    def _expected_return(row: pd.Series) -> float:
        growth = pd.to_numeric(row.get("cagr"), errors="coerce")
        dividend = pd.to_numeric(row.get("dividend_yield"), errors="coerce")
        growth = 0.0 if not np.isfinite(growth) else float(growth)
        dividend = 0.0 if not np.isfinite(dividend) else float(dividend)
        return max(growth, 0.0) + max(dividend, 0.0)

    @staticmethod
    def _base_score(row: pd.Series) -> float:
        score = pd.to_numeric(row.get("score_compuesto"), errors="coerce")
        if np.isfinite(score):
            return float(score)
        expected = RecommendationService._expected_return(row)
        return 40.0 + expected

    @staticmethod
    def _estimate_beta_from_sector(sector: str) -> float:
        if not sector:
            return 1.0
        sector = str(sector)
        if sector in LOW_RISK_SECTORS:
            return 0.85
        if sector in HIGH_RISK_SECTORS:
            return 1.15
        return 1.0

    def _adjust_score_for_mode(
        self,
        base_score: float,
        row: pd.Series,
        *,
        mode: str,
        expected_return: float,
        estimated_beta: float,
    ) -> float:
        score = base_score
        pe_ratio = pd.to_numeric(row.get("pe_ratio"), errors="coerce")
        pe_ratio = float(pe_ratio) if np.isfinite(pe_ratio) else None
        market_cap = pd.to_numeric(row.get("market_cap"), errors="coerce")
        market_cap = float(market_cap) if np.isfinite(market_cap) else None

        if mode == "max_return":
            score += expected_return * 1.6
            if pe_ratio is not None and pe_ratio > 32:
                score -= min((pe_ratio - 32) * 0.8, 20)
        elif mode == "low_risk":
            if estimated_beta < 0.95:
                score += 15
            elif estimated_beta > 1.05:
                score -= 12
            if market_cap is not None and market_cap > 0:
                score += min(np.log10(market_cap + 1_000) * 3, 18)
            if pe_ratio is not None and pe_ratio > 28:
                score -= min((pe_ratio - 28) * 0.9, 25)
            dividend = pd.to_numeric(row.get("dividend_yield"), errors="coerce")
            if np.isfinite(dividend):
                score += float(dividend) * 0.8
        else:  # diversify
            score += expected_return

        return float(score)

    def _apply_exposure_bias(
        self,
        score: float,
        *,
        sector: str,
        currency: str,
        tipo: str | None,
        mode: str,
        estimated_beta: float,
        analysis: dict[str, object],
    ) -> float:
        updated = score
        over = analysis["overexposed"]
        under = analysis["underrepresented"]

        if sector in over.get("sector", set()):
            updated -= 18
        if sector in under.get("sector", set()):
            updated += 14
        if currency in over.get("moneda", set()):
            updated -= 10
        if currency in under.get("moneda", set()):
            updated += 12

        tipo_value = tipo if tipo else ("CEDEAR" if currency.upper() == "USD" else "ACCION")
        if tipo_value in over.get("tipo", set()):
            updated -= 6
        if tipo_value in under.get("tipo", set()):
            updated += 8

        beta_bucket = None
        if estimated_beta < 0.9:
            beta_bucket = "Beta baja (<0.9)"
        elif estimated_beta > 1.1:
            beta_bucket = "Beta alta (>1.1)"
        else:
            beta_bucket = "Beta media (0.9-1.1)"

        if beta_bucket in over.get("beta", set()):
            updated -= 6
        if beta_bucket in under.get("beta", set()):
            updated += 6

        if mode == "max_return" and sector in HIGH_RISK_SECTORS:
            updated += 4
        if mode == "low_risk" and sector in LOW_RISK_SECTORS:
            updated += 6

        return float(updated)

    def _build_rationale(
        self,
        *,
        symbol: str,
        sector: str,
        currency: str,
        expected_return: float,
        estimated_beta: float,
        mode: str,
        analysis: dict[str, object],
        is_existing: bool = False,
    ) -> str:
        reasons: list[str] = []
        under = analysis["underrepresented"]

        sector_dist = analysis.get("sector_distribution")
        if isinstance(sector_dist, pd.Series) and sector in sector_dist.index:
            current = float(sector_dist.loc[sector]) * 100
            reasons.append(
                f"Sector {sector} hoy representa {current:.1f}% del portafolio"
            )
        elif sector in under.get("sector", set()) or sector:
            reasons.append(f"Agrega exposición al sector {sector}")

        if currency:
            currency_dist = analysis.get("currency_distribution")
            if isinstance(currency_dist, pd.Series) and currency in currency_dist.index:
                current = float(currency_dist.loc[currency]) * 100
                reasons.append(
                    f"{currency} equivale al {current:.1f}% de tu cartera"
                )
            elif currency in under.get("moneda", set()):
                reasons.append(f"Diversifica moneda sumando {currency}")

        beta_note = "beta ≈ {:.2f}".format(estimated_beta)
        if estimated_beta < 0.9:
            beta_note += " (defensivo)"
        elif estimated_beta > 1.1:
            beta_note += " (agresivo)"
        reasons.append(beta_note)

        reasons.append(
            f"Rentabilidad esperada {expected_return:.1f}% combinando crecimiento y dividendos"
        )

        if mode == "low_risk":
            reasons.append("Enfoque defensivo priorizando estabilidad")
        elif mode == "max_return":
            reasons.append("Potencia el retorno esperado del portafolio")
        else:
            reasons.append("Optimiza la diversificación global")

        if is_existing:
            reasons.append("Refuerza una posición ya presente en tu cartera")

        return ", ".join(reasons)

    @staticmethod
    def _infer_tipo(tipo: str | None, currency: str) -> str:
        if tipo:
            return str(tipo).strip().upper()
        return "CEDEAR" if str(currency).upper() == "USD" else "ACCION"

    def _lookup_beta(self, symbol: str, sector: str) -> float:
        if not self._risk_metrics_df.empty and "simbolo" in self._risk_metrics_df.columns:
            matches = self._risk_metrics_df[self._risk_metrics_df["simbolo"] == symbol]
            if not matches.empty:
                beta = pd.to_numeric(matches.iloc[0].get("beta"), errors="coerce")
                if np.isfinite(beta):
                    return float(beta)
        return self._estimate_beta_from_sector(sector)

    @staticmethod
    def _max_allocation_for_existing(
        current_weight: float,
        *,
        additional_amount: float,
        total_portfolio_value: float,
    ) -> float:
        if additional_amount <= 0 or total_portfolio_value <= 0:
            return MAX_ALLOCATION
        allowed = (
            (MAX_ALLOCATION * (total_portfolio_value + additional_amount))
            - (current_weight * total_portfolio_value)
        ) / additional_amount
        if not np.isfinite(allowed):
            return MAX_ALLOCATION
        return float(max(0.0, min(MAX_ALLOCATION, allowed)))

    @staticmethod
    def _apply_weight_constraints(
        weights: pd.Series,
        min_weights: pd.Series,
        max_weights: pd.Series,
    ) -> pd.Series:
        if weights.empty:
            return weights

        weights = weights.clip(lower=0.0)
        total = float(weights.sum())
        if total <= 0:
            weights = pd.Series(
                np.repeat(1.0 / len(weights), len(weights)), index=weights.index
            )
        else:
            weights = weights / total

        min_weights = min_weights.reindex_like(weights).fillna(MIN_ALLOCATION)
        max_weights = max_weights.reindex_like(weights).fillna(MAX_ALLOCATION)

        weights = weights.clip(lower=min_weights, upper=max_weights)

        for _ in range(12):
            total = float(weights.sum())
            if abs(total - 1.0) <= 1e-6:
                break
            if total > 1.0:
                excess = total - 1.0
                adjustable = weights > (min_weights + 1e-6)
                if not adjustable.any():
                    break
                available = float((weights[adjustable] - min_weights[adjustable]).sum())
                if available <= 0:
                    break
                ratio = min(1.0, excess / available)
                adjustment = (weights[adjustable] - min_weights[adjustable]) * ratio
                weights.loc[adjustable] -= adjustment
            else:
                deficit = 1.0 - total
                adjustable = weights < (max_weights - 1e-6)
                if not adjustable.any():
                    break
                capacity = float((max_weights[adjustable] - weights[adjustable]).sum())
                if capacity <= 0:
                    break
                ratio = min(1.0, deficit / capacity)
                adjustment = (max_weights[adjustable] - weights[adjustable]) * ratio
                weights.loc[adjustable] += adjustment

            weights = weights.clip(lower=min_weights, upper=max_weights)

        total = float(weights.sum())
        diff = 1.0 - total
        if abs(diff) > 1e-6:
            if diff > 0:
                adjustable = (max_weights - weights).astype(float)
                adjustable = adjustable[adjustable > 1e-6]
                if not adjustable.empty:
                    idx = adjustable.idxmax()
                    weights.loc[idx] = min(max_weights.loc[idx], weights.loc[idx] + diff)
            else:
                adjustable = (weights - min_weights).astype(float)
                adjustable = adjustable[adjustable > 1e-6]
                if not adjustable.empty:
                    idx = adjustable.idxmax()
                    weights.loc[idx] = max(min_weights.loc[idx], weights.loc[idx] + diff)

        weights = weights.clip(lower=min_weights, upper=max_weights)
        total = float(weights.sum())
        if not np.isclose(total, 1.0, atol=1e-6):
            diff = 1.0 - total
            if diff > 0:
                adjustable = (max_weights - weights).astype(float)
                adjustable = adjustable[adjustable > 1e-9]
                if not adjustable.empty:
                    idx = adjustable.idxmax()
                    weights.loc[idx] = min(
                        max_weights.loc[idx], weights.loc[idx] + diff
                    )
            else:
                adjustable = (weights - min_weights).astype(float)
                adjustable = adjustable[adjustable > 1e-9]
                if not adjustable.empty:
                    idx = adjustable.idxmax()
                    weights.loc[idx] = max(
                        min_weights.loc[idx], weights.loc[idx] + diff
                    )

            weights = weights.clip(lower=min_weights, upper=max_weights)
            total = float(weights.sum())

        if not np.isclose(total, 1.0, atol=1e-6):
            residual = 1.0 - total
            if residual > 0:
                adjustable = (max_weights - weights).astype(float)
                capacity = float(adjustable.sum())
                if capacity > 0:
                    increment = residual * (adjustable / capacity)
                    weights = (weights + increment).clip(lower=min_weights, upper=max_weights)
            else:
                adjustable = (weights - min_weights).astype(float)
                capacity = float(adjustable.sum())
                if capacity > 0:
                    decrement = (-residual) * (adjustable / capacity)
                    weights = (weights - decrement).clip(lower=min_weights, upper=max_weights)

            total = float(weights.sum())

        if not np.isclose(total, 1.0, atol=1e-6):
            weights = weights / total

        return weights

    def _existing_candidates(
        self,
        *,
        amount: float,
        analysis: dict[str, object],
        mode: str,
    ) -> list[Recommendation]:
        if self._portfolio_df.empty:
            return []

        weights = self._weight_series()
        if weights.empty:
            return []

        portfolio = self._merge_fundamentals()
        total_value = float(
            analysis.get(
                "total_value",
                float(self._portfolio_df.get("valor_actual", pd.Series(dtype=float)).sum()),
            )
        )

        recommendations: list[Recommendation] = []
        for idx, row in portfolio.iterrows():
            symbol = str(row.get("simbolo") or "").strip().upper()
            if not symbol:
                continue
            current_weight = float(weights.get(idx, 0.0))
            if current_weight >= MAX_ALLOCATION - 1e-6:
                continue

            sector = str(row.get("sector") or "Sin sector").strip() or "Sin sector"
            currency = str(row.get("moneda") or "ARS").upper()
            tipo = self._infer_tipo(str(row.get("tipo")) if row.get("tipo") is not None else None, currency)

            expected_return = pd.to_numeric(row.get("expected_return"), errors="coerce")
            if not np.isfinite(expected_return):
                expected_return = 5.0 + (1.0 - current_weight) * 5.0

            info_row = pd.Series(
                {
                    "pe_ratio": row.get("pe_ratio"),
                    "market_cap": row.get("market_cap"),
                    "dividend_yield": row.get("dividend_yield"),
                    "cagr": row.get("cagr"),
                }
            )

            base_score = 65.0 + expected_return + (1.0 - current_weight) * 20.0
            estimated_beta = self._lookup_beta(symbol, sector)
            score = self._adjust_score_for_mode(
                base_score,
                info_row,
                mode=mode,
                expected_return=expected_return,
                estimated_beta=estimated_beta,
            )
            score = self._apply_exposure_bias(
                score,
                sector=sector,
                currency=currency,
                tipo=tipo,
                mode=mode,
                estimated_beta=estimated_beta,
                analysis=analysis,
            )

            allocation_cap = self._max_allocation_for_existing(
                current_weight,
                additional_amount=amount,
                total_portfolio_value=total_value,
            )

            if allocation_cap < MIN_ALLOCATION:
                continue

            rationale = self._build_rationale(
                symbol=symbol,
                sector=sector,
                currency=currency,
                expected_return=expected_return,
                estimated_beta=estimated_beta,
                mode=mode,
                analysis=analysis,
                is_existing=True,
            )

            recommendations.append(
                Recommendation(
                    symbol,
                    float(score),
                    float(expected_return),
                    rationale,
                    sector,
                    currency,
                    tipo,
                    True,
                    float(allocation_cap),
                )
            )

        return recommendations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def recommend(
        self,
        amount: float,
        *,
        mode: str = "diversify",
        top_n: int = 5,
        post_filter: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Return the top ``top_n`` recommendations for the provided amount."""

        normalized_amount = _normalise_amount(amount)
        mode = mode if mode in SUPPORTED_MODES else "diversify"
        opportunities = self._ensure_opportunities()
        analysis = self.analyze_portfolio()

        if normalized_amount <= 0:
            return pd.DataFrame(
                columns=["symbol", "allocation_%", "allocation_amount", "rationale"]
            )

        candidates: list[Recommendation] = []
        for _, row in opportunities.iterrows():
            symbol = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
            if not symbol:
                continue
            sector = str(row.get("sector") or "Sin sector").strip() or "Sin sector"
            currency = str(row.get("currency") or "USD").upper()
            tipo = self._infer_tipo(str(row.get("tipo")) if row.get("tipo") is not None else None, currency)
            expected_return = self._expected_return(row)
            base_score = self._base_score(row)
            estimated_beta = self._lookup_beta(symbol, sector)
            score = self._adjust_score_for_mode(
                base_score,
                row,
                mode=mode,
                expected_return=expected_return,
                estimated_beta=estimated_beta,
            )
            score = self._apply_exposure_bias(
                score,
                sector=sector,
                currency=currency,
                tipo=tipo,
                mode=mode,
                estimated_beta=estimated_beta,
                analysis=analysis,
            )

            rationale = self._build_rationale(
                symbol=symbol,
                sector=sector,
                currency=currency,
                expected_return=expected_return,
                estimated_beta=estimated_beta,
                mode=mode,
                analysis=analysis,
            )
            candidates.append(
                Recommendation(
                    symbol,
                    float(score),
                    float(expected_return),
                    rationale,
                    sector,
                    currency,
                    tipo,
                    False,
                    MAX_ALLOCATION,
                )
            )

        candidates.extend(
            self._existing_candidates(amount=normalized_amount, analysis=analysis, mode=mode)
        )

        if not candidates:
            return pd.DataFrame(
                columns=["symbol", "allocation_%", "allocation_amount", "rationale"]
            )

        df = pd.DataFrame([rec.__dict__ for rec in candidates])
        df = df.sort_values("score", ascending=False)
        df = df.drop_duplicates(subset="symbol", keep="first")

        def _apply_balance(frame: pd.DataFrame) -> pd.DataFrame:
            over_sector = analysis.get("overexposed", {}).get("sector", set())
            over_tipo = analysis.get("overexposed", {}).get("tipo", set())
            selected_indices: list[int] = []
            sector_counts: dict[str, int] = {}
            tipo_counts: dict[str, int] = {}

            for idx, row in frame.iterrows():
                if len(selected_indices) >= top_n:
                    break
                sector = row.get("sector", "Sin sector")
                tipo = row.get("tipo", "ACCION")
                max_sector = 1 if sector in over_sector else 2
                max_tipo = 1 if tipo in over_tipo else 3
                if sector_counts.get(sector, 0) >= max_sector:
                    continue
                if tipo_counts.get(tipo, 0) >= max_tipo:
                    continue
                selected_indices.append(idx)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                tipo_counts[tipo] = tipo_counts.get(tipo, 0) + 1

            if len(selected_indices) < top_n:
                for idx in frame.index:
                    if len(selected_indices) >= top_n:
                        break
                    if idx not in selected_indices:
                        selected_indices.append(idx)

            return frame.loc[selected_indices].head(top_n)

        df = _apply_balance(df)

        if not df["is_existing"].any():
            existing_candidates = [rec for rec in candidates if rec.is_existing]
            best_existing = max(existing_candidates, key=lambda rec: rec.score, default=None)
            if best_existing is not None and best_existing.symbol not in set(df["symbol"]):
                df = pd.concat(
                    [
                        df.iloc[:-1],
                        pd.DataFrame([best_existing.__dict__]),
                    ]
                )

        df = df.sort_values("score", ascending=False).head(int(max(top_n, 1)))

        if callable(post_filter):
            try:
                df = post_filter(df)
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("Post filter failed; se preserva la selección original")

        scores = df["score"].clip(lower=0.0)
        if float(scores.sum()) <= 0:
            base_weights = pd.Series(
                np.repeat(1.0 / len(df), len(df)), index=df.index
            )
        else:
            base_weights = scores / float(scores.sum())

        min_constraints = pd.Series(MIN_ALLOCATION, index=df.index)
        max_constraints = pd.Series(
            df["max_allocation"].astype(float).clip(lower=MIN_ALLOCATION, upper=MAX_ALLOCATION),
            index=df.index,
        )
        weights = self._apply_weight_constraints(base_weights, min_constraints, max_constraints)

        df["allocation_%"] = weights * 100
        df["allocation_amount"] = df["allocation_%"] * normalized_amount / 100
        df["symbol"] = df["symbol"].astype(str)

        df = df[["symbol", "allocation_%", "allocation_amount", "rationale"]]

        return df


__all__ = [
    "RecommendationService",
]

