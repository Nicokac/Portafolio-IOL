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
    beta: float
    sector: str
    tipo: str
    currency: str
    is_existing: bool = False
    rationale_extended: str = ""


RECOMMENDATION_OUTPUT_COLUMNS: tuple[str, ...] = (
    "symbol",
    "allocation_%",
    "allocation_amount",
    "expected_return",
    "beta",
    "rationale",
    "rationale_extended",
    "sector",
    "tipo",
    "currency",
)


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
        self._risk_metrics_df = risk_metrics_df.copy() if isinstance(risk_metrics_df, pd.DataFrame) else pd.DataFrame()
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

    def _compute_portfolio_beta_average(self) -> float:
        if self._portfolio_df.empty or self._risk_metrics_df.empty:
            return float("nan")

        portfolio = self._portfolio_df[["simbolo"]].copy()
        portfolio["simbolo"] = portfolio["simbolo"].astype("string").str.upper()
        weights = self._weight_series()
        if weights.empty:
            return float("nan")

        portfolio = portfolio.assign(weight=weights)
        risk = self._risk_metrics_df.copy()
        symbol_col = "simbolo" if "simbolo" in risk.columns else "symbol"
        risk = risk[[symbol_col, "beta"]].copy()
        risk[symbol_col] = risk[symbol_col].astype("string").str.upper()
        risk["beta"] = pd.to_numeric(risk["beta"], errors="coerce")

        merged = portfolio.merge(risk, left_on="simbolo", right_on=symbol_col, how="left")
        merged["beta"] = pd.to_numeric(merged["beta"], errors="coerce")
        merged = merged[np.isfinite(merged["beta"]) & np.isfinite(merged["weight"])]
        if merged.empty:
            return float("nan")

        beta_avg = float((merged["weight"] * merged["beta"]).sum())
        return beta_avg

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

        beta_average = self._compute_portfolio_beta_average()

        analysis = {
            "total_value": float(self._portfolio_df.get("valor_actual", pd.Series(dtype=float)).sum()),
            "type_distribution": type_dist,
            "sector_distribution": sector_dist,
            "currency_distribution": currency_dist,
            "beta_distribution": beta_dist,
            "beta_average": beta_average,
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

        tipo = "CEDEAR" if currency.upper() == "USD" else "ACCION"
        if tipo in over.get("tipo", set()):
            updated -= 6
        if tipo in under.get("tipo", set()):
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

        return ", ".join(reasons)

    def _build_extended_rationale(
        self,
        *,
        allocation_pct: float,
        expected_return: float,
        beta: float,
        sector: str,
        analysis: dict[str, object],
        portfolio_beta: float,
    ) -> str:
        parts: list[str] = []

        contribution = float(allocation_pct) * float(expected_return) / 100.0
        if not np.isfinite(contribution):
            contribution = 0.0
        parts.append(
            f"Aporta {contribution:+.2f} % al retorno esperado del portafolio."
        )

        beta_value = float(beta)
        base_beta = float(portfolio_beta)
        if not np.isfinite(base_beta):
            base_beta = 1.0
        beta_delta = (base_beta - beta_value) * float(allocation_pct) / 100.0
        if np.isfinite(beta_delta) and abs(beta_delta) > 1e-6:
            if beta_delta > 0:
                parts.append(f"Reduce el beta total en {abs(beta_delta):.2f} %.")
            else:
                parts.append(f"Aumenta el beta total en {abs(beta_delta):.2f} %.")
        else:
            parts.append("Mantiene el beta total estable.")

        sector_label = str(sector or "Sin sector").strip() or "Sin sector"
        sector_dist = analysis.get("sector_distribution")
        if isinstance(sector_dist, pd.Series) and sector_label in sector_dist.index:
            share = float(sector_dist.loc[sector_label]) * 100
            parts.append(
                f"Contribuye a diversificación sectorial ({sector_label} hoy {share:.1f}%)."
            )
        else:
            parts.append(
                f"Contribuye a diversificación sectorial incorporando {sector_label}."
            )

        return " ".join(parts)

    def _existing_asset_candidates(
        self,
        *,
        analysis: dict[str, object],
        mode: str,
    ) -> list[Recommendation]:
        if self._portfolio_df.empty:
            return []

        under = analysis.get("underrepresented", {})
        over = analysis.get("overexposed", {})
        weights = self._weight_series()
        candidates: list[Recommendation] = []

        for idx, row in self._portfolio_df.iterrows():
            symbol = str(row.get("simbolo") or "").strip().upper()
            if not symbol:
                continue

            sector = str(row.get("sector") or "Sin sector").strip() or "Sin sector"
            currency = str(row.get("moneda") or "ARS").upper()
            tipo = str(row.get("tipo") or ("CEDEAR" if currency == "USD" else "ACCION")).strip() or "ACCION"

            if sector in over.get("sector", set()) and sector not in under.get("sector", set()):
                continue
            if tipo in over.get("tipo", set()) and tipo not in under.get("tipo", set()):
                continue

            qualifies = (
                sector in under.get("sector", set())
                or tipo in under.get("tipo", set())
                or currency in under.get("moneda", set())
            )
            if not qualifies:
                continue

            weight = float(weights.get(idx, 0.0))
            base_score = 52.0 + weight * 80.0
            estimated_beta = self._estimate_beta_from_sector(sector)

            score = self._adjust_score_for_mode(
                base_score,
                pd.Series(row),
                mode=mode,
                expected_return=0.0,
                estimated_beta=estimated_beta,
            )
            score = self._apply_exposure_bias(
                score,
                sector=sector,
                currency=currency,
                mode=mode,
                estimated_beta=estimated_beta,
                analysis=analysis,
            )

            rationale = self._build_rationale(
                symbol=symbol,
                sector=sector,
                currency=currency,
                expected_return=0.0,
                estimated_beta=estimated_beta,
                mode=mode,
                analysis=analysis,
            )

            candidates.append(
                Recommendation(
                    symbol=symbol,
                    score=score,
                    expected_return=0.0,
                    rationale=rationale,
                    beta=estimated_beta,
                    sector=sector,
                    tipo=tipo,
                    currency=currency,
                    is_existing=True,
                )
            )

        return candidates

    @staticmethod
    def _allocate_with_bounds(
        scores: pd.Series | np.ndarray,
        *,
        min_weight: float = 0.10,
        max_weight: float = 0.40,
    ) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
        n_assets = len(values)
        if n_assets == 0:
            return np.array([], dtype=float)

        if np.allclose(values.sum(), 0.0):
            values = np.ones(n_assets, dtype=float)

        values = values / float(values.sum())

        min_sum = min_weight * n_assets
        if min_sum >= 1.0:
            # Fallback to an even split when lower bound is infeasible.
            return np.repeat(1.0 / n_assets, n_assets)

        allocation = np.full(n_assets, min_weight, dtype=float)
        remaining = 1.0 - allocation.sum()
        if remaining <= 0:
            return allocation / float(allocation.sum())

        capacities = np.full(n_assets, max_weight - min_weight, dtype=float)
        weights = values.copy()

        active = allocation < max_weight - 1e-12
        while remaining > 1e-9 and np.any(active):
            shares = weights[active]
            total_shares = float(shares.sum())
            if total_shares <= 0:
                shares = np.ones_like(shares)
                total_shares = float(shares.sum())

            to_assign = remaining * (shares / total_shares)
            active_indices = np.flatnonzero(active)
            increments = np.minimum(capacities[active_indices], to_assign)
            allocation[active_indices] += increments
            remaining = 1.0 - float(allocation.sum())

            reached_cap = allocation >= max_weight - 1e-12
            active = ~reached_cap

            if remaining > 1e-9 and not np.any(active):
                # Distribute any small leftover evenly among those with headroom.
                headroom = max_weight - allocation
                mask = headroom > 1e-12
                if np.any(mask):
                    extra = remaining * headroom[mask] / float(headroom[mask].sum())
                    allocation[np.flatnonzero(mask)] += extra
                    remaining = 1.0 - float(allocation.sum())

        if not np.isclose(allocation.sum(), 1.0):
            if allocation.sum() > 1.0:
                reducible = allocation - min_weight
                mask = reducible > 1e-12
                if np.any(mask):
                    reduction = (allocation.sum() - 1.0) * reducible[mask] / float(
                        reducible[mask].sum()
                    )
                    allocation[np.flatnonzero(mask)] -= reduction
            else:
                headroom = max_weight - allocation
                mask = headroom > 1e-12
                if np.any(mask):
                    addition = (1.0 - allocation.sum()) * headroom[mask] / float(
                        headroom[mask].sum()
                    )
                    allocation[np.flatnonzero(mask)] += addition

        allocation = np.clip(allocation, min_weight, max_weight)
        allocation = allocation / float(allocation.sum())
        return allocation

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

        if normalized_amount <= 0 or opportunities.empty:
            return pd.DataFrame(columns=RECOMMENDATION_OUTPUT_COLUMNS)

        candidates: list[Recommendation] = []
        for _, row in opportunities.iterrows():
            symbol = str(row.get("symbol") or row.get("ticker") or "").strip().upper()
            if not symbol:
                continue
            sector = str(row.get("sector") or "Sin sector").strip() or "Sin sector"
            currency = str(row.get("currency") or "USD").upper()
            tipo = str(row.get("tipo") or ("CEDEAR" if currency == "USD" else "ACCION")).strip() or "ACCION"
            expected_return = self._expected_return(row)
            base_score = self._base_score(row)
            estimated_beta = self._estimate_beta_from_sector(sector)
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
                    symbol=symbol,
                    score=score,
                    expected_return=expected_return,
                    rationale=rationale,
                    beta=estimated_beta,
                    sector=sector,
                    tipo=tipo,
                    currency=currency,
                )
            )

        candidates.extend(
            self._existing_asset_candidates(analysis=analysis, mode=mode)
        )

        if not candidates:
            return pd.DataFrame(columns=RECOMMENDATION_OUTPUT_COLUMNS)

        base_df = pd.DataFrame([rec.__dict__ for rec in candidates])
        df = base_df.sort_values("score", ascending=False)
        df = df.drop_duplicates(subset="symbol", keep="first")
        df = df.head(int(max(top_n, 1)))

        if callable(post_filter):
            try:
                filtered = post_filter(df.copy())
                if isinstance(filtered, pd.DataFrame):
                    df = filtered
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("Post filter failed; se preserva la selección original")

        if df.empty:
            return pd.DataFrame(columns=RECOMMENDATION_OUTPUT_COLUMNS)

        if "symbol" not in df.columns:
            raise ValueError("El filtro debe preservar la columna 'symbol'")

        df["symbol"] = df["symbol"].astype(str)
        base_lookup = base_df.drop_duplicates(subset="symbol").set_index("symbol")
        required_cols = [
            "score",
            "expected_return",
            "rationale",
            "beta",
            "sector",
            "tipo",
            "currency",
            "is_existing",
            "rationale_extended",
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = base_lookup.reindex(df["symbol"])[col].values
            else:
                series = df[col]
                if series.dtype.kind in "fc":
                    mask = series.isna()
                else:
                    mask = series.isna()
                    mask |= series.astype("string").eq("")
                if mask.any():
                    df.loc[mask, col] = base_lookup.reindex(df.loc[mask, "symbol"])[col].values

        df = df.drop_duplicates(subset="symbol", keep="first")
        df = df.head(int(max(top_n, 1)))

        if not df.empty and not df["is_existing"].any():
            existing_pool = base_df.query("is_existing").sort_values(
                "score", ascending=False
            )
            if not existing_pool.empty:
                replacement = existing_pool.iloc[0]
                if replacement["symbol"] not in set(df["symbol"]):
                    df = pd.concat([df.iloc[:-1], replacement.to_frame().T], ignore_index=True)
                    df = df.sort_values("score", ascending=False)
                    df = df.head(int(max(top_n, 1)))

        type_over = analysis.get("overexposed", {}).get("tipo", set())
        type_under = analysis.get("underrepresented", {}).get("tipo", set())
        sector_over = analysis.get("overexposed", {}).get("sector", set())
        sector_under = analysis.get("underrepresented", {}).get("sector", set())

        balance_factors: list[float] = []
        for row in df.itertuples():
            factor = 1.0
            tipo = getattr(row, "tipo", "")
            sector = getattr(row, "sector", "")
            if tipo in type_under:
                factor += 0.2
            elif tipo in type_over:
                factor -= 0.15
            if sector in sector_under:
                factor += 0.2
            elif sector in sector_over:
                factor -= 0.15
            balance_factors.append(max(factor, 0.05))

        scores = df["score"].clip(lower=0.0) * np.array(balance_factors)
        weights = self._allocate_with_bounds(scores.to_numpy())

        df["allocation_%"] = weights * 100
        df["allocation_amount"] = df["allocation_%"] * normalized_amount / 100
        df["symbol"] = df["symbol"].astype(str)

        total_pct = float(df["allocation_%"].sum())
        if not np.isclose(total_pct, 100.0):
            adjustment = 100.0 - total_pct
            last_idx = df.index[-1]
            df.loc[last_idx, "allocation_%"] += adjustment
            df.loc[last_idx, "allocation_amount"] = (
                df.loc[last_idx, "allocation_%"] * normalized_amount / 100
            )

        df["expected_return"] = pd.to_numeric(
            df.get("expected_return"), errors="coerce"
        )
        df["beta"] = pd.to_numeric(df.get("beta"), errors="coerce")
        df["sector"] = (
            df.get("sector", pd.Series(dtype=str)).astype("string").fillna("Sin sector")
        )
        df["tipo"] = df.get("tipo", pd.Series(dtype=str)).astype("string")
        df["currency"] = df.get("currency", pd.Series(dtype=str)).astype("string")

        portfolio_beta = float(analysis.get("beta_average", float("nan")))
        extended: list[str] = []
        for _, row in df.iterrows():
            allocation_pct = float(row.get("allocation_%", 0.0))
            expected_val = pd.to_numeric(row.get("expected_return"), errors="coerce")
            expected_val = float(expected_val) if np.isfinite(expected_val) else 0.0
            beta_val = pd.to_numeric(row.get("beta"), errors="coerce")
            beta_val = float(beta_val) if np.isfinite(beta_val) else float("nan")
            sector_label = str(row.get("sector", ""))
            extended.append(
                self._build_extended_rationale(
                    allocation_pct=allocation_pct,
                    expected_return=expected_val,
                    beta=beta_val,
                    sector=sector_label,
                    analysis=analysis,
                    portfolio_beta=portfolio_beta,
                )
            )
        df["rationale_extended"] = extended

        return df[list(RECOMMENDATION_OUTPUT_COLUMNS)]


__all__ = [
    "RecommendationService",
]

