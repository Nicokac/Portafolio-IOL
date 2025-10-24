import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from application.portfolio_service import calc_rows, clean_symbol, scale_for
from shared.utils import _to_float

# Copia de la implementación original basada en bucles para comparar rendimientos


def calc_rows_loop(get_quote_fn, df_pos: pd.DataFrame, exclude_syms):
    cols = [
        "simbolo",
        "mercado",
        "tipo",
        "cantidad",
        "ppc",
        "ultimo",
        "valor_actual",
        "costo",
        "pl",
        "pl_%",
        "pl_d",
        "pld_%",
    ]

    if df_pos is None or df_pos.empty:
        return pd.DataFrame(columns=cols)

    ex = {clean_symbol(s) for s in (exclude_syms or [])}
    rows = []
    for _, p in df_pos.iterrows():
        simbolo = clean_symbol(p["simbolo"])
        if simbolo in ex:
            continue

        mercado = str(p["mercado"]).lower()
        cantidad = _to_float(p["cantidad"]) or 0.0
        ppc = _to_float(p.get("costo_unitario")) or 0.0

        last, chg_pct = None, None
        q = None
        try:
            q = get_quote_fn(mercado, simbolo)
        except Exception:
            pass

        if isinstance(q, dict):
            for k in ("last", "ultimo", "ultimoPrecio", "precio", "close", "cierre"):
                if k in q:
                    last = _to_float(q.get(k))
                    if last is not None:
                        break
            chg_pct = _to_float(q.get("chg_pct"))
        else:
            last = _to_float(q)

        tipo = str(p.get("tipo", "N/D"))
        scale_row = pd.Series(
            {
                "simbolo": simbolo,
                "tipo_estandar": tipo,
                "tipo": tipo,
                "moneda": p.get("moneda"),
                "moneda_origen": p.get("moneda_origen"),
                "proveedor_original": p.get("proveedor_original"),
                "pricing_source": p.get("pricing_source"),
                "provider": p.get("provider"),
            }
        )
        scale = scale_for(scale_row)

        costo = cantidad * ppc * scale
        valor = (cantidad * last * scale) if last is not None else np.nan
        pl = (valor - costo) if not np.isnan(valor) else np.nan
        pl_pct = (pl / costo * 100.0) if costo else np.nan

        if (chg_pct is not None) and (valor == valor):
            denom = 1.0 + (float(chg_pct) / 100.0)
            if denom != 0:
                pl_d = float(valor) * (float(chg_pct) / 100.0) / denom
                pld_pct = float(chg_pct)
            else:
                pl_d = np.nan
                pld_pct = np.nan
        else:
            pl_d = np.nan
            pld_pct = np.nan

        rows.append(
            {
                "simbolo": simbolo,
                "mercado": mercado.upper(),
                "tipo": tipo,
                "cantidad": cantidad,
                "ppc": ppc,
                "ultimo": last,
                "valor_actual": valor,
                "costo": costo,
                "pl": pl,
                "pl_%": pl_pct,
                "pl_d": pl_d,
                "pld_%": pld_pct,
            }
        )

    return pd.DataFrame(rows, columns=cols)


def dummy_quote(mkt, sym):
    return {"last": 100.0, "chg_pct": 1.0}


def make_df(n=1000):
    return pd.DataFrame(
        {
            "simbolo": [f"AAA{i}" for i in range(n)],
            "mercado": ["BCBA"] * n,
            "cantidad": np.arange(1, n + 1, dtype=float),
            "costo_unitario": np.linspace(10, 20, n),
            "tipo": ["N/D"] * n,
        }
    )


def test_calc_rows_perf():
    df = make_df(1000)
    t0 = time.perf_counter()
    calc_rows_loop(dummy_quote, df, [])
    t_loop = time.perf_counter() - t0

    t1 = time.perf_counter()
    calc_rows(dummy_quote, df, [])
    t_vec = time.perf_counter() - t1

    print("loop:", t_loop, "vectorizado:", t_vec)
    # Simple check: ambas ejecuciones deben producir tiempos positivos
    # (el benchmark queda registrado en la salida estándar).
    assert t_loop > 0 and t_vec > 0
