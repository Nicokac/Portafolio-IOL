# Scripts

## `export_analysis.py`

Genera reportes enriquecidos (CSV + Excel) a partir de snapshots persistidos en formato JSON.

### Uso rápido

```bash
python scripts/export_analysis.py \
  --input .cache/portfolio_snapshots \
  --output ./exports/informe_diario \
  --metrics total_value total_pl total_pl_pct positions \
  --charts pl_top composition timeline heatmap \
  --limit 20
```

- `--metrics help` y `--charts help` muestran todas las opciones disponibles.
- `--formats csv|excel|both` permite limitar los formatos generados.
- Cada snapshot produce un subdirectorio con los CSV (`kpis.csv`, `positions.csv`, `history.csv`, etc.) y, si corresponde, el Excel `analysis.xlsx`.
- Se crea además `summary.csv` en la carpeta raíz con los valores crudos de los KPIs seleccionados para todas las corridas.

> Requisitos: tener instalados `kaleido` y `XlsxWriter` (ambos incluidos en `requirements.txt`).
