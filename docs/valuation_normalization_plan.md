# Plan de validación de normalización de valuaciones

La versión **v0.8.4** introduce la corrección definitiva de escalas para bonos y
letras BOPREAL (series BPOA7–BPOC7). Este documento resume el flujo de
verificación recomendado para confirmar que los snapshots activos reflejan los
valores ajustados y que los totales del dashboard se sincronizan con los montos
oficiales de InvertirOnline.

## Objetivos

1. Verificar que `detect_bond_scale_anomalies()` no reporte escalas anómalas en
   los activos BOPREAL tras el recalculado de `calc_rows`.
2. Comparar la vista del portafolio contra la baseline v0.8.3 y auditar los
   deltas de `valor_actual`, `costo` y `pl` por símbolo.
3. Asegurar la sincronización de `PortfolioViewModelService` invalidando el
   cache incremental cuando `PORTFOLIO_TOTALS_VERSION` cambia.
4. Confirmar que los totales consolidados coincidan con los informados por la
   API oficial de IOL (margen ± 1 ARS para caja).

## Requisitos previos

- Variables de entorno `IOL_USERNAME` y `IOL_PASSWORD` configuradas o un dataset
  cacheado reciente.
- Instalar dependencias del proyecto (`pip install -r requirements.txt` y, si
  aplica, `-r requirements-dev.txt`).
- Dataset baseline (v0.8.3) exportado en formato CSV/Parquet/JSON para ejecutar
  la comparativa.

## Procedimiento de validación

1. **Ejecutar detección de escalas online (opcional, refresca dataset):**

   ```bash
   python -m scripts.check_bond_scale
   ```

   El script descargará el dataset actual, regenerará la vista del portafolio y
   mostrará el reporte de anomalías. Debería indicar *"Sin anomalías
   detectadas"* para las series BOPREAL.

2. **Validación offline con dataset cacheado:**

   ```bash
   python -m scripts.check_bond_scale --offline \
       --baseline-view archive/portfolio_v0.8.3.csv \
       --delta-output logs/bond_scale_delta_v0.8.4.csv
   ```

   - `--offline` reutiliza el dataset persistido por el `PortfolioDataFetchService`.
   - `--baseline-view` lee la vista agregada de la versión previa para generar el
     delta por símbolo.
   - `--delta-output` opcionalmente persiste el resultado para auditorías.

   El reporte mostrará las columnas `_prev`, `_curr` y `delta_` para
   `valor_actual`, `costo` y `pl`. Los valores BOPREAL deben reflejar la escala
   corregida (≈ 19.9 M ARS) y un P/L consistente con IOL (± 5 %).

3. **Sincronización de totales:**

   - Confirmar que `PORTFOLIO_TOTALS_VERSION=5.6` esté exportada en el entorno.
   - Iniciar la UI o ejecutar el flujo `get_portfolio_view()` para generar un
     snapshot nuevo.
   - Verificar que los metadatos del snapshot indiquen `totals_version = v5.6`
     y que los totales recalculados coincidan con los datos de IOL.

4. **Checklist final:**

   - [ ] `pytest -q tests/test_scale_bopreal.py`
   - [ ] `pytest -q application/test/test_calc_rows_performance.py`
   - [ ] `python -m scripts.check_bond_scale --offline` (sin anomalías)
   - [ ] Comparativa de totales entre dashboard e IOL (desvío ≤ 1 ARS)
   - [ ] Registrar resultados en `docs/qa/` si corresponde.

## Notas adicionales

- El modo offline requiere un dataset previamente cacheado. Si no hay datos
  disponibles, ejecutar primero el script sin `--offline` o refrescar la
  caché desde la UI.
- Los deltas exportados pueden adjuntarse a reportes de QA para documentar el
  impacto del recalculado v0.8.4 frente a v0.8.3.
