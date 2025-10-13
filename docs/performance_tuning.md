# Portfolio UI performance tuning

This document summarises the optimisations introduced in `v0.6.6-patch7` for the
portfolio dashboard. The goal of the iteration was to reduce the latency of the
Streamlit front-end while keeping the diagnostic tooling actionable.

## Rendering pipeline overview

`ui.controllers.portfolio_ui.render_portfolio_ui` is the Streamlit entry-point
for the portfolio tab. The controller now orchestrates a set of instrumented
stages inside `controllers.portfolio.portfolio.render_portfolio_section`:

1. **load_data** – calls `load_portfolio_data` to fetch and normalise positions.
2. **apply_filters** – renders the sidebar widgets and returns updated controls.
3. **build_viewmodel** – resolves the snapshot, applies filters and builds the
   `PortfolioViewModel` instance.
4. **notifications** – resolves notification flags to annotate tab titles.
5. **render_tab.<slug>** – renders the active tab (summary, analytics, risk,
   fundamentals or technicals).

Each stage is timed with `services.performance_metrics.measure_execution` via a
small `_record_stage` helper. The resulting durations are exposed through the
telemetry payload and stored as performance metrics.

## Caching strategy

The patch introduces selective Streamlit caching to avoid recomputing expensive
UI artefacts:

- `controllers.portfolio.charts._cached_basic_charts` uses `st.cache_data` to
  memoise the basic chart bundle (P/L, donut, distribution and daily P/L).
- `controllers.portfolio.charts._cached_portfolio_timeline` and
  `_cached_contribution_heatmap` cache the timeline and heatmap figures,
  respectively, which prevents heavy Plotly preparation when datasets are
  unchanged.
- `controllers.portfolio.portfolio._get_cached_favorites` leverages
  `st.cache_resource` to keep the `FavoriteSymbols` helper alive during the
  session, avoiding repeated disk reads.

Cached functions rely on Streamlit's hashing, so they automatically invalidate
when the underlying `DataFrame` inputs change.

## Render diferido por pestaña

La versión `v0.6.6-patch9a1` introduce render diferido por pestaña para el
portafolio. `controllers.portfolio.portfolio.render_portfolio_section` ahora
almacena en `st.session_state["render_cache"]` los placeholders y la firma de
datos asociada a cada pestaña. Solo la pestaña activa se renderiza; el resto
queda vacía hasta que el usuario la selecciona.

Cada pestaña reutiliza su placeholder cuando el usuario vuelve a visitarla, por
lo que la vista se muestra desde caché sin volver a ejecutar el código pesado.
Cuando cambian los datos (por ejemplo, un nuevo snapshot o valores distintos en
el `DataFrame`), la firma se invalida y el contenido se recalcula marcando la
fuente como `hot`.

Además, se muestra un mensaje contextual con el tiempo de carga y la fuente
(`cálculo inicial`, `recalculado` o `caché en memoria`). Las métricas de
latencia se registran mediante `services.health.record_tab_latency`, lo que
permite auditar la mejora desde el panel de diagnósticos y el sistema de
telemetría.

## Caché incremental por subcomponente (v0.6.6-patch9a2)

La iteración `patch9a2` suma una capa de cacheo más granular: resumen, tabla y
gráficos almacenan su propio placeholder y firma de datos mediante
`portfolio_id|filters_hash|tab`. Cada componente utiliza `CacheService` con un
TTL de cinco minutos para persistir el timestamp de cálculo y mostrarlo en la
UI.

- `_portfolio_dataset_key` genera la huella a partir de `snapshot_id` (si
  existe) o de hashes de `positions`, `historical_total` y `contribution_metrics`.
- Los filtros de tabla (orden, dirección y USD) invalidan únicamente la tabla;
  cambios en `top_n`, dataset o favoritos fuerzan la recomputación del resto.
- `_record_stage("render_summary")`, `render_charts` y `render_table`
  registran tiempos parciales que ahora se exponen en el panel de diagnósticos
  sin depender del backend histórico.

## Actualización de cotizaciones con Stale-While-Revalidate (v0.6.6-patch9b2)

La canalización `quotes_refresh` ahora divide los símbolos del portafolio en
lotes homogéneos por tipo de activo y los resuelve en paralelo respetando
`MAX_QUOTE_WORKERS`. Cada lote se almacena en un caché persistente con la
estrategia *Stale-While-Revalidate*: si los datos están dentro del TTL efectivo
(`QUOTES_SWR_TTL_SECONDS`) se sirven al instante; cuando caducan pero continúan
dentro de la ventana de gracia (`QUOTES_SWR_GRACE_SECONDS`) se devuelven desde
caché mientras un *refresh* asincrónico actualiza el lote en segundo plano.

Los resultados se instrumentan con métricas Prometheus:

- `quotes_swr_served_total` etiqueta cuántos lotes se sirven en modo `fresh`,
  `refresh` o `stale`.
- `quotes_batch_latency_seconds` resume la latencia de cada lote indicando si se
  ejecutó en segundo plano.

Los logs estructurados incluyen `group`, `symbols`, `size` y el modo de servicio
de cada lote, lo que facilita auditar cuellos de botella o lotes con alta tasa
de staleness. La mejora reduce la latencia observada en el front-end de ~5.5 s a
aprox. 2–3 s bajo carga moderada, al eliminar esperas innecesarias en la UI.

## Diagnostics panel

`ui.panels.diagnostics.render_diagnostics_panel` now renders a dedicated table
labelled *Portfolio UI (subcomponentes)*. It surfaces the latest timings for the
stages listed above, enabling quick identification of slow render branches.
Additional instrumented metrics continue to appear in a separate
*Métricas instrumentadas* table. Both tables are powered by
`services.performance_metrics.get_recent_metrics` and can be exported as CSV for
offline analysis.

## Operational notes

- When adding new render steps to the portfolio UI, wrap them with
  `_record_stage("your_stage", timings)` to inherit both telemetry tracking and
  diagnostics reporting.
- Prefer `st.cache_data` for pure functions that return `DataFrame` or Plotly
  objects, and `st.cache_resource` for stateful helpers.
- The diagnostics tables only appear after the respective stages have executed
  at least once; trigger the relevant tabs locally before taking measurements.
