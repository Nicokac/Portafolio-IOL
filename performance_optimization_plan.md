# Plan de optimización de performance – Portafolio IOL v0.6.13

## Diagnóstico general
- Las mediciones previas al lazy load de Kaleido capturadas en `performance_metrics_5.csv` mantienen a `quotes_refresh` y `portfolio_view.render` como los dos cuellos de botella principales (7.1 s y 7.5 s respectivamente) dentro de un tiempo total de carga de 83 s.【F:performance_metrics_5.csv†L1-L5】
- El muestreo posterior (`performance_metrics_6.csv`) confirma que, aun con sublotes concurrentes, `quotes_refresh` consume 6.68 s y el render del viewmodel 7.21 s, mientras que la carga UI total asciende a 127 s cuando se ejecutan procesos de precarga adicionales.【F:performance_metrics_6.csv†L1-L5】
- La telemetría más reciente (`performance_metrics_9.csv`) muestra 6.94 s para `quotes_refresh`, sublotes de hasta 1.32 s y un hit ratio del memoizador del portafolio de 55.56 % (21.38 s acumulados en `portfolio_view.render`).【F:performance_metrics_9.csv†L1-L11】【F:docs/fixtures/telemetry/quotes_refresh_logs.jsonl†L1-L20】
- Las proyecciones de la iteración adaptativa (`performance_metrics_10.csv`) indican que es posible bajar `quotes_refresh_total_s` a 5.78 s si se sostiene un tamaño de sublote promedio 5 y 596 ms por lote.【F:performance_metrics_10.csv†L1-L4】
- Los snapshots de cache del viewmodel confirman misiones de 3.1 s a 3.8 s por `apply` y 0.9 s–1.1 s en `totals`, lo que deja margen para reducir cálculos redundantes.【F:docs/fixtures/telemetry/portfolio_view_cache.json†L2-L55】
- `performance_metrics_14.csv` y `performance_metrics_15.csv` registran la corrida del 18/10 (refresh de 8.43 s, render de pestaña hasta 16.70 s y snapshots en ~5 s), habilitando comparativas pre/post Kaleido con datos reales.【F:performance_metrics_14.csv†L2-L27】【F:performance_metrics_15.csv†L2-L27】

## Tiempos actuales vs estimaciones propuestas
| Etapa | Tiempo actual | Fuente | Mejora estimada tras ajustes | Impacto aproximado |
| --- | --- | --- | --- | --- |
| `quotes_refresh` | 6.94 s | Telemetría v0.6.13【F:performance_metrics_9.csv†L2-L4】 | 3.5 s (−49 %) aplicando warm-start desde caché persistente + sublotes 4–5 estables | 3.4 s menos |
| `portfolio_view.apply` | 3.1 s promedio (hasta 3.85 s) | Snapshot de memoizador【F:docs/fixtures/telemetry/portfolio_view_cache.json†L19-L54】 | 1.5 s (−52 %) omitiendo cálculo de contribuciones/historial cuando la pestaña no está activa y memoizando retornos derivados | 1.6 s menos |
| `startup.render_portfolio_complete` (proxy `ui_total_load`) | 83 s | Métrica baseline UI【F:performance_metrics_5.csv†L1-L3】 | 9 s (<10 s objetivo) tras adelantar hidratos de sesión, diferir gráficos pesados y reutilizar controles | 74 s menos (el valor incluye tiempos de espera actuales)

## Cuellos de botella y recomendaciones
### 1. `quotes_refresh`
- La canalización agrupa símbolos por tipo y normaliza claves en cada solicitud, incluso cuando el dataset no cambia, provocando sublotes de hasta 1.32 s.【F:controllers/portfolio/load_data.py†L119-L142】【F:docs/fixtures/telemetry/quotes_refresh_logs.jsonl†L1-L20】
- `_get_quote_cached` bloquea hasta agotar TTL antes de caer al JSON persistido; sólo usa los datos guardados si la API devolvió `last=None`, desaprovechando el warm-start local.【F:services/cache/quotes.py†L546-L737】
- `fetch_quotes_bulk` normaliza cada entrada y vuelve a registrar métricas aunque la respuesta venga íntegramente de caché SWR, lo que suma ~180 ms de overhead por lote.【F:services/cache/quotes.py†L740-L820】

**Acciones sugeridas**
1. **Prehidratar la caché en memoria desde el archivo persistido antes del primer refresh** para cada símbolo, devolviendo datos inmediatos mientras se lanza el refresh en background (`StaleWhileRevalidateCache`). Aprovecha `_recover_persisted_quote` antes de llamar a la API para los lotes iniciales, reduciendo el primer paint ~1.5 s.【F:services/cache/quotes.py†L588-L737】
2. **Memoizar `build_quote_batches` por hash del dataset** (`PortfolioViewModelService._hash_dataset`) y reusar la última asignación de grupos cuando `df_pos` no cambia, evitando recalcular la agrupación por tipo en cada tick (~300 ms).【F:controllers/portfolio/load_data.py†L168-L209】【F:services/portfolio_view.py†L739-L776】
3. **Limitar temporalmente el tamaño de sublote a 4 símbolos cuando la media supere 650 ms** utilizando `AdaptiveBatchController.observe`, estabilizando los picos de 1.3 s y acercando el total a la proyección de 5.78 s.【F:services/cache/quotes.py†L741-L820】【F:performance_metrics_10.csv†L2-L4】
4. **Registrar `quotes_refresh_total_s` en `performance_metrics_15.csv`** cada vez que se ejecute la canalización para visibilidad pre/post Kaleido.

### 2. `portfolio_view.apply`
- El memoizador siempre recalcula contribuciones e historial aunque el usuario permanezca en pestañas que no los consumen.【F:services/portfolio_view.py†L329-L420】【F:controllers/portfolio/portfolio.py†L640-L776】
- `render_portfolio_section` construye el `viewmodel` completo (incluyendo totales, historial y conversiones FX) antes de conocer la pestaña seleccionada, incluso cuando la sesión sólo quiere ver las notificaciones.【F:controllers/portfolio/portfolio.py†L1061-L1199】
- El snapshot se persiste de forma síncrona en `_persist_snapshot`, bloqueando el hilo UI aún cuando no se consultará historial inmediatamente.【F:services/portfolio_view.py†L969-L1020】

**Acciones sugeridas**
1. **Calcular contribuciones/historial bajo demanda**: condicionar `compute_contributions_fn` y `_update_history` a que la pestaña básica esté activa o que `render_charts_section` los solicite, guardando resultados en `self._incremental_cache` para reutilizarlos cuando sea necesario (ahorro estimado 1.0 s).【F:services/portfolio_view.py†L329-L420】【F:controllers/portfolio/portfolio.py†L640-L776】
2. **Separar el armado del viewmodel en dos fases** (datos mínimos vs. agregados) para poder responder al UI con posiciones/totales básicos y postergar cálculos costosos hasta después del render inicial (≈0.6 s menos en `apply_elapsed`).【F:controllers/portfolio/portfolio.py†L1061-L1184】
3. **Persistir snapshots en background** usando `threading.Thread` o `asyncio.create_task`, marcando el snapshot como “pending persist” para no bloquear la respuesta (≈0.3 s).【F:services/portfolio_view.py†L988-L1019】
4. **Extender la memoización de retornos** guardando `returns_df` en `_incremental_cache` y validando su fingerprint para evitar recomputar cuando sólo cambian filtros no temporales (≈0.2 s).【F:services/portfolio_view.py†L329-L420】

### 3. `startup.render_portfolio_complete`
- La métrica de 83 s incluye espera de autenticación más cálculos previos; `streamlit_overhead_ms` (~1.8 s) sugiere que la mayor parte del tiempo se concentra en la fase lógica previa al primer render.【F:performance_metrics_5.csv†L1-L3】【F:performance_metrics_9.csv†L6-L9】
- `render_portfolio_section` dispara carga de datos, render del sidebar y construcción completa del viewmodel antes de mostrar cualquier feedback al usuario.【F:controllers/portfolio/portfolio.py†L1061-L1184】

**Acciones sugeridas**
1. **Precalcular favoritos, tasas FX y controles en `session_state` durante el login** para evitar recomputes dentro de la primera ejecución de `render_portfolio_section` (≈2 s menos percibidos).【F:controllers/portfolio/portfolio.py†L1091-L1113】
2. **Introducir placeholders ligeros** para tabla y gráficos (por ejemplo, skeletons) y lanzar el render completo mediante `st.experimental_rerun` cuando la data esté lista, aprovechando el memoizador para rellenar sin bloquear (≈3 s menos de espera visual).【F:controllers/portfolio/portfolio.py†L1150-L1189】
3. **Restaurar la escritura de `startup.render_portfolio_complete` en `performance_metrics_15.csv`** desde `app.py` para medir el progreso real del objetivo sub-10 s.

## Revisión técnica de módulos clave
### controllers/portfolio/portfolio.py
- Reutilizar el `viewmodel` almacenado en `st.session_state` para saltarse `build_portfolio_viewmodel` cuando `controls` no cambiaron, devolviendo la última respuesta instantáneamente.【F:controllers/portfolio/portfolio.py†L1115-L1119】
- Permitir que `render_basic_tab` reciba un flag `lazy_metrics` para no pasar `historical_total` y `contribution_metrics` hasta que la carta de composición se solicite.【F:controllers/portfolio/portfolio.py†L640-L776】

### services/cache/quotes.py
- Mover la lectura del JSON persistido antes del rate limiter y guardar los resultados en `_QUOTE_CACHE` con TTL corto para que el primer paint de cada símbolo no dependa de la red.【F:services/cache/quotes.py†L546-L737】
- Exponer métricas agregadas (avg/95p) del `QuoteBatchStats` y serializarlas a `performance_metrics_15.csv` para correlacionar mejoras con el lazy load de Kaleido.【F:services/cache/quotes.py†L754-L820】

### services/portfolio_view.py
- Añadir un control explícito para omitir `_compute_contribution_metrics` cuando no haya pestañas que lo consuman y almacenar `history_df` en `self._incremental_cache` para rehidratarlo sin recalcular.【F:services/portfolio_view.py†L329-L420】【F:services/portfolio_view.py†L1013-L1019】
- Externalizar `_persist_snapshot` en un worker configurable que pueda degradarse a no persistir durante la ventana crítica de render inicial.【F:services/portfolio_view.py†L988-L1019】

## Plan priorizado
### 🔹 Rápido (1–2 días, riesgo bajo)
- Activar la escritura de métricas en `performance_metrics_14/15.csv` y cargar warm-start de cotizaciones desde disco antes del refresh. Impacto estimado: −1.8 s (`quotes_refresh`) + visibilidad inmediata.【F:services/cache/quotes.py†L546-L737】
- Cachear el resultado de `build_quote_batches` por hash de dataset/filtros para eliminar recomputos triviales. Impacto: −0.4 s en `quotes_refresh`.【F:controllers/portfolio/load_data.py†L119-L142】【F:services/portfolio_view.py†L969-L1019】

### 🔸 Medio (1 – 2 sprints, riesgo moderado)
- Dividir `build_portfolio_viewmodel` en capa mínima + cálculos diferidos y condicionar contribuciones/historial a pestañas activas. Impacto combinado: −1.6 s en `portfolio_view.apply` y −3 s percibidos en startup.【F:controllers/portfolio/portfolio.py†L1061-L1184】【F:controllers/portfolio/portfolio.py†L640-L776】
- Ajustar `AdaptiveBatchController` para reducir sublotes cuando la media supere 650 ms y paralelizar la precarga de FX/favoritos durante login. Impacto: −2.0 s (`quotes_refresh`) + −2.0 s startup.【F:services/cache/quotes.py†L741-L820】【F:controllers/portfolio/portfolio.py†L1091-L1113】

### 🔺 Alto (requiere análisis profundo)
- Reemplazar la persistencia síncrona de snapshots por colas/background jobs y considerar eliminar gráficos poco usados (ej. composición avanzada) hasta que el usuario abra la pestaña. Impacto: −0.8 s (`portfolio_view.apply`) + −2.5 s UI inicial.【F:services/portfolio_view.py†L988-L1019】【F:controllers/portfolio/portfolio.py†L640-L776】
- Explorar precálculo incremental vía servicio dedicado para cotizaciones de alta frecuencia, almacenando resultados en Redis/S3 y cargándolos en caliente. Impacto esperado: llevar `quotes_refresh` a ~2.5 s.

La combinación de los cambios rápidos y medios debería dejar la carga inicial por debajo de los 10 s (6.9 s → ~4.5 s en lógica + 1.8 s UI ≈ 6.3 s totales), mientras que las iniciativas de alto impacto consolidan el margen para crecer sin degradar tiempos.
