# Arquitectura de renderizado y ciclos de rerun

Esta nota describe el pipeline de renderizado de la aplicación Streamlit, los componentes clave y los gatillos de rerun que generan rehidrataciones en cascada o pantallas en blanco durante la navegación por los paneles de monitoreo.

## Pipeline principal
```mermaid
graph TD
    A[app.py.main()] --> B[bootstrap.init_app()]
    B --> C[ui.orchestrator.render_main_ui()]
    C --> D{¿Autenticado?}
    D -->|No| E[ui.orchestrator._render_login_phase() → safe_stop]
    D -->|Sí| F[controllers.portfolio.portfolio.render_portfolio_section()]
    F --> G[controllers.portfolio.load_data.load_portfolio_data()]
    G --> H[services.portfolio_view.PortfolioViewModelService.*]
    F --> I[ui.sidebar_controls.render_sidebar()]
    F --> J[controllers.portfolio.render_* sections]
    J --> K[ui.lazy.runtime.lazy_fragment() → safe_rerun]
    C --> L[ui.health_sidebar.render_health_monitor_tab()]
    L --> M[ui.panels.iol_raw_debug.render_iol_raw_debug_panel() → st.stop]
```
* `bootstrap.init_app` establece sesión, logging y registra el evento “App initialized…”, lo que ocurre en cada rerun porque el script completo vuelve a ejecutarse.【F:bootstrap/config.py†L70-L101】
* `render_main_ui` arranca un identificador de flujo, aplica el guard de monitoreo, programa la precarga científica y renderiza encabezado, sidebar de salud y el cuerpo del portafolio.【F:ui/orchestrator.py†L459-L520】
* `render_portfolio_section` coordina la carga de datos, filtros, viewmodel y secciones de resumen, tabla y gráficos. Las métricas extendidas se derivan en un hilo que reejecuta la app al completar.【F:controllers/portfolio/portfolio.py†L1914-L2160】
* Los fragmentos perezosos usan `ui.lazy.runtime` para esperar hidratación; si la sesión no está lista, se solicita un rerun “lazy_fragment_ready”.【F:ui/lazy/runtime.py†L723-L748】
* La barra de salud permite abrir paneles (p.ej. 🔍 IOL RAW). `_render_active_monitoring_panel` pinta el panel y emite un `st.stop()` para cortar el resto del script, lo que explica los pantallazos en blanco cuando el rerun tarda en completar.【F:ui/health_sidebar.py†L2088-L2164】

## Árbol de componentes/páginas
1. `render_main_ui`
   - Encabezado (`ui.header.render_header`).
   - Sidebar de salud (`ui.health_sidebar.render_health_monitor_tab`).
     - Atajos de monitoreo (`_render_monitoring_shortcuts`).
     - Panel activo (`_render_active_monitoring_panel` → módulos `ui.panels.*`).
   - Cuerpo principal (`controllers.portfolio.portfolio.render_portfolio_section`).
     - Carga de dataset (`controllers.portfolio.load_data.load_portfolio_data`).
     - Sidebar de controles (`ui.sidebar_controls.render_sidebar`).
     - Viewmodel (`services.portfolio_view.PortfolioViewModelService`).
     - Secciones de resumen, tabla y charts (`controllers.portfolio.render_summary_section` / `render_table_section` / `render_charts_section`).
     - Fragmentos lazy (`ui.lazy.runtime.lazy_fragment`).
   - Pie (`ui.footer.render_footer`).

## Gatillos de rerun y refresh
| Causa | Archivo/Línea | Call stack típico | Condición de disparo | Impacto | Mitigación propuesta |
| --- | --- | --- | --- | --- | --- |
| Hidratación inicial “hydration_unlock” | `ui/orchestrator.py` L692-L695 | `render_main_ui → safe_rerun` | Primer render tras login, desbloquea componentes diferidos | Rerun inmediato tras completar métricas | Cachear flag en `st.session_state` para evitar rerun extra cuando no hay componentes diferidos. |
| Auto-refresh del portafolio | `ui/orchestrator.py` L700-L710 | `render_main_ui → safe_rerun("portfolio_autorefresh")` | `refresh_secs` cumplido y `last_refresh` expira | Rerun completo aunque haya panel de monitoreo activo | Pausar cuando `is_monitoring_active()` sea verdadero y reanudar tras cerrar panel. |
| Lazy metrics extendidas | `controllers/portfolio/portfolio.py` L2091-L2138 | `render_portfolio_section → _compute_and_rerun → safe_rerun("portfolio.extended_metrics_ready")` | Hilo de métricas termina y dataset coincide | Rerun en hilo secundario, puede superponerse con paneles | Cancelar cuando `freeze_heavy_tasks()` esté activo; usar `st.session_state` para aplazar si monitoreo está activo. |
| Lazy fragment hydration | `ui/lazy/runtime.py` L741-L748 | `lazy_fragment → _trigger_fragment_context_rerun → safe_rerun` | Guardián detecta contexto incompleto | Bucles hasta que guardian hidrata, provoca flashes | Propagar estado de guardián al monitor y añadir debounce configurable. |
| Reset del sidebar | `ui/sidebar_controls.py` L337-L342 | `render_sidebar → safe_rerun("sidebar_reset")` | Usuario pulsa “Reiniciar filtros” | Rerun inmediato | Ejecutar reset dentro de contenedor `st.form` para consolidar cambios y evitar reruns múltiples. |
| Logout forzado | `application/auth_service.py` L169-L191 | `logout → safe_rerun("auth_logout_force_login")` | Logout exitoso | Regresa a login, limpia sesión | Dejar placeholder persistente mientras re-renderiza login para evitar pantalla vacía. |
| Botón de logout | `ui/actions.py` L32-L70 | `render_action_menu → safe_rerun("logout_requested")` + `safe_stop` | Acción de usuario | Rerun + stop, puede cortar layout | Mostrar banner persistente antes del stop y diferir rerun si monitoreo activo. |
| Sincronización de apariencia | `ui/ui_settings.py` L92-L107 | `_sync_setting → safe_rerun("ui_settings_sync")` | Cambios de layout/theme | Rerun inmediato (dos intentos por compatibilidad) | Reemplazar por `st.experimental_update_query_params` o batching de cambios. |
| Dashboard de performance | `ui/tabs/performance_dashboard.py` L215-L220 | `render_performance_dashboard_tab → safe_rerun("performance_dashboard_refresh")` | Botón manual | Rerun del tab completo | Ejecutar refresh en hilo y actualizar tabla vía `st.dataframe` sin rerun global. |
| Monitoreo inline | `ui/health_sidebar.py` L2149-L2164 | `_render_active_monitoring_panel → st.stop()` ⚠️ | Panel activo (incl. 🔍 IOL RAW) | Corta render de resto del layout → pantallazo en blanco durante rerun | Reemplazar `st.stop()` por contenedores condicionales (placeholder/persistencia) y mantener header/footer renderizados. |
| Fragment guardian | `ui/controllers/portfolio_ui.py` L46-L65 & `ui/lazy/runtime.py` L302-L335 | `get_fragment_state_guardian → wait_for_hydration` | Rehidratación de bloques guardados | Reruns encadenados si guardia detecta inconsistencia | Añadir timeout/debounce (ver P1) para evitar loops cuando monitoreo está activo. |

> Nota: cualquier mutación de `st.session_state` (p. ej. filtros, toggles) provoca rerun implícito; el guardian y los safe wrappers sólo lo hacen explícito para registro y trazabilidad.【F:ui/sidebar_controls.py†L337-L349】【F:shared/debug/rerun_trace.py†L38-L89】

## Métricas por flujo
| Flujo | Paso | Duración (ms) | Notas |
| --- | --- | --- | --- |
| Render portafolio (A) | bootstrap_and_preload | 3 200 | Skeleton y precarga científica inicial.【F:perf/flow_portfolio_timeline.csv†L1-L5】 |
| | fetch_quotes | 8 427 | `quotes.fetch_bulk` posterior a filtros.【F:perf/flow_portfolio_timeline.csv†L1-L5】 |
| | render_viewmodel | 7 500 | Construcción de snapshot básico/extendido.【F:perf/flow_portfolio_timeline.csv†L1-L5】 |
| | ui_tabs_hydration | 5 253 | Hidratación de tabs y métricas pendientes.【F:perf/flow_portfolio_timeline.csv†L1-L5】 |
| Monitoreo 🔍 IOL RAW (B) | capture_snapshot | 1 250 | Fetch + sanitización del payload crudo.【F:perf/flow_monitoring_iol_raw.csv†L1-L4】 |
| | render_panel | 800 | Render JSON paginado y métricas del panel.【F:perf/flow_monitoring_iol_raw.csv†L1-L4】 |
| Retorno desde monitoreo (C) | panel_cleanup | 450 | Limpia `_monitoring_active_panel` y telemetría `monitoring.exit`.【F:perf/flow_return_to_portfolio.csv†L1-L4】 |
| | rerender_monitoring_hub | 620 | Rehidrata shortcuts y centro de control.【F:perf/flow_return_to_portfolio.csv†L1-L4】 |
| | portfolio_resume | 980 | Rehidrata portafolio tras cerrar panel.【F:perf/flow_return_to_portfolio.csv†L1-L4】 |

Los decoradores `@timeit` capturan métricas detalladas en hotspots como la precarga científica, la construcción del viewmodel y el fetch de cotizaciones, escribiendo `timings_<flow_id>.csv` cuando `DEBUG_TIMELINE=1` (ver `shared/debug/timing.py`).【F:ui/helpers/preload.py†L181-L204】【F:services/portfolio_view.py†L2262-L2306】【F:services/cache/quotes.py†L858-L899】

## Pantallas en blanco y cascada de reruns
* `_render_active_monitoring_panel` invoca `st.stop()` al final del render, lo que impide que el resto del layout (header, skeletons) quede en pantalla; durante el rerun sólo se ve un lienzo vacío.【F:ui/health_sidebar.py†L2149-L2164】
* Mientras un panel está activo, el guard de monitoreo sólo pausa la precarga científica y trabajos post-login, pero no bloquea los reruns automáticos (autorefresh, guardian de fragmentos, hilos de métricas).【F:ui/orchestrator.py†L499-L507】
* Los hilos de métricas (`portfolio.extended_metrics`) y los fragmentos lazy continúan emitiendo reruns y pueden pisar `_monitoring_active_panel`, rebotando a la vista principal y provocando flashes blancos cuando el panel intenta restablecerse.【F:controllers/portfolio/portfolio.py†L2091-L2160】【F:ui/lazy/runtime.py†L741-L748】

## Mensajes “App initialized …” duplicados
El log se emite dentro de `bootstrap.init_app`. Dado que Streamlit vuelve a ejecutar `app.py` completo en cada rerun, `init_app` se ejecuta repetidamente y registra el mensaje para cada ciclo.【F:bootstrap/config.py†L70-L101】 Una bandera en `st.session_state` o un `logging.once` evitaría duplicados.

## Plan de mejoras priorizado
### P0 — Imprescindible
1. **Mantener layout persistente al renderizar paneles de monitoreo**: envolver `_render_active_monitoring_panel` en un contenedor que pinte placeholders y reemplazar `st.stop()` por un flag de retorno; así, header/sidebar permanecen visibles mientras se completa el rerun.【F:ui/health_sidebar.py†L2088-L2164】
2. **Desactivar reruns automáticos cuando `is_monitoring_active()`**: extender los checks existentes para pausar `portfolio_autorefresh`, hilos de métricas extendidas y `lazy_fragment_ready` hasta que el panel se libere.【F:ui/orchestrator.py†L699-L710】【F:controllers/portfolio/portfolio.py†L2091-L2160】【F:ui/lazy/runtime.py†L741-L748】

### P1 — Alto impacto
1. **Debounce de guardian/fragmentos**: añadir control de frecuencia en `wait_for_fragment_context_end` para no disparar reruns consecutivos cuando la hidratación tarde >500 ms.【F:ui/lazy/runtime.py†L723-L748】
2. **Externalizar métricas pesadas a singletons**: mover inicializaciones costosas (p. ej. `PortfolioService`, `TAService`) a `st.singleton` o fábricas perezosas para reducir el costo por rerun.【F:controllers/portfolio/portfolio.py†L1975-L2007】
3. **Throttle de sincronización UI**: reemplazar los dos `safe_rerun` consecutivos de `_sync_setting` por un diff sobre `st.session_state` y `st.experimental_update_settings`, evitando loops al alternar layout.【F:ui/ui_settings.py†L92-L107】

### P2 — Higiene
1. **Actualizar configuraciones Plotly**: reemplazar `plotly.io.kaleido.scope.*` por `plotly.io.defaults` en los módulos de charts para evitar deprecations futuros.【F:controllers/portfolio/charts.py†L1-L40】
2. **Instrumentar Kaleido guard**: al detectar monitoreo activo, omitir exportes a imagen para evitar bloqueos cuando Kaleido no está disponible (ya aparece warning en tests).【F:ui/panels/iol_raw_debug.py†L34-L88】
3. **Centralizar logging “App initialized”**: proteger la emisión mediante `st.session_state.setdefault("_startup_logged", True)` o `logging.Logger.isEnabledFor` para limpiar el ruido en logs.【F:bootstrap/config.py†L70-L101】

