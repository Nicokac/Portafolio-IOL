# Secuencia al abrir 🔍 IOL RAW

```mermaid
sequenceDiagram
    participant U as Usuario
    participant Sidebar as ui.health_sidebar.render_health_monitor_tab
    participant Guard as _activate_monitoring_panel
    participant Renderer as _render_active_monitoring_panel
    participant Panel as ui.panels.iol_raw_debug.render_iol_raw_debug_panel
    participant ST as Streamlit runtime

    U->>Sidebar: Click en “🔍 IOL RAW”
    Sidebar->>Guard: safe_page_link(render_fallback)
    Guard->>ST: session_state["_monitoring_active_panel"] = {module, attr, label}
    Guard->>Renderer: Solicitar render inline
    Renderer->>Panel: Importa módulo y ejecuta renderer
    Panel->>Panel: st.button captura snapshot RAW (spinner, métricas)
    Panel->>ST: Actualiza session_state con snapshot/tiempos
    Panel-->>Renderer: Retorna
    Renderer->>ST: st.stop() ⚠️ (corta el resto del layout)
    ST-->>U: Rerun completo del script
    U->>Renderer: Pulsar “Volver al monitoreo”
    Renderer->>Guard: _clear_active_monitoring_panel()
    Guard->>ST: Limpia state y registra monitoring.exit
    Renderer->>ST: st.stop() ⚠️ (forza rerun de retorno)
```

* `render_health_monitor_tab` detecta la selección activa y delega en `_render_active_monitoring_panel`, que importa el renderer, pinta encabezado y ofrece el botón de regreso.【F:ui/health_sidebar.py†L2088-L2164】
* `_activate_monitoring_panel` persiste la selección en `st.session_state` y emite telemetría `monitoring.enter`, de modo que el rerun siguiente conoce qué panel rehidratar.【F:ui/health_sidebar.py†L1989-L2015】
* `render_iol_raw_debug_panel` captura el snapshot RAW cuando se pulsa el botón, guarda métricas de fetch/parse y renderiza tabla + JSON paginado.【F:ui/panels/iol_raw_debug.py†L190-L266】
* Al finalizar, `_render_active_monitoring_panel` llama a `st.stop()` (⚠️) para evitar que el cuerpo principal siga renderizando; esto provoca pantallas en blanco hasta que el siguiente rerun reconstruye la UI.【F:ui/health_sidebar.py†L2149-L2164】
* El botón “Volver al monitoreo” limpia el panel activo y vuelve a llamar a `st.stop()`, por lo que el retorno al portafolio implica otro rerun completo.【F:ui/health_sidebar.py†L2102-L2164】

