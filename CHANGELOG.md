# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Servicios `services.iol_exchange_rates.get_exchange_rates` y `services.iol_ratios_service.get_ceear_ratio` para cachear cotizaciones de `/estadocuenta` y ratios CEDEAR desde `/Titulos`, con TTL de 30 minutos y pruebas dedicadas sobre payloads reales.
- Modo seguro de valorización (`SAFE_VALUATION_MODE`) con telemetría estructurada (proveedor, `fx_aplicado`, `ratioCEDEAR`) y tooltip de advertencia en la UI cuando se utilizan cotizaciones estimadas de proveedores externos.
- feat: Introduced PORTFOLIO_TOTALS_VERSION to invalidate outdated portfolio summaries and enforce recalculation after valuation logic updates.

## v0.9.6.0 — BOPREAL Consistency Hotfix
- Reescalado forzado del campo “ultimo” para bonos BOPREAL ARS truncados.
- P/L y porcentajes coherentes con los datos oficiales de InvertirOnline.

### Changed
- UI: Human-readable asset type labels via shared formatter helper.
- UI summary validation and FX consistency.
- Cleanup: eliminamos la suite `application/test` y duplicados en `infrastructure/test`, junto con el repositorio local de
  portafolios sin uso, reduciendo dependencias cruzadas fuera de `tests/`.

### Dashboard visual simplification & cash semantics
- Reorganised the portfolio summary into stacked cards separating totales and liquidez, con tooltips para tipo de cambio y efectivo consolidado.
- Añadido selector ARS/USD que reutiliza una única estructura de `PortfolioTotals`, recalculando métricas y caption aclaratorio sobre /estadocuenta.
- Encabezado simplificado con solo Oficial y MEP, resaltando la cotización activa usada en los totales y evitando renders innecesarios mediante `summary_hash`.

### Testing cleanup after asset-type simplification
- Actualizamos la suite de UI para reutilizar los contenedores reales de `FakeStreamlit`, eliminando stubs obsoletos y restaurando la cobertura de pruebas tras la simplificación de tipos de activos.

### Testing and UI consistency after direct IOL type adoption
- Se reforzó la suite de UI y aplicación para comprobar que los tipos entregados por IOL (`Cedear`, `Acciones`, `Bono`, `Letra`, `FCI`) se propaguen sin alias en filtros, tablas, panel de riesgo y exportaciones, verificando la consistencia entre DataFrame, totales y vistas.

### Fixed
- fix(cash-scale): normalize redundant USD→ARS conversion when consolidating cash totals from `/api/v2/estadocuenta`.
- Fix: conditional bond scaling and USD cash display normalization.

## v0.9.5.1-hotfix1 — Streamlit Compatibility
- Eliminado el uso directo del argumento `alignment` en `TextColumn` para compatibilidad con versiones previas de Streamlit.
- No hay cambios funcionales en la exportación CSV.

## v0.9.5.0 — CSV Export Dashboard (Comparativa IOL)
### Added
- Panel "📊 Comparativa IOL" en la interfaz principal con tabla alineada al layout de InvertirOnline y botón de exportación directa en formato CSV oficial.
- Helper `application.portfolio_service.to_iol_format` para mapear `calc_rows()` y `PortfolioViewModelService` al esquema IOL, reutilizable por otras integraciones y cubierto por pruebas dedicadas.
- Suite de pruebas que valida estructura, codificación UTF-8 con BOM y la presencia del botón de descarga en el nuevo panel.

## v0.9.4.1 — UI Version Sync
- Sincronizada la versión visible en toda la aplicación.
- Actualizados metadatos de build y fecha de release.
- Sin cambios funcionales ni en dependencias.

## 0.9.4.0 — Auditoría de consistencia y verificación cruzada
### Added
- Helper `validate_portfolio_consistency` para contrastar `valor_actual`, `ppc`, `pl`, `pl_%` y `valorizado` entre `calc_rows` y el payload oficial, registrando desvíos `[Audit]` y adjuntando resultados en `df_view.attrs`.
- `PortfolioViewModelService` inyecta los chequeos de consistencia al finalizar `_compute_viewmodel_phase`, expone `inconsistency_count` en el snapshot, marca el dataset como `stale` cuando hay desvíos y publica telemetría `portfolio_consistency` vía `shared.telemetry.log_metric`.
- Suite de pruebas `tests/services/test_portfolio_consistency.py` para validar la detección de desvíos, la ausencia de falsos positivos y la propagación del bloque de auditoría en el snapshot.

### Changed
- Se agregó `log_metric` al pipeline de telemetría compartida para emitir métricas simples reutilizando el backend CSV por defecto.

## 0.9.3.0 — Corrección integral de BOPREAL ARS
### Fixed
- Corrigimos la valuación de bonos BOPREAL en ARS reescalando `ultimo`, `valor_actual` y P/L dentro de `calc_rows`, registrando la auditoría del factor aplicado.
- Ajustamos el post-merge del modelo de vista para detectar símbolos `BPO` en ARS con precios truncados, reescalar los totales y etiquetar la corrección en `audit` junto con un log `[Audit]` explícito.

### Added
- Pruebas dedicadas que cubren el reescalado runtime de BOPREAL y el parche post-merge de `PortfolioViewModelService`.

## 0.9.1 — Refactor estructural y linting global
- Configurados linters Ruff, Flake8 y Black.
- Limpieza y reordenamiento de imports.
- Unificación de typing y docstrings.
- Eliminado código legacy y duplicado.
- Resolución de warnings de deprecación.

## 0.9.0.1-patch3 — Test Discovery sin UI pesada
- Añadidas guardas UNIT_TEST en módulos UI para prevenir render en import.
- Se vació tests/__init__.py y se agregó stub de Streamlit.
- Configurado pytest.ini con testpaths, norecursedirs y marcador "integration".
- Entorno offline validado y suite libre de dependencias de red.

## 0.9.0.1-patch4 — Stable Offline Fixtures & Deterministic Cache
- Añadido stub coherente de IOLClient con datos simulados.
- Reexportados atributos mínimos de cache para compatibilidad.
- Normalizados asserts en tests de integración offline.

## 0.9.0.1-patch2 – Offline Fixtures & Stable Cache
- Introducido stub global para aislar red durante pytest.
- Eliminados fixtures obsoletos que invocaban API real.
- Estabilizado entorno de pruebas previo a linting.

## 0.9.0.1-patch1 – Compatibility Shim
- Restored temporary exports (st, IOLAuth, record_fx_api_response) in services.cache.
- Preserved backward compatibility for legacy modules pending refactor.

## 0.9.0.1 – Hotfix: Detección y eliminación de código duplicado
- Consolidación de funciones redundantes entre application, controllers y services.
- Limpieza de constantes duplicadas y normalización de helpers.
- Validación de compatibilidad total con la versión 0.9.0.

## 0.9.0 – Fase 7.0 Codebase Cleanup
- Eliminado código obsoleto y duplicado según arquitectura de capas.
- Consolidado helpers y normalizado imports.
- Mejorada mantenibilidad y tiempo de build (–15 %).
- Sin cambios funcionales.

## [0.8.9.1] — Hotfix 6.1.1
### Fixed
- Sanitizamos los atributos (`DataFrame.attrs`) generados en `calc_rows` para eliminar objetos no serializables (locks, métodos, módulos) evitando el `TypeError: cannot pickle '_thread.RLock' object` al clonar la vista de posiciones.
- Añadimos la prueba `tests/test_attrs_serialization.py` que garantiza que `copy.deepcopy` funciona correctamente sobre el DataFrame enriquecido y documenta la regresión cubierta para futuras auditorías.

## [0.8.9.0] — Market fallback for BOPREAL ARS
### Added
- `IOLClient.fetch_market_price()` consulta los endpoints de cotización de títulos (`/Cotizacion` y `/CotizacionDetalle`) con reintentos controlados y devuelve el último precio disponible o el promedio bid/ask cuando corresponde.
- Nueva prueba `tests/test_bopreal_market_fallback.py` que cubre el flujo truncado (≈200 k ARS), la revaluación de mercado (~19.9 M ARS) y la propagación de `quotes_hash` en la auditoría.

### Changed
- `calc_rows` detecta precios truncados (`ultimoPrecio` < 10 000) para BOPREAL ARS, aplica la revaluación directa de mercado, etiqueta `pricing_source = "market_revaluation_fallback"` y registra `override_bopreal_market` junto con `market_price_source`, `timestamp_fallback` y `quotes_hash` en `attrs['audit']`.
- `controllers.portfolio.load_data` y `apply_filters` propagan el `market_price_fetcher` autenticado hacia el pipeline de valoración.
- Se incrementa `PORTFOLIO_TOTALS_VERSION` → 6.1 y se versiona el paquete a 0.8.9.0 para invalidar snapshots previos y difundir el nuevo cálculo en la UI.
## [0.8.8.1] — Hotfix BOPREAL valuation cache invalidation
### Fixed
- Se fuerza la revaluación de bonos BOPREAL en ARS incluso cuando el payload marca `pricing_source=valorizado`, ampliando los proveedores confiables y ajustando la auditoría para preservar el factor ×100 sin intervención manual.
- Incremento de `PORTFOLIO_TOTALS_VERSION` (→ 6.0) para invalidar snapshots con escalas erróneas y refrescar totales en UI tras aplicar la corrección automática.

## [0.8.8.0] — Fase 6.0 — Forced Revaluation Patch BOPREAL ARS
### Changed
- `calc_rows` aplica un factor `×100` sobre `ultimoPrecio` y `valor_actual` para series BOPREAL en ARS provenientes de IOL, etiquetando la fila con `pricing_source=override_bopreal_forced` y preservando el monto corregido aunque el payload traiga valores truncados.
- Auditoría enriquecida en `attrs['audit']['bopreal']` con el precio original, el factor aplicado y el valor ajustado para facilitar la trazabilidad del override.
- `detect_bond_scale_anomalies` detecta precios truncados en BOPREAL aun cuando `scale==1`, estimando el impacto real con el nuevo factor forzado.
- El parche post-merge de `PortfolioViewModelService` respeta overrides previos (`pricing_source=override_bopreal_forced`) y evita recalcular el valor real.

### Added
- Pruebas unitarias que cubren la revaluación forzada, casos donde no debe aplicarse el override y la preservación del ajuste tras invalidaciones por `quotes_hash`.

## [0.8.7.0] — Refresco proactivo desde endpoints IOL
### Changed
- Forzamos `PortfolioDataFetchService.get_dataset(force_refresh=True)` inmediatamente después de autenticar al usuario de IOL, registrando la traza `auth_refresh_forced` para auditoría y etiquetando los snapshots como `source=live_endpoint` cuando los datos provienen del endpoint.
- El fingerprint del dataset ahora incorpora `quotes_hash`, invalidando los caches incrementales del viewmodel cuando cambian las cotizaciones y evitando que `_incremental_cache` reutilice bloques obsoletos.
- `PortfolioViewModelService` propaga `quotes_hash` en el metadata del snapshot y ajusta el pipeline para recalcular `calc_rows` aun cuando el payload de posiciones no cambia.

### Added
- Nuevas pruebas unitarias (`test_force_refresh_after_login`, `test_quotes_hash_invalidation`) que aseguran el refresco forzado tras login y la invalidación de dataset cuando solo se actualizan las cotizaciones.

## v0.6.5 — Fase 5.5
- Fix: corrección de escala para bonos BOPREAL (BPOA7–BPOC7)
- Ajuste en `scale_for()` para discriminar moneda ARS vs USD
- Añadido registro "override_bopreal_ars" en auditoría de escalas
- Incremento de `PORTFOLIO_TOTALS_VERSION` → 5.5

## [0.8.6] — Post-merge Sanity Patch BOPREAL ARS
### Changed
- Añadido parche post-merge en `PortfolioViewModelService` para recalcular `valor_actual` de BOPREAL ARS tras fusionar datasets, conservando la valuación forzada (~19.9 M ARS).
- Se evita que el `valorizado` del payload sobrescriba la corrección en snapshots cacheados, etiquetando `pricing_source` como `override_bopreal_postmerge` y registrando la decisión en `audit.scale_decisions`.
- Incremento de `PORTFOLIO_TOTALS_VERSION` → 5.8 y versionado del paquete a 0.8.6.0.

## [0.8.5] — Revaluación forzada BOPREAL ARS
### Changed
- `calc_rows` ignora `valorizado` de payload para series BOPREAL en ARS, priorizando `ultimoPrecio` y recalculando totales con la escala 1.0.
- Auditoría de escalas ahora etiqueta las filas BOPREAL con `override_bopreal_ars_forced_revaluation` e incluye `valorizado_rescaled` como fuente.
- Incremento de `PORTFOLIO_TOTALS_VERSION` → 5.7 para invalidar snapshots previos y propagar la nueva valuación.

### Fixed
- Las letras BOPREAL (BPOA7–BPOD7) reflejan ~19.9 M ARS en `valor_actual`, evitando el rezago de ~199 k heredado del payload.

## [0.8.4] — Validación post-fix BOPREAL y sincronización de snapshots
### Added
- Script `python -m scripts.check_bond_scale --offline` para validar la eliminación de escalas anómalas reutilizando datasets cacheados.
- Comparativa automática entre snapshots v0.8.3 y v0.8.4 con delta de `valor_actual`, `costo` y `pl` por símbolo.
- Documentación del flujo de auditoría en `docs/valuation_normalization_plan.md`.

### Changed
- `PortfolioViewModelService` invalida el cache incremental cuando cambia `PORTFOLIO_TOTALS_VERSION`, forzando el recálculo de totales.
- Se publicó el plan de verificación para normalización de valuaciones y checklist de QA.

### Fixed
- Residuo de cache heredado que conservaba `scale=0.01` para BPOC7.
- Valuación y P/L de series BOPREAL sincronizados con los montos oficiales de IOL.

## [0.8.3] — Reconciliación de efectivo y tasa de cambio
### Changed
- `calculate_totals` evita el doble conteo del efectivo al detectar saldos duplicados entre filas IOLPORA/PARKING y `_cash_balances`, preservando la visibilidad en la tabla pero sin inflar los totales combinados.
- `PortfolioTotals` propaga `usd_rate` y la UI de totales muestra el desglose de ARS/USD junto con el tipo de cambio informado por `/estadocuenta`, incluyendo un tooltip que aclara si corresponde a la cotización oficial, MEP o es desconocida.

## [0.8.2] — Clasificación y valorizado alineados con API IOL
### Changed
- `classify_asset` ahora devuelve tanto la etiqueta normalizada como el texto original provisto por IOL, y `calc_rows` propaga las columnas `tipo_estandar` y `tipo_iol` hacia la vista final para mantener trazabilidad.
- La normalización conserva `titulo.tipo` y `titulo.descripcion` en columnas dedicadas y reutiliza `activos[].valorizado` como respaldo cuando faltan cotizaciones externas.
- Las métricas de P/L usan los valores de IOL (`valorizado`, `variacionDiaria`) cuando no hay precios frescos, manteniendo consistencia con la API oficial.

## [0.8.1] - Cash Flow dinámico (sincronizado con IOL)
### Added
- Integración con `/api/v2/estadocuenta` para calcular el efectivo disponible en ARS y USD, refrescando tokens automáticamente.
- Nuevos totales del portafolio (`total_cash_ars`, `total_cash_usd`, `total_cash_combined`) y UI actualizada para mostrar el desglose y el total combinado.
- Exportaciones y snapshots enriquecidos con el detalle de efectivo, preservando los históricos y compatibilidad con Money Market.

### Enriquecimiento de portafolio con metadatos de IOL
- La normalización del portafolio conserva `moneda`, `plazo`, `ultimoPrecio`, `variacionDiaria`, `tienePanel` y `riesgo`, con fallback seguros cuando la API no provee los campos.
- Las valuaciones reutilizan `ultimoPrecio` y `variacionDiaria` originales como respaldo cuando no hay cotizaciones externas disponibles.

## [0.8.0] - UI Minimalista y Reorganización
### Overview
Primera entrega de la nueva interfaz minimalista y reorganización visual de Portafolio-IOL.

### Changes
- Moved duplicated sidebar components to the Monitoreo tab.
- Simplified the home screen to show only login, title, and footer.
- Removed “Resumen de release” section from the footer.
- Reorganized sidebar logic and orchestrator layout.
- Prepared UI for upcoming monitoring dashboard and typed login state.

### Technical
- ✅ Lint: passes cleanly
- ⚠️ Typing: legacy modules pending cleanup
- ⚠️ Tests: partial Streamlit stub dependencies remain

## [0.7.2] - Clean Final Release
### Overview
Consolidation and final cleanup of the Portafolio-IOL codebase after modular refactoring (phases 1–6).

### Changes
- Removed all redundant and legacy tests from ui/test and test/ directories.
- Unified fixtures under tests/fixtures/.
- Completed full lint and formatting compliance (ruff check . passes 100%).
- Pruned dead code and duplicate utilities across controllers, services, and shared.
- Refactored predictive_engine and data modules with full type consistency.
- Cleaned scripts and tools, adding python -m tools entrypoint.
- Modernized UI components (login, header, health sidebar) and normalized HTML markup.
- Updated documentation under docs/testing.md to reflect new structure.

### Technical status
- ✅ Lint: passes cleanly
- ⚠️ Typing: residual mypy issues (legacy modules)
- ⚠️ Tests: external-dependency warnings (Kaleido/Streamlit)

### Notes
This release concludes the 0.7.x cleanup cycle and prepares the foundation for 0.8.0, 
which will focus on type completion, test modernization, and performance regression tracking.

## [Unreleased]
### Added
- Endpoints `/cache/status`, `/cache/invalidate` y `/cache/cleanup` con autenticación y métricas consolidadas del caché de mercado.
- Cache observability & metrics integration.
- Caché incremental para resumen, tablas y gráficos del portafolio con TTL intradía y marca de tiempo visible en la UI.
- Telemetría de subetapas del portafolio expuesta en el panel de diagnósticos.
### Changed
- `api/main.py` incluye el router de caché y los tests cubren limpieza e invalidación del backend en memoria/persistente.
- Render diferido por pestaña en el portafolio con caché de contenido y telemetría de latencia por pestaña activa.

### Removed
- `portfolio_comparison` module y controles de comparación de snapshots del portafolio.

## [0.7.1] — 2025-10-21
### Changed
- Pulido final de la infraestructura de tests tras consolidación de stubs (v0.7.0).
- Refactor menor en api/routers/__init__.py con loader lazy para evitar ciclos.
- Normalización de imports en suite de UI e integración.
- Documentación y naming consistentes en tests/fixtures/.

### Fixed
- IndentationError en tests/ui/test_portfolio_ui.py tras limpieza global.
- Warnings de import y duplicación de fixtures resueltos.

### Notes
- La suite de pruebas ahora es modular, determinística y totalmente reutilizable.
- Próximo objetivo: extender cobertura hacia controladores de datos y endpoints externos.

## [0.7.0] — 2025-10-21

### Added
- Bootstrap modular en `bootstrap/startup.py` y `bootstrap/config.py` que prepara cachés, telemetría y factories compartidas para UI, API y jobs batch.
- Panel de health modularizado (`ui/health_sidebar*.py`) con proveedores dedicados en `services.health` y métricas diferenciadas por superficie.
- Esquema de telemetría unificado en `shared/telemetry.py`, `shared/visual_cache_prewarm.py` y `shared/snapshot.py`, incluyendo `build_signature`, `dataset_hash` y métricas lazy de UI.

### Changed
- Flujo de autenticación reorganizado en `controllers/auth` y `services/auth` para desacoplar la emisión de tokens y el refresco incremental del runtime UI.
- Layout del caché reestructurado: `services/cache/*` delega en factories por dominio y documenta TTL consistentes con el bootstrap.
- Runtime de UI desacoplado en `ui/lazy/` y `ui/controllers/` con factories específicas para fragmentos (ej. `ui/lazy/table_fragment`).
- Dependencias clave fijadas: `streamlit-javascript==0.1.5`, `plotly==6.3.1`, `kaleido==0.2.1`, `streamlit-vega-lite==0.1.0` y sincronizadas en `pyproject.toml`, `requirements.txt` y `requirements.lock`.

### Fixed
- Visibilidad de fragmentos lazy al sincronizar `st.session_state` con los nuevos factories, evitando renderizados en blanco en tablas y gráficos.
- Advertencias de Kaleido/Plotly al forzar el renderer correcto durante exportaciones y documentar el fallback en la UI.

### Removed
- Se oficializa la retirada de alias legacy y reexports redundantes en `controllers/__init__`, `ui/__init__` y capas intermedias, alineando la documentación con el estado del código.
- Imports implícitos del runtime antiguo en `app.py` y `application/__init__.py`, reemplazados por inicialización explícita vía bootstrap.

### Testing
- `pip install -r requirements.txt`
- Instalación reproducible con entorno virtual (`python -m venv .venv`) y regeneración de `requirements.lock`.
- Imports de `streamlit`, `plotly`, `streamlit_javascript`, `streamlit_vega_lite` y `kaleido.scopes.plotly` verificados en modo bare.

### Known Issues
- El flujo lazy de Streamlit puede mostrar el warning `missing ScriptRunContext` al ejecutar scripts en modo bare; el bootstrap lo documenta pero no lo oculta automáticamente.
- Exportaciones Plotly dependen de Chromium cuando se usa el renderer `browser`; Kaleido sigue siendo el camino recomendado y queda monitoreado en el healthcheck.

### Environment
- Python 3.10+ con `streamlit-javascript==0.1.5`, `streamlit-vega-lite==0.1.0`, `plotly==6.3.1`, `kaleido==0.2.1` y `vega-lite` provisto por el bundle de `streamlit-vega-lite`.
- Usar `requirements.lock` para despliegues inmutables y evitar drift de dependencias en CI/CD.

## 🩹 Portafolio IOL v0.6.22-patch2 — Fix lazy reruns & Skeleton singleton (Febrero 2026)

### 🚑 Hotfix
- Los triggers diferidos de tabla y gráficos ahora usan `st.session_state['load_table']` y `st.session_state['load_charts']`, evitando reruns completos de Streamlit y reusando los placeholders existentes.
- El sistema de skeletons se inicializa una única vez por sesión, registra la primera pintura inmediatamente y muestra un skeleton base antes de iniciar tareas pesadas.
- La capa de exportación omite totalmente Kaleido en modo `browser`, sin reintentos en segundo plano cuando Chromium no está disponible.

### 🛠 Internals
- `_prompt_lazy_block` reemplaza `st.button` por controles persistentes (`toggle`/`checkbox`) y sincroniza las banderas con el almacén dataset-aware para mantener una sola telemetría `portfolio.lazy_component` por dataset.
- `app.py` inserta el skeleton inicial antes de cargar dependencias y conserva `ui_first_paint_ms` en `st.session_state` para métricas de arranque.
- `shared.skeletons.initialize` devuelve un booleano indicando si la sesión ya estaba inicializada, protegiendo contra logs duplicados.

### 🧪 Tests
```bash
pytest -q --override-ini addopts='' tests/ui/test_streamlit_lazy_fix.py
pytest -q --override-ini addopts='' tests/performance/test_rerun_prevention.py
```

## 🧩 Portafolio IOL v0.6.22 — Lazy Charts + Fix rehidratación de tabla (Febrero 2026)

### 🚀 Cambios principales
- Estado diferido persistente para tabla y gráficos usando `st.session_state["lazy_blocks"]` y banderas dataset-aware (`load_table`/`load_charts`) que evitan rehidrataciones y placeholders duplicados tras cada `rerun`.
- Sistema de skeletons estabilizado: los placeholders se marcan una sola vez por sesión y los contenedores se reutilizan sin reinicializar al volver a presionar "Cargar tabla" o "Cargar gráficos".
- Lazy-load extendido a las visualizaciones del portafolio (líneas, barras y heatmap) con placeholders progresivos y telemetría coherente (`lazy_loaded_component=chart`).
- Telemetría visual reforzada (`ui_first_paint_ms`, `ui_total_load_ms`, `lazy_load_ms`) con encabezados homogéneos en los CSV y validaciones automáticas bajo 10 s para renders completos.
- Fallback global para Kaleido cuando Chromium no está disponible, forzando `plotly.renderers.default = "browser"` y registrando el cambio del renderer.

### 🛠 Internals
- `controllers.portfolio.portfolio` conserva las banderas diferidas por hash de dataset, evita bucles de rehidratación y sincroniza el caché visual con los nuevos placeholders persistentes.
- `shared.export` detecta la ausencia de Chromium antes de inicializar Kaleido, documenta el switch del renderer y degrada la exportación a imagen de forma segura.
- `shared.telemetry` añade los campos visuales al header estándar y garantiza que `lazy_loaded_component` y `lazy_load_ms` se serialicen en todos los CSV.

### 🧪 Tests
```bash
pytest -q --override-ini addopts='' tests/ui/test_streamlit_lazy_charts.py
pytest -q --override-ini addopts='' tests/performance/test_visual_stability.py
```

## 🩹 Portafolio IOL v0.6.21-patch1 — Skeletons visibles y fallback de Kaleido (Enero 2026)

### 🚑 Hotfix
- Skeletons visibles al entrar en el tab “Portafolio”, con placeholders que se actualizan automáticamente al cumplirse las condiciones diferidas (`st.session_state["load_table"]`).
- Logging explícito de cada render de skeleton (`🧩 Skeleton render called for …`) para diagnosticar la secuencia de placeholders.
- Fallback de exportación Plotly usando el renderer `browser` cuando Kaleido falla o Chromium no está disponible, evitando gráficos en blanco.
- Telemetría visual reactivada (`skeleton_render_ms`, `ui_first_paint_ms`) en los CSV para monitorear el tiempo hasta la primera pintura.
- Prevención de estados en blanco re-renderizando tabla y gráficos al completarse el lazy-load y sincronizando el placeholder con el dataset.

### 🧪 Tests
```bash
pytest -q --override-ini addopts='' tests/ui/test_streamlit_skeletons_patch1.py
pytest -q --override-ini addopts='' tests/performance/test_lazy_render_fallback.py
```

## 🧩 Portafolio IOL v0.6.20 — Render diferido de componentes pesados (Diciembre 2025)

### 🚀 Cambios principales
- El resumen del portafolio se muestra al instante mientras que la tabla principal y los gráficos intradía/heatmap se cargan bajo demanda mediante botones dedicados.
- El arranque registra tiempos de carga diferidos por componente y los asocia al hash del dataset para monitorear el impacto en `startup.render_portfolio_complete`.

### 🛠 Internals
- `render_basic_tab` mantiene `st.session_state["lazy_blocks"]` con los estados `pending`/`loaded`, renderiza placeholders persistentes y registra telemetría `portfolio.lazy_component` para cada carga diferida.
- `shared.telemetry` incorpora las columnas `lazy_loaded_component` y `lazy_load_ms` en los CSV de métricas y normaliza el encabezado de `performance_metrics_14.csv`/`performance_metrics_15.csv`.
- El controlador limpia el estado diferido al cambiar de usuario y evita renderizar tablas/gráficos hasta que el usuario interactúa con la UI.

### 🧪 Tests
```bash
pytest -q tests/ui/test_streamlit_lazy_loading.py
pytest -q tests/performance/test_lazy_component_overhead.py
```

## 🧩 Portafolio IOL v0.6.19 — Renderización incremental de placeholders (Noviembre 2025)

### 🚀 Cambios principales
- El tab de portafolio reutiliza placeholders persistentes para resumen, tabla y gráficos, evitando reconstrucciones del DOM cuando el dataset no cambia.
- Las actualizaciones parciales registran `incremental_render` y `ui_partial_update_ms`, permitiendo medir la latencia de refrescos incrementales.
- Los KPIs del resumen se muestran inmediatamente mientras que tabla y gráficos se actualizan progresivamente usando referencias almacenadas en `st.session_state["render_refs"]`.

### 🛠 Internals
- `render_basic_tab` conserva referencias de contenedores en sesión, sincroniza el hash del dataset y actualiza cada sección con los nuevos helpers incrementales del servicio de viewmodel.
- `services.portfolio_view` incorpora `update_summary_section`, `update_table_data` y `update_charts` para refrescar componentes existentes sin invocar `empty()`.
- `shared.telemetry` y `performance_metrics_15.csv` incluyen las columnas `incremental_render` y `ui_partial_update_ms` para correlacionar los beneficios de la renderización parcial.
- Se persisten métricas de refresco incremental en `st.session_state` y se limpian junto con el caché visual al cambiar de usuario.

### 🧪 Tests
```bash
pytest -q tests/ui/test_streamlit_incremental_render.py
pytest -q tests/performance/test_incremental_overhead_reduction.py
```

## 🧩 Portafolio IOL v0.6.18 — Limpieza de caché visual por sesión (Noviembre 2025)

### 🚀 Cambios principales
- La UI limpia automáticamente el caché visual cuando el usuario cambia de cuenta o cierra sesión, evitando placeholders con datos obsoletos.
- El portafolio registra en telemetría el indicador `visual_cache_cleared` para correlacionar reinicios del layout con métricas de performance.

### 🛠 Internals
- `infrastructure.iol.auth` expone `get_current_user_id()` y sincroniza `st.session_state['last_user_id']` tras login/logout para que la UI detecte cambios de usuario.
- `render_portfolio_section` invalida `cached_render`/`dataset_hash` al detectar cambios de usuario, loguea el evento `controllers.portfolio.session` y propaga la bandera `visual_cache_cleared`.
- `shared.telemetry` agrega la columna `visual_cache_cleared` en `performance_metrics_15.csv` para mantener consistencia en los reportes.

### 🧪 Tests
```bash
pytest -q tests/ui/test_streamlit_cache_reset.py
pytest -q tests/ui/test_streamlit_cache_reuse.py
```

## 🧩 Portafolio IOL v0.6.17 — Caché visual por hash del dataset (Noviembre 2025)

### 🚀 Cambios principales
- El portafolio reutiliza el resumen, la tabla y los gráficos cuando el hash del dataset no cambia, evitando repintados completos en Streamlit.
- Los placeholders de cada sección se persisten en `st.session_state["cached_render"]`, reduciendo la rehidratación del layout.

### 🛠 Internals
- `render_portfolio_section` calcula y conserva `dataset_hash`, controla el caché visual por dataset y registra la telemetría `portfolio.visual_cache` con `reused_visual_cache`.
- `shared.telemetry` incorpora la columna `reused_visual_cache` y `portfolio_ui` expone métricas de caché visual en la telemetría del runtime.

### 🧪 Tests
```bash
pytest -q tests/ui/test_streamlit_cache_reuse.py
pytest -q tests/performance/test_optimization_recommendations.py
```

## 🧩 Portafolio IOL v0.6.16 — Optimización media: viewmodel diferido y cálculos on-demand

### 🚀 Cambios principales
- El portafolio ahora construye un snapshot mínimo en la primera pasada y calcula métricas extendidas bajo demanda, mostrando los datos esenciales en menos tiempo.
- El render de la pestaña principal admite un modo `lazy_metrics` que muestra un spinner mientras las métricas completas se materializan y re-renderiza automáticamente al finalizar.

### 🛠 Internals
- `PortfolioViewModelService` separa las fases básica y extendida (`build_minimal_viewmodel` y `compute_extended_metrics`), marca métricas pendientes y reutiliza resultados desde `_incremental_cache`.
- La persistencia de snapshots se ejecuta en background y registra la nueva fase `snapshot.persist_async`; se añadieron las fases `portfolio_view.apply_basic` y `portfolio_view.apply_extended` en la telemetría unificada.
- `render_portfolio_section` coordina la ejecución diferida, registra banderas `lazy_metrics` y dispara `st.experimental_rerun` cuando las métricas extendidas están listas.

### 🧪 Tests
```bash
pytest -q tests/services/test_portfolio_view_lazy_metrics.py tests/ui/test_portfolio_lazy_render.py
```

## 🧩 Portafolio IOL v0.6.15 — Optimización rápida de carga (Noviembre 2025)

### 🚀 Cambios principales
- Reactivada la escritura de `performance_metrics_14.csv` y `performance_metrics_15.csv` con telemetría normalizada para `quotes_refresh`, `portfolio_view.apply` y `startup.render_portfolio_complete`.
- El arranque del portafolio registra el tiempo total de login/render en los nuevos CSV y conserva el hash del dataset para correlacionar mejoras.

### 🛠 Internals
- `services/cache/quotes` precarga el caché en memoria desde disco (_warm-start_) antes del primer refresh, registra telemetría consolidada y expone `set_active_dataset_hash` para correlacionar métricas.
- `controllers/portfolio/load_data` memoiza `build_quote_batches` por hash de dataset/filtros y sincroniza el dataset hash con la telemetría de `quotes_refresh`.
- `services/portfolio_view` registra la duración y el ratio de memoización de `portfolio_view.apply` en los CSV de métricas.
- `shared/telemetry` centraliza la escritura de métricas con cabecera común y logging consistente.

### 🧪 Tests
```bash
pytest -q tests/performance/test_quick_optimizations.py
```

## 🧩 Portafolio IOL v0.6.13 — Carga diferida de Kaleido (Noviembre 2025)

### 🚀 Cambios principales
- Kaleido se carga de manera diferida tras el render del portafolio, evitando bloquear el arranque de Streamlit.
- Se registra la métrica `kaleido_load_ms` en `performance_metrics_15.csv` para monitorear la latencia del import.

### 🛠 Internals
- `shared.export` realiza un import perezoso de Kaleido con instrumentación de métricas y advertencias coherentes.
- `services.environment` expone `mark_portfolio_ui_render_complete` y persiste el lazy-load en la nueva telemetría.

### 🧪 Tests
```bash
pytest -q tests/shared/test_export_lazy_kaleido.py
pytest -q tests/services/test_environment_imports.py
# streamlit run app.py --server.headless true --server.port 8501  # opcional manual
```

## 🧩 Portafolio IOL v0.6.12 — Render del portafolio sin histórico pesado (Noviembre 2025)

### 🚀 Cambios principales
- Eliminado el gráfico "Evolución histórica del portafolio" y la lógica asociada para priorizar métricas en vivo.
- Simplificada la sección principal del portafolio dejando solo resumen, métricas consolidadas y P/L diario.
- Actualizada la telemetría a `performance_metrics_14.csv` con campos `portfolio_tab_render_s`, `streamlit_overhead_ms` y `profile_block_total_ms`.

### 🛠 Internals
- Ajustado el caché incremental de pestañas para almacenar la nueva métrica de render sin depender de `portfolio_history`.
- Se generó el encabezado inicial de `performance_metrics_14.csv` para habilitar la nueva telemetría.

### 🧪 Tests
```bash
pytest -q tests/ui/test_portfolio_ui.py
pytest -q tests/controllers/test_portfolio_filters.py
pytest -q tests/ui/test_portfolio_charts_rendering.py  # opcional, marcada como lenta
```

## 🧩 Portafolio IOL v0.6.10 — Optimización de rendimiento y diagnóstico avanzado (Octubre 2025)

### 🧠 Rendimiento y Telemetría
- Implementado auditor de caché de cotizaciones (`scripts/quotes_cache_audit.py`) con métricas de batch y ratio de aciertos (hit ratio 82.5 %, stale 13.3 %).
- Detectados sublotes lentos en tickers de Bonos/Energía (> 1 s).
- Añadidas métricas `quotes_refresh_total_s`, `avg_batch_time_ms`, `quotes_hit_ratio`, `stale_ratio` en `performance_metrics_9.csv`.

### 💾 Cache y Renderizado del Portafolio
- Instrumentado `services.portfolio_view` con métricas de memoización y fingerprints (`portfolio_cache_hit_ratio`, `cache_miss_count`, `fingerprint_invalidations`).
- Nuevas pruebas de regresión en `tests/services/test_portfolio_view_cache.py` y `tests/controllers/test_portfolio_filters.py`.

### 🎨 Overhead de Streamlit
- Incorporada métrica `streamlit_overhead_ms` para aislar la latencia del layout.
- Añadidas visualizaciones de sparklines y consejos automáticos en `ui/tabs/performance_dashboard.py`.
- Cobertura extendida con `tests/ui/test_performance_dashboard.py`.

### 🧩 Nuevos artefactos
- `scripts/quotes_cache_audit.py`
- `docs/fixtures/telemetry/quotes_refresh_logs.jsonl`
- `docs/fixtures/telemetry/portfolio_view_cache.json`
- `performance_metrics_9.csv`

### Notas
- Esta versión completa la etapa de diagnóstico de rendimiento iniciada en v0.6.8 y sienta las bases para el tuning adaptativo planificado en v0.6.11.
- No se introducen cambios funcionales visibles al usuario final, solo mejoras de rendimiento y observabilidad.

## 🧩 Portafolio IOL v0.6.9 — Simplificación estructural

**Fecha:** 15 de octubre de 2025
**Tipo:** Refactor / Cleanup

### 🚀 Cambios principales
- Eliminado el módulo **“Empresas con oportunidad”**, incluyendo sus controladores, servicios y pestañas de UI.  
- Simplificado el layout principal de Streamlit: ahora solo se muestran **Portafolio**, **Recomendaciones** y **Monitoreo**.  
- Removidas dependencias obsoletas y referencias en `services/health.py`, `ui/health_sidebar.py` y `controllers/opportunities.py`.  
- Eliminados más de **700 líneas de código** y **10 archivos de prueba** relacionados con el screener de oportunidades.  
- Reducción del tiempo de arranque y carga de dependencias en modo Streamlit-only.  

### 🧪 Tests
```bash
pytest tests/test_health_sidebar_rendering.py
pytest tests/ui/test_layout_components.py
pytest tests/ui/test_login_startup_subsecond.py
pytest tests/integration/test_snapshot_export_flow.py
```

### 🗂️ Archivos modificados
- app.py
- ui/health_sidebar.py
- controllers/__init__.py
- ui/tabs/recommendations/__init__.py
- services/health.py
- shared/config.py
- pyproject.toml
- shared/version.py

### 🗑️ Archivos eliminados
- application/screener/*
- ui/tabs/opportunities.py
- controllers/opportunities.py
- tests/application/test_opportunities_*.py
- tests/controllers/test_opportunities_*.py
- tests/ui/test_opportunities_ui.py

## [v0.6.8] — Streamlit 1.50 + Predictive optimization (2025-10-17)
### Added
- Compatibilidad con Streamlit 1.50 adoptando `st.metric` con `chart_data` para renderizar sparklines de CPU, RAM y duración en tiempo real.
- Exportación dedicada `performance_sparkline.csv` con los datos de las métricas recientes para análisis fuera de la app.
- Registro de métricas `performance_metrics_7.csv` y `performance_metrics_8.csv` con nuevas series `predictive_runtime_s` y `batch_success_rate`.
- Cobertura de interfaz y stubs actualizados para validar los parámetros extendidos y el flujo histórico/promedio del dashboard de performance.

### Changed
- Gradiente dinámico verde/rojo en las métricas según tendencia y toggle persistente en `st.session_state` para alternar entre “Última ejecución” y “Promedio histórico”.
- Servicio `adaptive_predictive_service` instrumentado con `profile_block()` por fase, procesamiento en sub-batches concurrentes (~10 tickers) y liberaciones parciales del lock con `lock_timeout_s=60`.
- Reducción de la retención del lock adaptativo (<30s en escenarios normales) y manejo de reintentos más seguro durante fetch/persist.
- Orden descendente en el gráfico de asignaciones de Markowitz aprovechando el nuevo parámetro `sort="descending"`.
- Consolidación de logs al modo Streamlit-only y supresión de advertencias de Kaleido.
- Modernización de la suite de pruebas (`pytest` actualizado sin coverage) para evitar bloqueos en CI.

### Testing
- `pytest -q tests/ui/test_performance_dashboard.py`
- `pytest -q tests/application/test_adaptive_predictive_service.py`
- `pytest -q tests/domain/test_adaptive_cache_lock.py`
- `streamlit run app.py --server.headless true --server.port 8501`

## [v0.6.6-patch11e-1] — Lazy preload refactor (2025-10-16)
### Changed
- Split startup in pre-login and post-login phases: the preload worker now starts paused and resumes ~500 ms after the first authentication, keeping login under 1 s (p95) while warming `pandas`, `plotly`, and `statsmodels` before dashboards render.
- Added a Streamlit gate (`ui.helpers.preload.ensure_scientific_preload_ready`) that displays a short spinner until the scientific preload finishes, preventing premature imports of heavy controllers.
- Exposed structured telemetry and Prometheus gauges for `preload_total_ms` plus per-library timings; `/metrics` now shows `preload_pandas_ms`, `preload_plotly_ms`, and `preload_statsmodels_ms`.
- Introduced a bytecode warm-up step (`scripts/warmup_bytecode.py`) executed from `scripts/start.sh`, along with deployment defaults that enable `.pyc` generation (`PYTHONDONTWRITEBYTECODE=0`).

## [v0.6.6-patch11d-2] — Lazy startup optimisation (2025-10-16)
### Changed
- Startup optimization: reduced initial render time from 8–12 s to under 2 s through lazy imports and asynchronous preload of heavy dependencies.

## [v0.6.6-patch11d] — Implementación de lazy imports y optimización de arranque inicial (2025-10-16)
### Changed
- Implementación de lazy imports y optimización de arranque inicial.

## [v0.6.6-patch11c] — UI tests alignment & risk stub fixes (2025-10-16)
### Changed
- Alineamos la suite de UI con el flujo de render diferido en tres etapas, verificando métricas de fingerprint cache entre renderizados consecutivos.
- El panel de diagnósticos ahora muestra siempre los hits/misses y la última clave utilizada del caché de fingerprints.
### Fixed
- Ajustamos los stubs de riesgo para respetar filtros de tipo de activo en los tests y evitar dependencias de históricos reales.

## [v0.6.6-patch11b] — Portfolio fingerprint memoization (2025-10-16)

- Memoised `_portfolio_dataset_key` across portfolio components using an LRU
  cache keyed by `snapshot_id` and dataset filters, eliminating redundant DataFrame
  hashing during a render.
- Recorded fingerprint cache hit/miss telemetry through
  `performance_timer.record_stage("portfolio_ui.fingerprint_cache")` and surfaced
  the stats in the diagnostics panel next to `render_tab.*` timings.
- Added regression coverage to ensure the fingerprint is computed once per
  dataset snapshot, re-used across renders and measurably faster on 5k–10k row
  simulations.

## [v0.6.6-patch11a] — Startup telemetry performance hotfix (2025-10-14)
### Changed
- Eliminamos la actualización redundante del gauge `ui_total_load_ms` en `app.py`, delegando en `record_stage` para evitar escrituras duplicadas y mantener el indicador visible en la UI y `/metrics`.
- `services.startup_logger` ahora utiliza una cola asincrónica con worker dedicado para persistir `logs/app_startup.log`, eliminando bloqueos de I/O en el hilo principal y añadiendo `flush_startup_logger()` para sincronizar en tests y apagado.

## [v0.6.6-patch9b2] — Optimization Nexus (2025-10-13)
- Updated version metadata in shared/version.py
- Marks completion of predictive and quotes optimization cycle
- Stable build for deployment validation (<10s total render time)

## v0.6.6-patch9b1 — Predictive worker async and cache reuse.
### Added
- `application/predictive_jobs` con un worker asíncrono compartido que permite
  `submit()`, `get_latest()` y `status(job_id)` con TTL sincronizado con
  `MarketDataCache`.
- Superficie de `predictive_job_status` y metadatos en `predict_sector_performance`
  para que la UI y los controladores puedan mostrar el progreso del cálculo.
- Spinner informativo en recomendaciones cuando las predicciones se recalculan en
  background.
### Changed
- `predict_sector_performance` reusa el último resultado cacheado mientras una
  corrida nueva se ejecuta en segundo plano, evitando bloqueos de la UI.
- `MarketDataCache` expone `resolve_prediction_ttl` para unificar la caducidad de
  predicciones entre el cache y el worker.

## v0.6.6-patch3f — Deferred market_data_cache import and added safe fallback for missing dependencies during startup.
### Fixed
- Diferimos la importación de `market_data_cache` y proveímos un fallback seguro para iniciar la aplicación cuando faltan dependencias de caché.

## v0.6.6-patch3d — Added persistent startup logger (logs/app_startup.log) to capture detailed import errors before Streamlit masking.
### Added
- Logger de arranque persistente que captura excepciones de importación con PID, versión y traceback completo en `logs/app_startup.log`.

## v0.6.6-patch3c — Fixed persistent import loop between sqlite_maintenance and shared.settings, refactored initialization to runtime-safe phase.
### Fixed
- Broke the circular dependency by deferring the SQLite maintenance configuration until runtime and ensuring Prometheus metrics continue to register safely.
### Changed
- Added `services.maintenance.configure_sqlite_maintenance` to refresh scheduler settings without importing `shared.settings` at module load.

## v0.6.6-patch3b — Fix SQLite maintenance import dependency
### Fixed
- Deferred the SQLite maintenance scheduler imports to avoid circular dependencies during app/bootstrap while preserving Prometheus metrics.
- Added regression coverage that imports `app.py` and validates the lazy scheduler bootstrap.

## v0.6.6-patch2b2 — Cache management documentation & CI integration
### Added
- Guía operativa `docs/cache_management.md` con ejemplos y recomendaciones para la gestión del caché.
- Script `scripts/test_smoke_endpoints.sh` y job de CI que validan los endpoints `/cache/*`, tiempos < 2 s y generan un reporte JSON.
- Nuevos escenarios en `tests/api/test_cache_endpoints.py` que cubren errores de backend, límites y logs estructurados.

## v0.6.5-monitoring-and-observability — Observability layer for performance telemetry.
### Added
- JSON structured telemetry with daily rotation at `logs/performance/structured.log` and optional Redis streaming.
- Prometheus summaries and gauges (duration, CPU, RAM) exposed via the new `/metrics` router with module/label/success labels.
- SQLite persistence helper (`services/performance_store.store_entry`) habilitado automáticamente en `APP_ENV=prod`.
- Streamlit performance dashboard ahora incluye filtros por bloque/tiempo/keywords, percentiles P50/P95/P99, alertas y exportes CSV/JSON.

### Changed
- `services/performance_timer` ahora utiliza `QueueHandler` + `QueueListener` para desacoplar I/O, integra métricas Prometheus y elimina `_flush_logger`.
- El formato de log plano se controla con `PERFORMANCE_VERBOSE_TEXT_LOG` y se simplificó la cabecera del dashboard.

## v0.6.4-patch4b — Security claims and endpoint guardrails.
### Added
- Claims enriquecidos (`iss`, `aud`, `version`, `session_id`) en los tokens Fernet y registro en memoria de sesiones activas.
- Endpoint `/auth/refresh` con rotación automática dentro de los últimos 5 minutos y auditoría `token_refreshed`.
- Suite de pruebas para validar el ciclo de vida de tokens, la protección de `/profile` y la sanitización de logs del broker.

### Changed
- El TTL configurable (`FASTAPI_AUTH_TTL`) queda limitado a 15 minutos y los tokens se revocan automáticamente al hacer logout.
- `/profile` ahora exige autenticación explícita y `/cache` se deshabilita hasta contar con implementación final.
- Los logs de fallos en IOL omiten cuerpos de respuesta, registrando sólo `Auth failed (code=<status>)`.

## v0.6.4-patch4a — Security hardening for credential storage and telemetry.
### Added
- Variable de entorno `FASTAPI_TOKENS_KEY` dedicada a los tokens internos del backend y validación de `APP_ENV`.
- Pruebas de seguridad para impedir fugas de credenciales en la telemetría y para validar políticas de almacenamiento de tokens.

### Changed
- Telemetría de sesión redacta automáticamente tokens, claves y contraseñas antes de registrarse en `analysis_logger`.
- El backend y el broker IOL exigen claves Fernet distintas y abortan el arranque si coinciden.
- `allow_plain_tokens` registra advertencias explícitas y se bloquea automáticamente en `APP_ENV=prod`.

### Added
- Visualizaciones Altair en la pestaña de oportunidades: barra de score promedio por sector y línea temporal de indicadores macro reutilizando el caché del backend.
- Indicador de “preset activo” con recuento de filtros aplicados y selector interactivo de vista (`Sector` ↔ `Tiempo`) en el resumen del screening.

### Changed
- El helper `make_symbol_url` centraliza la construcción de enlaces de Yahoo Finance y se reutiliza en el screener, controlador y UI para evitar duplicación de formato.
- La gestión de presets se simplificó eliminando estados intermedios y aplicando los guardados directamente desde la UI, manteniendo coherencia entre sesión y caché.

### Fixed
- El fallback de enlaces en tablas ahora valida `NaN`/`NA` antes de generar la URL, previniendo vínculos inválidos cuando faltan símbolos.

### Added
- Ejecución concurrente del screener de oportunidades mediante `ThreadPoolExecutor` (8 workers) con métricas por símbolo y respetando `YAHOO_REQUEST_DELAY`.
- Prevalidación `_precheck_symbols` que descarta símbolos con `market_cap`, `pe_ratio` o `revenue_growth` fuera de umbrales antes de solicitar históricos.

### Changed
- El resumen del screener incluye tiempos promedio por símbolo, ratio de descarte de precheck y detalle de errores por ticker para telemetría.

### Fixed
- Se rehízo la capa de caché de `YahooFinanceClient` para evitar dependencias en decoradores in-memory y compartir resultados entre instancias.

## v0.6.4-patch2b — Validation hardening and adaptive UI consolidation.
### Added
- Validación de payload para `/forecast/adaptive` limitando a 10 000 filas o 30 símbolos mediante `AdaptiveForecastRequest`.
- Nueva utilidad `build_adaptive_history` que fusiona históricos reales y sintéticos con cacheo y clipping seguro de predicciones.
- Archivo `ui/utils/formatters.py` centralizando formatos de badges, porcentajes y variaciones para controladores y Streamlit.

### Changed
- El controlador de recomendaciones emite contexto de símbolo/sector/perfil en los logs y expone el estado del badge de caché.
- La pestaña de correlaciones reutiliza `build_adaptive_history_view`, propagando el perfil del inversor y registrando históricos sintéticos cuando corresponde.
- Se estandarizó el uso de formatters en la UI y se consolidó el manejo de estado adaptativo con toasts y métricas consistentes.

### Fixed
- Se truncan valores fuera de rango en `predicted_return_pct` antes de generar históricos adaptativos, registrando advertencias explícitas.
- La UI evita construir payloads vacíos para la simulación adaptativa cuando no hay histórico disponible.

## v0.6.4-patch2a — Predictive caching per símbolo/período y lock adaptativo global.
### Added
- Creado `domain/adaptive_cache_lock.py` con lock global reentrante y diagnósticos de retención/espera para proteger operaciones concurrentes del motor adaptativo.
- Nuevas pruebas `tests/domain/test_adaptive_cache_lock.py` que validan exclusión mutua, reentrancia y warnings por bloqueos prolongados.

### Changed
- `predict_sector_performance` ahora reutiliza `MarketDataCache` para cachear predicciones por símbolo/período, registra métricas vía `update_cache_metrics` y emite trazas con contexto de símbolos/sectores.
- `RecommendationService` comparte el lock adaptativo al consultar predicciones y eliminó ejecuciones redundantes del motor predictivo.
- `simulate_adaptive_forecast` y `update_model` protegen las llamadas a `run_adaptive_forecast` con el nuevo lock, evitando corrupción en archivos Parquet bajo cargas simultáneas.

### Fixed
- Se normalizó la actualización de métricas de caché para evitar lecturas desfasadas y se añadieron advertencias cuando el lock permanece retenido por más de cinco segundos.

## v0.6.4-patch1 — Shared market-data cache, lazy analytics and resilient risk metrics.
### Added
- Introduced `services/cache/market_data_cache.py` to persist historical prices and fundamentals with a shared TTL of 6 horas.
- Persisted `PortfolioService` y `TAService` en `st.session_state` para reutilizar instancias entre renders.
- Agregamos controles de carga diferida para timeline, heatmap y simulación Monte Carlo con feedback de progreso en la UI.
- Incorporamos mensajes UX específicos cuando el portafolio no devuelve posiciones (filtros vs. autenticación).

### Changed
- El análisis avanzado y de riesgo ahora reutiliza históricos/fundamentales cacheados y convierte las métricas a `float32` para evitar columnas `object`.
- El controlador de riesgo omite símbolos individuales cuando fallan los históricos, mostrando el badge “⚠️ Datos incompletos”.

### Fixed
- Se normalizó la gestión de errores parciales en `controllers/portfolio/risk.py`, evitando abortar pestañas completas ante fallas puntuales.
- Los avisos de portafolio vacío ahora guían al usuario sobre filtros activos o problemas de login.

## v0.6.4-perf-diagnostics — Performance telemetry, CPU/RAM logging and QA dashboard.
- Added `services/performance_timer` with optional psutil integration, structured log export and helpers to consume recent entries.
- Instrumented authentication (login & token refresh), portfolio loading, filter application, quote refresh, predictive computations and portfolio UI rendering with CPU/RAM metrics.
- Introduced the Streamlit tab `ui.tabs.performance_dashboard` and controller helpers to surface performance logs in-app.
- Extended diagnostics sidebar navigation, added regression tests for the timer utilities and refreshed documentation/version metadata.

## v0.6.3-patch3 — Hardened Kaleido export, updated Plotly calls to modern API, and silenced deprecated kwargs warnings.
- Wrapped Kaleido initialisation and runtime checks to gracefully disable image export when Chromium is missing.
- Added Chromium availability warning during environment inspection to highlight limited export support.
- Updated Streamlit Plotly invocations to use the modern `width="stretch"` signature with responsive config.

## v0.6.3-patch2 — Added synthetic fixture fallback, updated deprecated Plotly and Pandas calls, and added Kaleido dependency check for graphics export.
- Added automatic synthetic fixture generation for missing backtesting datasets with safe persistence.
- Updated Streamlit Plotly rendering calls to the modern `width="stretch"` signature.
- Filtered empty frames before concatenation in predictive utilities to avoid pandas warnings.
- Logged Kaleido availability during environment inspection to disable exports gracefully when missing.

## v0.6.3-patch1 — Implemented lazy FastAPI import in services/auth to ensure compatibility with Streamlit-only environments.
- Deferred FastAPI imports in `services/auth` with safe fallbacks for Streamlit-only deployments.
- Logged the active mode to differentiate between FastAPI and Streamlit executions.
- Declared FastAPI, Uvicorn, python-multipart, and updated cryptography requirements for consistent deployments.

## v0.6.3-part3c — Secured predictive_engine FastAPI microservice and integrated Engine API badge into Streamlit UI.
- Protected `/engine/predict`, `/engine/forecast/adaptive` y `/engine/history` con autenticación compartida usando `get_current_user`.
- Añadió badge “Engine API active 🔮” en el login al verificar `/engine/info` correctamente.
- Documentó los endpoints del microservicio en el README con ejemplos `curl` autenticados.
- Extendió las pruebas de integración para cubrir los nuevos requisitos de autenticación del engine.

## v0.6.3-part3b — Implemented /engine/predict, /engine/forecast/adaptive, and /engine/history endpoints using predictive_engine integration.
- Added FastAPI endpoints for `/engine/predict`, `/engine/forecast/adaptive`, and `/engine/history` wired to the standalone `predictive_engine` package.
- Serialised pandas outputs via the engine helpers with performance instrumentation for observability.

## v0.6.3-part3a — Added FastAPI engine router with /engine/info endpoint and base structure.
- Added FastAPI engine router with `/engine/info` endpoint and base structure.

## v0.6.3-part2 — Integrated adaptive forecast persistence and vectorized predictive engine.
- Added `predictive_engine.storage` with Parquet/SQLite helpers and warm-start support for the adaptive history.
- Vectorised beta-shift and error computations in the predictive core and exposed `run_adaptive_forecast` with performance metrics instrumentation.
- Updated the adaptive application service to consume the new adapter API and added regression tests for persistence and warm-start flows.

## v0.6.2-part2 — Implemented shared authentication between Streamlit and FastAPI.
- Added a Fernet-based token service reused by Streamlit and FastAPI to issue and validate auth tokens.
- Secured predictive and cache endpoints behind a common `get_current_user` dependency expecting `Authorization: Bearer` headers.
- Streamlit login now issues API tokens, reuses them for backend requests, and documentation explains the unified flow.

## v0.6.2-part1c — Integrated FastAPI backend with UI indicator and test coverage.
- Added root-level aliases for predictive FastAPI endpoints and documented API mode usage.
- Surfaced an "API mode available" badge on the login screen when the backend is reachable.
- Created automated tests for predictive, adaptive forecast and cache status endpoints.

## v0.6.2-part1b — Implemented FastAPI endpoints for predictive, adaptive, and cache services.
- Added predictive `/predict` endpoint and adaptive forecast simulation API with Pydantic schemas.
- Exposed cache statistics endpoint backed by the core cache helpers.
- Delivered placeholder profile summary endpoint returning structured JSON payloads.

## v0.6.2-part1a — Created base FastAPI structure and routers skeleton.
- Introduced the foundational FastAPI app with health endpoint and logging.
- Registered placeholder routers for predictive, profile, and cache services.

## v0.6.1c-part2 — Performance observability instrumentation.
- Added `services/performance_metrics` to capture execution timings and memory deltas for predictive workloads and log them with versioned timestamps.
- Instrumented `predict_sector_performance` and `simulate_adaptive_forecast` with the new tracker and surfaced aggregated metrics via the 🩺 Diagnóstico panel.
- Introduced a diagnostics UI panel with CSV export and cache hit visibility, plus unified logging through the update checker.
- Bumped documentation and version metadata to v0.6.1c-part2.

## v0.6.1c-part1 — Added automated QA/CD tools and coverage pipeline.
- Added local QA orchestration via `nox` with lint, type-check, tests and security sessions.
- Documented QA checklist, coverage template and pipeline summary under `docs/qa/`.
- Updated documentation with coverage badge, QA instructions and coverage/security tooling metadata.

## v0.6.1b-part2 — Added recommendations controller and completed UI modularization.
- Added recommendations controller and completed UI modularization.

## v0.6.1b-part1 — Split major UI sections of the recommendations tab into modular subcomponents.
- Modularized the recommendations UI into `cache_badge`, `simulation_panel` and `correlation_tab` packages.
- Preserved the testing helper `_render_for_test` while delegating rendering to the new package entry point.
- Updated the Streamlit tab to consume the refactored components and refreshed the visible version label.

## v0.6.1a-part1 — Moved predictive cache core logic to services/cache/core.py

## v0.6.1a-part2 — Split quotes and UI adapter from cache monolith
- Quote cache management now lives in `services/cache/quotes.py` with dedicated persistence helpers.
- Streamlit-facing helpers moved to `services/cache/ui_adapter.py`, keeping `services/cache.py` as a thin compatibility layer.


## v0.6.0-patch1 — Navegación segura del panel Acerca de
- Nuevo helper `ui.helpers.navigation.safe_page_link` que verifica el registro de páginas de Streamlit y provee un fallback compatible cuando la página no está disponible.
- El login ahora reutiliza `safe_page_link` y permite abrir el panel “ℹ️ Acerca de” inline como alternativa segura.
- Pruebas de regresión para `safe_page_link` y fumadores del login que cubren tanto el registro de la página como el render inline.

## v0.6.0 — Auto-Restart y Panel Acerca de
- Implementado reinicio automático tras actualización.
- Nuevo panel “ℹ️ Acerca de” con información de sistema y logs recientes.
- Mejoras en la trazabilidad del flujo de actualización.

## v0.5.9 — Mejora del sistema de actualización
- Registro estructurado de verificaciones y actualizaciones.
- Badge azul “Actualizando…” durante el proceso.
- Panel con historial de las últimas actualizaciones.

## v0.5.8 — Mejoras en el sistema de actualización
- Registro de la última verificación de versión (timestamp persistente).
- Enlace directo al changelog de GitHub.
- Badge verde en el login cuando la app está actualizada.
- Botón “Forzar actualización” disponible en el panel avanzado.

## v0.5.7 — Verificador de versión manual
- Nueva función `check_for_update()` con conexión a GitHub.
- Integración en la pantalla de inicio de sesión con confirmación manual de actualización.
- Script local para `git pull` + `pip install --upgrade`.
- Actualizada documentación y metadatos del proyecto.

## v0.5.6-patch2 — Corrección de Plotly y estados Streamlit
- Migradas llamadas a st.plotly_chart() para usar config={"responsive": True}.
- Refactorizado mapeo seguro de estados en _render_cache_status().
- Añadidas pruebas de regresión para cache y Plotly.
- Actualizada versión visible en UI y metadatos del proyecto.

## v0.5.6-patch1 — Corrección de estado inválido en st.status
- Reemplazado color directo por mapeo seguro a estados válidos ('complete', 'running', 'error') en el indicador de caché.
- Añadida prueba de validación de mapeo de estados.

## v0.5.6 — QA y Documentación Consolidada
- Añadido smoke test de `_render_for_test` para ejecución offline.
- Nueva guía de desarrollo `docs/dev_guide.md`.
- Limpieza de documentación legacy (<v0.4.x).
- Validación de versión y flujo offline completo.

## v0.5.4 — Auditoría y QA de la serie 0.5.x
### Added
- `tests/application/test_regression_v054.py` ejecuta el flujo adaptativo completo con fixtures,
  valida MAE adaptativo < MAE estático y asegura ratio de cache ≥ 45 % con reporte Markdown.
- `shared/logging_utils.silence_streamlit_warnings` centraliza filtros de logging/warnings para
  ejecuciones offline sin ruido de Streamlit.
- Reporte de QA en `docs/qa/v0.5.4-validation-report.md` con métricas de cache, tiempos de render
  y sumario de validaciones.

### Changed
- `application.predictive_service` ahora expone snapshots de caché con % de hits y marca temporal
  normalizada, corrigiendo imports faltantes y formalizando el docstring del módulo.
- `services.cache.CacheService` formatea `last_updated` en `YYYY-MM-DD HH:MM:SS`, agrega método
  `stats()` e incrementa la trazabilidad de hits/misses.
- `ui/tabs/recommendations.py` muestra el ratio de hits en porcentaje, última actualización y usa
  el helper de logging compartido para suprimir warnings en modo bare.
- Fixtures de recomendaciones incluyen columna `sector` para consolidar la preparación histórica.

### Fixed
- Se sincronizaron exports de `application.__init__` para evitar importaciones implícitas y ciclos.
- El flujo adaptativo reutiliza caches dedicados con TTL estable, respetando el formato ISO en los
  reportes y evitando residuos tras los tests de regresión.

## v0.5.3 — Métricas extendidas del motor adaptativo
### Added
- `simulate_adaptive_forecast` ahora calcula `beta_shift_avg`, `sector_dispersion` y genera un resumen legible con metadatos de caché.
- Nuevo `export_adaptive_report` produce un reporte Markdown con resumen global, tabla temporal e interpretación de métricas.
- Pestaña **Correlaciones sectoriales** incorpora mini-card de β-shift/σ, botón de exportación y metadata de caché.

### Changed
- `CacheService` registra `hit_ratio` y `last_updated` en formato HH:MM:SS, reutilizados por la UI adaptativa.
- Logging del motor adaptativo reducido a nivel DEBUG para evitar ruido en consola.

## v0.5.2 — Aprendizaje adaptativo y correlaciones dinámicas
### Added
- `application.adaptive_predictive_service` introduce un estado persistente con TTL de 12 horas, cálculo de correlaciones adaptativas vía EMA y simulaciones históricas que reportan MAE, RMSE y bias.
- `tests/application/test_adaptive_predictive_service.py` cubre la evolución temporal del modelo, la persistencia de estado y la reducción de error frente a las predicciones originales.
- Nuevo tab **Correlaciones sectoriales** en `ui/tabs/recommendations.py` con matrices histórica/rolling/adaptativa, resumen de β promedio y dispersión sectorial más las métricas del motor adaptativo.
- `ui/charts/correlation_matrix.py` genera la visualización β-shift y se documenta el flujo en `docs/adaptive_learning_overview.md`.

### Changed
- Insight automático y `_render_for_test()` incorporan los datos adaptativos para exponer β-shift promedio y la correlación dinámica junto al resto de métricas.

## v0.5.1 — Forecasting y Retornos Proyectados
### Added
- `application.predictive_service.predict_sector_performance` con suavizado EMA,
  penalización por correlaciones intrasectoriales y métricas de confianza.
- Columna **Predicted Return (%)** y toggle *Incluir predicciones* en la pestaña
  de recomendaciones, además del contador de hits/misses del caché predictivo.
- Tests unitarios para el motor predictivo y la integración de retornos
  proyectados en `RecommendationService`.

### Changed
- Insight automático recalculado con promedios ponderados por asignación y
  racional extendido que destaca la predicción sectorial.

## v0.5.0-dev — Inicio del ciclo de consolidación predictiva
- Se incorpora `application.backtesting_service.BacktestingService`, reutilizando indicadores de `ta_service` y datos de fixtures para ejecutar backtests sin depender de la API de IOL.
- Nueva `CacheService` con TTL configurable en `services/cache.py` para cachear precios históricos, resultados simulados y adaptadores offline durante la transición a la serie 0.5.x.
- Fixtures offline en `docs/fixtures/default/` (precios con indicadores, perfil base y recomendaciones de ejemplo) que alimentan `_render_for_test()` y las pruebas unitarias.
- Versión sincronizada a `0.5.0-dev` en `pyproject.toml`, `shared/version.py`, README y CHANGELOG manteniendo la compatibilidad funcional de la release 0.4.4.

## v0.4.4 — Perfil inversor persistente y comparativas con benchmarks
- Nuevo `ProfileService` con almacenamiento cifrado que sincroniza tolerancia al riesgo, horizonte e
  inclinación estratégica entre `session_state`, `config.json` y `st.secrets`.
- La pestaña de recomendaciones permite ajustar el perfil mediante selectores dedicados, mostrando un
  badge con el perfil activo y aplicando sesgos en `RecommendationService.recommend()`.
- Bloque comparativo frente a Merval, S&P 500 y Bonos que resume ΔRetorno, ΔBeta y Tracking Error
  usando el nuevo `compute_benchmark_comparison()`.
- Documentación y versión actualizadas para la release 0.4.4, junto con pruebas unitarias de perfil y
  benchmarking.

## v0.4.3 — Recomendaciones exportables y explicadas al instante
- Incorporadas descargas "📤 Exportar CSV" y "📥 Exportar XLSX" con promedios finales de retorno y beta.
- Añadido racional extendido que cuantifica aporte al retorno, impacto en beta y diversificación sectorial.
- Insight automático enriquecido con la detección del sector dominante dentro de las sugerencias.

## v0.4.2 — Simulador de Inversión Inteligente
- Refinado algoritmo de recomendación con límites de peso y balanceo sectorial.
- Agregada visualización gráfica de distribuciones sugeridas (Pie y Barras).
- Implementado simulador de impacto con métricas Antes/Después (valor total, retorno, beta).
- Tests unitarios y lógicos validados por script en entorno QA.
- Pendiente: validar renderizado visual completo cuando el mock de API IOL esté disponible.

## v0.4.0 — Factor & Benchmark Analysis (Dec 2025)

**Fecha:** 2025-12-05

**Novedades principales:**
- Incorporado el módulo de *Análisis de Factores y Benchmark* con métricas de Tracking Error,
  Active Return e Information Ratio directamente en el tablero de riesgo.
- Nuevo servicio `application.benchmark_service` para centralizar cálculos de seguimiento y
  regresiones multi-factoriales con soporte para factores macroeconómicos opcionales.
- Visualización de betas por factor con indicación de R², más exportaciones CSV/XLSX desde el
  controlador de riesgo.
- Cobertura de pruebas unitarias e integradas para los cálculos y la nueva UI, junto con
  documentación actualizada en README y guías de testing.

**QA Check:**
✅ Verificar que el panel “Análisis de Factores y Benchmark” renderice correctamente.
✅ Confirmar coherencia entre Tracking Error y Information Ratio.
⚠️ Cuando no haya datos de benchmark, mostrar aviso de datos insuficientes.

## v0.3.4.4.6 — Clasificación y visualización completa por tipo de activo (Nov 2025)

### Summary
- El heatmap de riesgo ahora genera pestañas para cada tipo de activo detectado en el portafolio
  (CEDEAR, Acciones locales, Bonos, Letras, FCI, ETFs y Otros) aun cuando no existan suficientes
  símbolos para calcular correlaciones, mostrando advertencias contextuales cuando corresponde.
- Se amplió el mapeo canónico de tipos (`_TYPE_ALIASES`) para contemplar variantes frecuentes como
  "Bonos Dólar", "Letras del Tesoro" o fondos money market, manteniendo etiquetas visuales
  estandarizadas.
- Nuevas pruebas en `tests/controllers/test_risk_filtering.py` cubren la presencia de todas las
  pestañas y las advertencias asociadas; README y documentación de testing actualizados junto con el
  incremento de versión a 0.3.4.4.6.

## v0.3.4.4.5 — Local Equity Tab in Risk Heatmap (Nov 2025)

### Summary
- El análisis de correlaciones crea una pestaña dedicada para **Acciones locales**, reutilizando la
  clasificación del catálogo base para separar claramente CEDEARs y renta variable doméstica.
- Se preserva la exclusión de tickers locales al seleccionar el grupo de CEDEARs, evitando que
  LOMA, YPFD o TECO2 aparezcan en matrices cruzadas con instrumentos del exterior.
- Documentación, guías de prueba y materiales de comunicación actualizados para reflejar el
  comportamiento del nuevo heatmap junto con el incremento de versión a 0.3.4.4.5.

## v0.3.4.4.4 — Asset Type Alignment in Risk Analysis (Nov 2025)

### Summary
- El cálculo de correlaciones ahora se apoya exclusivamente en la clasificación del portafolio
  base antes de solicitar históricos, aplicando un mapeo canónico por símbolo para evitar que
  instrumentos de distintos tipos se mezclen en el heatmap.
- Los CEDEARs filtran explícitamente los tickers locales (LOMA, YPFD, TECO2) aunque el payload
  de precios o cotizaciones los etiquete erróneamente, manteniendo matrices homogéneas por
  categoría.
- Se añadieron pruebas de controlador que validan el filtro corregido y la asignación de tipos
  desde el catálogo maestro, junto con documentación y materiales de release actualizados para la
  versión 0.3.4.4.4.

## v0.3.4.4.3 — Risk Heatmap Polishing Pass (Nov 2025)

### Summary
- Elimina del cálculo de correlaciones a los activos con rendimientos de varianza nula o indefinida,
  evitando coeficientes erráticos y matrices singulares.
- Los heatmaps de correlación ahora muestran títulos contextualizados por tipo de activo (por
  ejemplo, "Matriz de Correlación — CEDEARs"), lo que refuerza la segmentación aplicada en los
  filtros del análisis de riesgo.
- README y materiales de release actualizados para documentar el descarte de columnas sin
  movimiento y el nuevo etiquetado por grupo.

## v0.3.4.4.2 — Vertical Sidebar Layout (Nov 2025)

### Summary
- Reorganiza los grupos de controles de la barra lateral en tarjetas apiladas verticalmente, manteniendo títulos, captions y tooltips consistentes.
- Mejora la lectura de filtros y acciones al asignar una fila completa a cada bloque (Actualización, Filtros, Moneda, Orden, Gráficos y Acciones) con padding uniforme.
- Conserva el feedback visual al aplicar filtros, resaltando únicamente la sección afectada sin alterar la lógica del formulario.

## v0.3.4.4.1 – Header Centering & Cleanup Hotfix (Nov 2025)

### Summary
- Centra el hero principal del dashboard y elimina el bloque redundante de "Enlaces útiles" del encabezado, manteniendo el bloque únicamente en el footer.
- Refina la composición visual inicial para que el título, subtítulo y resumen FX queden alineados sin alterar datos ni microinteracciones previas.

## v0.3.4.4 — UX Consistency & Interaction Pass (Nov 2025)

### Summary
- Consolidación de microinteracciones en la barra lateral y los formularios clave: estados _hover_, enfoque visible y tooltips sincronizados con los controles de presets y filtros.
- Confirmaciones in-app homogéneas: toasts, banners y contadores sincronizados entre el panel principal y la pestaña **Monitoreo** para que cada acción de screening muestre feedback inmediato.
- Ajustes de ritmo visual y tiempos de carga: skeletons y spinners consistentes en dashboards, exportaciones y healthcheck para reducir saltos al cambiar de contexto.

### Added
- Puerta de calidad de seguridad en CI que ejecuta `bandit` sobre el código crítico y `pip-audit`
  sobre los requirements para bloquear vulnerabilidades antes del merge.

### Documentation
- `docs/testing.md` actualizado con los comandos oficiales de auditoría (`bandit` y `pip-audit`) y
  la exigencia de cobertura configurada por defecto en `pytest`.

### Tests
- Configuración de `pytest` actualizada para imponer cobertura sobre `application`, `controllers` y
  `services` en cada ejecución, alineada con la nueva puerta de seguridad de CI.

## v0.3.4.3 — Layout Consolidation & Sidebar Unification (Nov 2025)

### Summary
- Se creó la pestaña **Monitoreo** para alojar el healthcheck completo y se añadió un badge global de estado en la cabecera.
- Todos los controles del portafolio, el panel de control y las preferencias de apariencia se reubicaron en la barra lateral bajo un contenedor colapsable.
- La vista principal del portafolio aprovecha el ancho completo con espaciado uniforme tras retirar el panel superior.
- El footer incorpora un bloque de enlaces útiles con acceso directo a documentación y soporte.

### Documentation
- `README.md`, `docs/testing.md` y `banners/README` describen el nuevo flujo con sidebar unificado y la pestaña de Monitoreo.
- La versión de la aplicación se actualizó a 0.3.4.3 en código y materiales de release.

## v0.3.4.2 — Visual Polish Pass (Nov 2025)

### Summary
- Incremento de padding y márgenes clave para asegurar el respiro visual del panel superior y las
  tarjetas de KPIs en resoluciones medianas.
- Tarjetas contrastadas y tipografía reajustada para reforzar la jerarquía de información en los
  indicadores del dashboard.
- Alineación central consistente de los bloques del header y filtros, evitando saltos laterales en el
  selector de riesgo.
- Ajustes en el footer: espaciado, alineación de enlaces y consistencia con la narrativa “Observabilidad
  operativa”.

## v0.3.4.1 — Layout y Filtros de Análisis de Riesgo (Nov 2025)

### Summary
- El panel superior del dashboard de análisis se reposicionó como una franja horizontal fija, sobre
  la grilla de contenido, liberando espacio lateral para los gráficos.
- Se adoptó un layout de ancho completo en la vista principal para priorizar la lectura del heatmap
  de riesgo y los indicadores asociados.
- Los filtros del heatmap incorporan un selector por tipo de instrumento que permite acotar el
  análisis sin depender de la antigua barra lateral.

### Documentation
- `README.md`, `docs/testing.md` y `banners/README` reflejan el nuevo layout horizontal y la
  liberación del sidebar para controles.

## v0.3.4.0 — UI Experience Refresh (Oct 2025)

### Summary
Consolidación del roadmap UX/UI iniciado en la release 0.3.30.13, con foco en accesibilidad, jerarquía visual y coherencia narrativa dentro del panel de usuario.  
La versión 0.3.4.0 representa una evolución estética y funcional del dashboard, manteniendo la estabilidad del backend y el enfoque en observabilidad operativa.

### Highlights
- **Refinamiento visual (Fase 1):** Reestructuración del encabezado en formato hero de dos columnas, nuevo resumen FX y reorganización del bloque de seguridad en la pantalla de login.  
- **Experiencia interactiva (Fase 2):** Conversión del menú de acciones en panel persistente con tooltips, layout de doble columna con control fijo y health sidebar expandible con secciones delimitadas.  
- **Personalización guiada (Fase 3):** Chips visuales para filtros activos, vista previa de exportaciones con métricas y fallbacks Kaleido reforzados, además de badges dinámicos por pestaña.  
- Unificación de estilos, tamaños de fuente y espaciado entre secciones clave.  
- Coherencia entre encabezado, footer y panel lateral bajo la narrativa “Observabilidad operativa”.

### Testing
- Validado con `python -m compileall` en módulos UI y layout actualizados.  
- Ejecución parcial de `pytest --override-ini addopts=''` confirmando integridad de componentes.  
- CI visual manual en entorno Streamlit (QA offline).

## [0.3.30.13] — Observabilidad reforzada en ejecución

### Added
- Telemetría de entorno con snapshot automático de variables críticas (Python, Streamlit, Kaleido y
  binarios del sistema) visible desde la UI y embebida en `analysis.log` para acelerar diagnósticos
  remotos.
- Rotación automática de logs con compresión diaria y retención configurable que evita que `~/.portafolio_iol/logs`
  crezca sin control en estaciones con screenings intensivos.
- Controles de dependencias al inicio que advierten por UI y CLI cuando falta Kaleido, faltan
  binarios de exportación o la versión de Python está fuera del rango soportado.

### Changed
- Barra lateral y pantalla de login muestran un bloque de "Observabilidad" con accesos rápidos para
  descargar snapshots de entorno y el paquete de logs rotados.
- Documentación de descarga guiada dentro de la UI para educar a los analistas sobre cómo compartir
  snapshots, logs y artefactos de exportación al escalar incidentes.

### Fixed
- Se evitó la sobrescritura silenciosa de `analysis.log` cuando el proceso se relanza en entornos con
  permisos restringidos, delegando la rotación en un handler tolerante a fallas.

## [0.3.30.12.1] — Hotfix: diagnóstico de inicio resiliente

### Fixed
- Se restauró el registro de diagnósticos de inicio para tolerar snapshots mal formados,
  conservar la telemetría en el health sidebar y evitar que el flujo de login falle.

## [0.3.30.12] — Estabilización y Monitoreo de Sesión

### Added
- Timeline de sesión en el health sidebar con `session_tag`, timestamps y origen de cada hito (login,
  screenings, exportaciones) para diagnosticar degradaciones y rebotes de UI sin revisar logs crudos.
- Etiquetas de sesión en `analysis.zip`, `analysis.xlsx` y `summary.csv` para rastrear qué ejecución
  generó los artefactos y correlacionarlos con los eventos registrados en `analysis.log`.

### Changed
- Banners de login/sidebar actualizados para resaltar "Estabilización y monitoreo de sesión" y el nuevo
  badge de timeline visible para QA.
- README, guías de testing y troubleshooting ajustadas para reflejar el monitoreo de sesión, los TTL
  en vivo y los pasos de verificación asociados en pipelines.

### Fixed
- Normalización del `session_tag` almacenado en `st.session_state` para evitar duplicados tras reruns
  y asegurar que los contadores de resiliencia conserven la trazabilidad de cada sesión.

## [0.3.30.11] — Mantenimiento, observabilidad y optimización de logs/cache.

### Changed
- TTL de caché revisado para mantener los paneles cálidos sin sacrificar consistencia ni forzar
  rehidrataciones innecesarias en los screenings nocturnos.
- Panel de health actualizado con métricas de observabilidad que enlazan directamente con
  `analysis.log`, facilitando el seguimiento de degradaciones y alertas proactivas.

### Fixed
- Limpieza del pipeline de logging para eliminar archivos huérfanos y entradas duplicadas en
  `analysis.log`, reduciendo ruido operativo y facilitando auditorías.

## [0.3.30.10.2] - Robust Excel export

### Fixed
- Reforzada la generación de `analysis.xlsx` para reintentar exportes con hojas vacías y conservar
  el archivo dentro de `analysis.zip` aun cuando la primera iteración falle.
- Normalizados los nombres de hojas y encabezados para evitar errores de `ExcelSheetNameError` en
  portafolios con símbolos extensos o caracteres especiales.
- Alineada la conversión de tipos mixtos en columnas numéricas para impedir que se descarten
  registros al aplicar formatos durante la exportación.

## [0.3.30.10.1] - Hotfix entorno Kaleido

### Changed
- Limpieza y resincronización de dependencias en `pyproject.toml` y los requirements planos
  para evitar paquetes redundantes en CI/CD y entornos mínimos.

### Fixed
- Restaurado el fallback de exportación cuando `kaleido` no está disponible: la aplicación
  mantiene los artefactos CSV/Excel, etiqueta el estado en los banners y registra la ausencia
  de PNG para los pipelines.

### Documentation
- README, guías de testing y troubleshooting actualizadas con la release 0.3.30.10.2, el hotfix
  de Kaleido y el mensaje visible en los banners.

## [0.3.30.10] - 2025-10-15

### Fixed
- Se restableció la tubería de logging para que todos los flujos de screening y exportación vuelvan a
  registrar eventos en `analysis.log`, incluyendo los `snapshot_hits`, degradaciones controladas y la
  procedencia de los datos consumidos por la UI.
- Los exports (`analysis.zip`, `analysis.xlsx`, `summary.csv`) vuelven a generarse con el set completo
  de archivos, preservan los timestamps de ejecución y adjuntan la bitácora consolidada en los artefactos
  de CI.

### Documentation
- README, guías de troubleshooting y banners actualizados para recalcar la release 0.3.30.10 y los
  fixes de logging/export que devuelven la trazabilidad a pipelines y operadores.

## [0.3.30.9] - 2025-10-10

### Fixed
- Se reparó el flujo de cotizaciones en vivo: `/Titulos/Cotizacion` vuelve a sincronizarse con
  `/Cotizacion`, respeta el fallback jerárquico y expone el origen real de cada precio en la UI.
- Se corrigió el sidebar para mostrar el estado actualizado del feed live, la versión `0.3.30.9` y la
  salud de los proveedores sin mensajes inconsistentes.

### Added
- Integración del país de origen en el portafolio para habilitar filtros, dashboards y exports
  multi-país en los análisis de cartera.

### Documentation
- README, guías de testing y troubleshooting actualizadas para destacar la release 0.3.30.9, las
  cotizaciones en vivo restauradas y las verificaciones necesarias en banners y pipelines.

## [0.3.30.8] - 2025-10-06

### Added
- Sesiones legacy cacheadas para reutilizar credenciales válidas y reducir latencia al restaurar
  contextos degradados.
- Rate limiting integrado en los clientes de datos para proteger los umbrales de APIs externas y
  evitar bloqueos al ejecutar pipelines intensivos.
- Recuperación automática de valorizaciones recientes cuando la fuente primaria falla, garantizando
  que la UI y los reportes mantengan cifras consistentes.

## [0.3.30.7] - 2025-10-05

### Fixed
- Corrección del fallback jerárquico que perdía el escalón secundario cuando el proveedor primario
  devolvía credenciales inválidas, garantizando que la degradación continúe hasta el snapshot
  persistido.
- Sincronización del banner de login y del health sidebar para reflejar la procedencia real de los
  datos servidos durante la degradación, evitando mensajes inconsistentes.
- Ajuste del contador `snapshot_hits` para propagar correctamente los resultados recuperados por el
  fallback endurecido y mantener la telemetría alineada en dashboards y exportaciones.

### Documentation
- README, guías de testing y troubleshooting actualizadas con la release 0.3.30.7 y los pasos para
  validar los fixes del fallback jerárquico.

## [0.3.30.5] - 2025-10-04

### Fixed
- Se normalizó la publicación de cotizaciones nulas para evitar excepciones en telemetría y dashboards.
- El backend de snapshots ahora se auto-configura en inicializaciones en frío, evitando estados parciales.
- Se restauró el fallback legacy para consultas de mercado cuando el proveedor principal no responde.
- Se reactivó la valorización de portafolios tras interrupciones de caché, garantizando cifras consistentes.

## [0.3.30.4] - 2025-10-04

### Added
- Nuevo endpoint `/Cotizacion` que publica cotizaciones normalizadas para los consumidores internos y externos.

### Fixed
- Manejo reforzado de errores HTTP 500 provenientes de upstream para evitar caídas en dashboards y telemetría.

### Tests
- Prueba de cobertura dedicada que valida los flujos de cotización bajo escenarios de error y resiliencia.

## [0.3.30.3] - 2025-10-04

### Fixed
- Corrección definitiva del backend de snapshots para asegurar que `_ensure_configured()` se ejecute
  antes de cualquier lectura en dashboards o pipelines CI, evitando inicializaciones incompletas.
- Normalización de la firma `IOLClient.get_quote()` y de los flujos de cotizaciones para aceptar
  `(market, symbol, panel)` sin romper la telemetría ni los consumidores existentes.

## [0.3.30.2] - 2025-10-04

### Fixed
- Agregado `_ensure_configured()` en `services/snapshots.py` para evitar errores de inicialización.
- Corregida la firma de `IOLClient.get_quote()` para aceptar `(market, symbol, panel)`.
- Validación completa de persistencia de snapshots y consultas de mercado sin errores.

## [0.3.30.1] - 2025-12-01

### Changed
- Limpieza de escenarios duplicados y migración final de controladores/servicios fuera de
  `infrastructure.iol.legacy`, consolidando el uso de `IOLClientAdapter` y
  `PortfolioViewModelService` como fuentes únicas para la UI y los scripts.
- Ajuste de los pipelines para auditar importaciones legacy con `rg` y reforzar que `pytest` sólo
  recolecte suites modernas.

### Documentation
- README, guía de pruebas y troubleshooting actualizados con la versión 0.3.30.1, instrucciones de
  migración (helpers reemplazados, stub oficial de Streamlit) y comandos para ejecutar suites sin
  módulos legacy.

### Tests
- Checklist de CI actualizada para exigir `pytest --ignore=tests/legacy`, auditorías de importaciones
  legacy y verificación de artefactos (`coverage.xml`, `htmlcov/`, `analysis.zip`, `analysis.xlsx`,
  `summary.csv`).

## [0.3.29.2] - 2025-11-24

### Changed
- Hardening de CI/cobertura alineado con los hitos [CI resiliente 0.3.29.2](https://github.com/Portafolio-IOL/portafolio-iol/milestone/43)
  y [Cobertura exportaciones 0.3.29.2](https://github.com/Portafolio-IOL/portafolio-iol/milestone/44), incorporando validaciones
  cruzadas entre `pytest`, `coverage.xml` y los artefactos de exportación (CSV, ZIP y Excel) para bloquear merges sin evidencia
  de reportes completos.
- El pipeline ahora normaliza la recolección de artefactos (`htmlcov/`, `summary.csv`, `analysis.zip`, `analysis.xlsx`) y marca como
  fallidos los jobs que no adjuntan cobertura o exportaciones esperadas.

### Documentation
- README, guía de pruebas y troubleshooting actualizados para la release 0.3.29.2 con la nueva sección **CI Checklist** y ejemplos
  de exportación alineados a los artefactos `analysis.zip`, `analysis.xlsx` y `summary.csv`.

### Tests
- Checklist de CI incorporada en la documentación para garantizar que `pytest --cov` publique `htmlcov/` y `coverage.xml`, y que las
  suites de exportación validen la presencia de CSV, ZIP y Excel antes de dar por válidos los pipelines.

## [0.3.29.1] - 2025-11-22

### Changed
- Hardening de arquitectura y exportaciones: las validaciones de Markowitz ahora bloquean presets
  inconsistentes y sincronizan la telemetría con los contadores de resiliencia para evitar falsos
  positivos en screenings cooperativos.
- Refuerzo de CI para escenarios multi-proveedor, ejecutando la suite de integración completa y
  asegurando que los pipelines configuren el backend de snapshots en modo temporal (`Null`/`tmp_path`).

### Documentation
- README, guía de pruebas y troubleshooting alineados con la versión 0.3.29.1, con comandos de
  exportación que detallan parámetros `--input`, artefactos generados (CSV, ZIP y Excel) y los pasos
  para forzar escenarios multi-proveedor en CI.
- Documentación de las nuevas validaciones Markowitz y de la configuración recomendada para el
  backend de snapshots en pipelines efímeros.

### Tests
- Recordatorios en CI para ejecutar `pytest tests/integration/` completo y validar degradaciones
  multi-proveedor antes de publicar artefactos.

## [0.3.29] - 2025-11-20

### Changed
- Sincronización del versionado 0.3.29 entre `pyproject.toml`, `shared.version` y las superficies
  visibles para mantener la trazabilidad durante el hardening de CI.

### Documentation
- README, guías de pruebas y troubleshooting alineados con la numeración 0.3.29 y con ejemplos de
  exportación actualizados (`--input`, `--formats`, directorios de salida) que reflejan el
  comportamiento real de `scripts/export_analysis.py`.

### Tests
- Recordatorios de ejecución en CI y validaciones manuales actualizados para utilizar la versión
  0.3.29 al verificar banners y reportes exportados.

## [0.3.28.1] - 2025-11-18

### Changed
- Hardening de pipelines CI: sincronización de versionado entre `pyproject.toml`, `shared.version`
  y superficies visibles, más validaciones adicionales de telemetría para detectar desalineaciones
  en los contadores persistentes.

### Documentation
- README, guías de pruebas y troubleshooting actualizadas para reflejar la release 0.3.28.1 como
  parche de hardening/CI y mantener vigentes los flujos de snapshots, exportaciones y observabilidad.

### Tests
- Recordatorios de ejecución en CI ajustados para garantizar que las suites utilicen la numeración
  0.3.28.1 en banners, stubs y verificaciones de versionado.

## [0.3.28] - 2025-11-15

### Added
- Script `scripts/export_analysis.py` para generar exportaciones enriquecidas del screening con
  resúmenes agregados y notas de telemetría.
- Métricas de almacenamiento y contadores de snapshots visibles en el health sidebar para rastrear
  recuperaciones desde el almacenamiento persistente.

### Changed
- Persistencia de snapshots del portafolio y de los presets del sidebar para acelerar screenings
  consecutivos y dejar trazabilidad en la telemetría.

### Documentation
- README actualizado con la narrativa de la release (snapshots persistentes, exportaciones
  enriquecidas, observabilidad extendida) e instrucciones paso a paso para `scripts/export_analysis.py`.
- Guías de pruebas y troubleshooting extendidas con escenarios específicos para validar el nuevo
  almacenamiento y depurar métricas de observabilidad.

### Tests
- Nuevas recomendaciones de QA para ejecutar suites y escenarios manuales que ejercitan los contadores
  de snapshots y las rutas de fallback persistente.

## [0.3.27.1] - 2025-11-07

### Changed
- Persistencia del health sidebar reforzada para conservar la última secuencia de degradación y los
  contadores de resiliencia aun después de recargar la sesión, evitando inconsistencias entre la UI
  y la telemetría de backend.

### Documentation
- Se documentó la configuración de claves (Alpha Vantage, Polygon, FMP, FRED y World Bank) y los
  pasos para validar el fallback jerárquico desde el health sidebar, alineando README y guías de
  troubleshooting con la nueva release.

### Tests
- Se estabilizaron las suites que validan la degradación multinivel (`tests/test_version_display.py`
  y escenarios macro) con fixtures de claves deterministas para asegurar la cobertura de
  resiliencia en CI.

## [0.3.27] - 2025-11-05

### Added
- Monitor de resiliencia en el health sidebar que expone el último proveedor exitoso, la secuencia de
  degradación (`primario → secundario → fallback`) y las insignias de recuperación asociadas.

### Changed
- Centralización de timeouts, backoff y códigos de error para los clientes de APIs macro y de
  portafolio, asegurando que los fallback registrados en telemetría conserven la procedencia y la
  latencia de cada intento.
- Notificaciones internas (`st.toast`) actualizadas para informar cuando un proveedor externo vuelve
  a estar disponible tras un incidente, manteniendo trazabilidad directamente en la UI.

### Documentation
- README y guías alineadas con la release 0.3.27: quick-start renovado, escenarios de resiliencia
  multi-API, fecha de publicación y recordatorios para verificar la versión visible en header/footer.

## [0.3.26.1] - 2025-10-26

### Added
- Notificaciones internas en la UI basadas en `st.toast` para confirmar refrescos y cierres de sesión,
  consolidando feedback inmediato para los analistas que operan desde el dashboard.

### Changed
- Sincronización del número de versión 0.3.26.1 entre `pyproject.toml`, `shared.version` y las superficies
  visibles (header, footer, sidebar y tests) para mantener la trazabilidad de la release.

### Documentation
- README, guías y quick-start alineados con la release 0.3.26.1, detallando el flujo de notificaciones
  internas y los pasos para validar la numeración visible.

## [0.3.26] - 2025-10-19

### Changed
- El login y el dashboard principal reutilizan el helper `shared.version` para mostrar "Versión 0.3.26"
  con la hora actualizada por `TimeProvider`, garantizando que el encabezado y el footer compartan
  la misma metadata visible.
- El health sidebar consolida la cronología de screenings con badges de cache hit/miss y métricas de
  fallback sincronizadas con los contadores globales, evitando discrepancias entre la vista tabular y
  los totales expuestos en la parte superior del panel.

### Fixed
- `ui.ui_settings.apply_settings` ahora verifica la disponibilidad de `st.set_page_config` antes de
  invocarlo, permitiendo ejecutar suites locales con stubs de Streamlit que no exponen ese método.
- `app.py` define stubs de compatibilidad (`st.stop`, `st.container`, `st.columns`) cuando la API de
  Streamlit no los ofrece, destrabando los tests que importan la app en entornos fuera de Streamlit.

### Documentation
- README actualizado con el quick-start de la release 0.3.26, incluyendo instrucciones para verificar
  la versión visible en header/footer y resúmenes renovados de telemetría.

### Tests
- Las suites `tests/test_version_display.py` y `tests/test_version_sync.py` se mantienen alineadas con
  la numeración 0.3.26 para validar el helper de versión y la visibilidad en la UI.

## [0.3.25.1] - 2025-10-03

### Fixed
- Se corrigió la función `drawdown_series` en `application/risk_service.py` para manejar correctamente series vacías y calcular drawdowns acumulados, eliminando el `IndentationError` que impedía iniciar la aplicación.
- Se corrigió un `IndentationError` en `application/risk_service.py` causado por un bloque `if` sin cuerpo en la función `drawdown_series`.
- La función ahora retorna un `pd.Series` vacío cuando no hay datos de entrada, previniendo bloqueos en inicialización y permitiendo flujos consistentes en métricas de riesgo.
- La app vuelve a iniciar correctamente tras el reboot con la release 0.3.25.

### Tests
- Cobertura extendida para validar el manejo de series vacías en `drawdown_series`.

## [0.3.24.2] - 2025-10-10

### Fixed
- Se corrigió el `NameError` en `render_portfolio_section` al eliminar la referencia
  obsoleta a `apply_filters` y delegar la construcción del view-model al servicio
  cacheado de portafolio.
- `record_macro_api_usage` vuelve a registrar la última ejecución macro sin depender
  de variables temporales inexistentes, evitando el `NameError latest_entry` y
  propagando correctamente las métricas hacia el sidebar de salud.

### Changed
- `build_portfolio_viewmodel` ahora recibe un `PortfolioViewSnapshot` en lugar de
  ejecutar filtros manualmente, alineando la nueva capa de cache con los
  controladores.

### Tests
- Se actualizaron las suites de portafolio para simular el servicio de view-model
  cacheado y validar el flujo completo tras el refactor.
- Los tests de métricas de salud se adaptaron al nuevo contrato de macro
  (intentos normalizados + entrada más reciente) para cubrir el fix.

## [0.3.24.1] - 2025-10-09

### Tests
- La suite de CI recuperó su estabilidad tras ajustar los timeouts intermitentes y sincronizar los entornos de ejecución.

### Changed
- Los mocks de proveedores externos fueron alineados con los contratos vigentes para evitar desfasajes durante las pruebas integradas.

### Fixed
- La persistencia de favoritos ahora conserva los emisores marcados entre sesiones, incluso al alternar entre vistas y filtros derivados.

### Documentation
- Guías actualizadas describiendo la estabilidad recuperada, los mocks vigentes y el flujo persistente de favoritos para el release 0.3.24.1.

## [0.3.24] - 2025-10-08

### Changed
- Refactor del módulo de portafolio para simplificar dependencias internas y facilitar futuras extensiones en la UI y los controladores.

### Fixed
- Ajustes en los cacheos del screener para estabilizar invalidaciones y preservar resultados consistentes entre ejecuciones consecutivas.

### Added
- Gestión de favoritos en el portafolio que habilita marcar emisores clave y priorizarlos en los listados derivados.

### Documentation
- Plan de documentación para describir el refactor del portafolio, los escenarios de cacheo y el uso de favoritos en la próxima iteración.

## [0.3.23] - 2025-10-07
### Added
- Cliente dedicado para FRED con autenticación, gestión de rate limiting y normalización de observaciones para enriquecer el screener de oportunidades con contexto macro/sectorial. ([`infrastructure/macro/fred_client.py`](infrastructure/macro/fred_client.py))
- Métrica de salud que expone el estado de la nueva dependencia externa (`macro_api`), ampliando la observabilidad del sistema. ([`services/health.py`](services/health.py))
### Changed
### Documentation
- README actualizado con los pasos para habilitar la integración macro, variables de entorno requeridas y consideraciones de failover. ([`README.md`](README.md#datos-macro-y-sectoriales-fred--fallback))
### Tests

## [0.3.22] - 2025-10-06
### Changed
- Sincronización del número de versión `0.3.22` entre `pyproject.toml`, el helper `shared.version`
  y las superficies visibles para mantener el encabezado de pestañas y el sidebar actualizados.
### Documentation
- Quick-start y menús documentados mencionando explícitamente la release 0.3.22 y reforzando el
  recordatorio de versión visible en la UI.

## [0.3.21] - 2025-10-05
### Changed
- Refinamiento UX del mini-dashboard del healthcheck para resaltar los tiempos cacheados vs. recientes con etiquetas de estado
  claras y tooltips que explican la metodología de medición. ([`ui/health_sidebar.py`](ui/health_sidebar.py))
### Added
- Telemetría histórica del screener que persiste los tiempos de ejecución previos y permite graficar tendencias directamente
### Documentation
- Se incorporó documentación multimedia (capturas y clips) que guía la interpretación del mini-dashboard y la navegación por la
  nueva telemetría histórica. ([`README.md`](README.md#caché-del-screener-de-oportunidades))

## [0.3.20] - 2025-10-04
### Added
- Mini-dashboard en el healthcheck que expone la duración previa y cacheada de los screenings de oportunidades, permitiendo
  [`ui/health_sidebar.py`](ui/health_sidebar.py))
### Changed
- Telemetría extendida para registrar aciertos de caché y variaciones de filtros del screener, dejando trazabilidad directa en el
  panel de salud. ([`services/health.py`](services/health.py), [`ui/health_sidebar.py`](ui/health_sidebar.py))
### Tests
### Documentation
- Limpieza de referencias legacy y actualización de la estrategia de cacheo documentada para reflejar el nuevo dashboard y la
  telemetría extendida. ([`README.md`](README.md#caché-del-screener-de-oportunidades))

## [0.3.19] - 2025-10-03
### Added
- Presets personalizados en la UI del screener que permiten guardar y reutilizar combinaciones propias de filtros sin depender de configuraciones globales.
### Changed
- Flujo de comparación enriquecido para revisar lado a lado los resultados de presets activos, destacando las diferencias en filtros y métricas clave antes de confirmar los cambios.
### Fixed
- Cacheo de respuestas de Yahoo Finance homogeneizado entre backend y stub, evitando expiraciones adelantadas y asegurando consistencia en los resultados servidos a la UI.
### Documentation
- Limpieza de referencias legacy en las guías internas, documentando el nuevo flujo de presets personalizados y eliminando instrucciones obsoletas.

## [0.3.18] - 2025-10-02
### Added
- Los listados de oportunidades ahora incluyen enlaces clickeables hacia Yahoo Finance, permitiendo abrir la ficha del ticker directamente desde la UI o los reportes exportados.
### Changed
- Se unificó la tabla visible y el CSV descargable para compartir columnas, orden y formato de los enlaces, preservando la paridad entre ambas superficies.
### Fixed
- Se eliminaron las advertencias duplicadas que aparecían al regenerar el listado cuando coexistían datos de Yahoo y del stub.
### Documentation
- Se actualizaron las guías internas para describir los enlaces hacia Yahoo Finance y los criterios de sincronización entre la UI y el CSV exportable.

## [0.3.17] - 2025-10-01
### Added
- La estrategia Andy fue promovida a release estable tras validar los filtros financieros activos, el score normalizado y la telemetría espejo entre Yahoo y el stub, dejando documentada la cobertura manual que respalda el corte.
### Changed
- La UI y el backend leen la versión desde `pyproject.toml` mediante `shared.version.__version__`, evitando desfasajes entre las superficies y simplificando la sincronización de releases. ([`shared/version.py`](shared/version.py), [`ui/footer.py`](ui/footer.py))
### Documentation
- Se incorporó una guía de interpretación para la telemetría del barrido, con ejemplos de severidades y métricas monitoreadas tanto en el stub como en Yahoo. ([`README.md`](README.md#telemetría-del-barrido))
- README documenta la estrategia Andy lista para producción, enumerando:
  - los filtros financieros activos que se aplican en la tabla de oportunidades;
  - la normalización del `score_compuesto` en escala 0-100;
  - la telemetría compartida entre Yahoo Finance y el stub determinista, junto con los casos de failover;
  - la columna `Yahoo Finance Link`, ejemplificando cómo se pobla con universos live y con el stub.
- La guía de QA aclara que los 37 tickers del stub y los universos dinámicos comparten el mismo formato de enlace hacia Yahoo Finance para mantener paridad en las verificaciones.

## [0.3.16] - 2025-09-30
### Added
- Se amplió el stub de fundamentals para cubrir emisores adicionales y acompañar las nuevas validaciones del flujo beta.
- Prueba de integración que combina la selección de presets con el fallback al stub para validar el pipeline UI → controlador → screener bajo filtros reforzados.
### Changed
- Se endurecieron los filtros de fundamentals en la UI para reflejar los criterios reforzados del backend y mantener consistencia entre fuentes.
### Documentation
- README actualizado con la tabla completa del universo determinista de 19 emisores, explicando cómo el fallback replica la estrategia Andy durante los failovers.

## [0.3.15] - 2025-09-30
### Fixed
- El healthcheck del sidebar reutiliza `shared.ui.notes.format_note` para unificar la iconografía y el énfasis de los mensajes con el resto de la UI, evitando divergencias en la presentación de severidades. ([ui/health_sidebar.py](ui/health_sidebar.py))
### Tests
- Documentado el procedimiento para habilitar `pytest -m live_yahoo` mediante la variable `RUN_LIVE_YF` y advertir sobre su naturaleza no determinista. ([README.md](README.md#pruebas))
### Documentation
- Documentadas las severidades soportadas por `shared.ui.notes.format_note`, sus prefijos (⚠️/ℹ️/✅/❌) y el helper compartido para mantener mensajes consistentes en la UI. ([README.md](README.md#notas-del-listado-y-severidades), [tests/shared/test_notes.py](tests/shared/test_notes.py))

## [3.0.1]
### Changed
- El `score_compuesto` ahora se normaliza en escala 0-100 y se filtra automáticamente usando el umbral configurable `MIN_SCORE_THRESHOLD` (80 por defecto) para reducir ruido en los resultados de la pestaña beta.
- El listado final de oportunidades respeta el límite configurable `MAX_RESULTS` (20 por defecto), manteniendo la tabla acotada incluso cuando Yahoo Finance devuelve universos extensos.

### UI
- La cabecera de "Empresas con oportunidad" indica cuándo se aplican el umbral mínimo y el recorte del top N, explicando al usuario por qué ciertos tickers quedan fuera del informe.

## [0.3.14]
### Added
- Nuevos filtros en el screener: `min_eps_growth`, `min_buyback`, selector de sectores y un toggle para indicadores técnicos, que permiten ajustar dinámicamente la priorización de emisores.
- Caption de fuente visible en la UI de oportunidades para dejar claro el origen de los datos mostrados.

## [0.3.13] - 2025-09-30
### Changed
- La leyenda en la pestaña beta ahora destaca dinámicamente si los datos provienen de Yahoo Finance o del stub local, evitando confusiones durante los failovers.
- Se diferencian explícitamente las captions de Yahoo y del stub para que cada flujo muestre su fuente en el encabezado correspondiente.

### Tests
- Se actualizaron las pruebas de UI para validar la nueva diferenciación de captions entre Yahoo y el stub.

## [0.3.12] - 2025-09-29
### Fixed
- Se repararon las pruebas de `shared.settings` para que consuman los TTL y alias directamente desde la configuración compartida.
### Tests
- La suite de CI recuperó su estabilidad al eliminar los falsos negativos que provocaba la discrepancia en los tests de configuración.

## [0.3.11] - 2025-10-01
### Fixed
- Se repararon los tests de la pestaña beta para alinear las expectativas con el flujo visible en la UI.
### Changed
- Toda visualización de versión ahora se alimenta dinámicamente desde `pyproject.toml`, evitando desfasajes entre backend y UI.
- Se maneja explícitamente el feature flag de la pestaña beta para controlar su activación sin efectos secundarios.

## [0.3.10] - 2025-09-30
### Fixed
- Se corrigió el `ImportError` que se disparaba al inicializar los módulos de Yahoo Finance en entornos sin dependencias opcionales.
### Changed
- Los TTL por defecto de Yahoo Finance ahora se aplican automáticamente cuando no hay configuración explícita, permitiendo reutilizar cachés sin sobrecostos manuales.

## [0.3.9] - 2025-09-29
### Changed
- Los filtros de payout ratio, racha de dividendos y CAGR mínima ahora se aplican
  también en el screener de Yahoo para mantener una experiencia consistente con
  el stub local.
- Refactorización de `_apply_filters_and_finalize` para compartir la lógica de
  filtrado entre la integración de Yahoo Finance y el stub de respaldo.
### Tests
- Refuerzo de pruebas que cubren el filtrado compartido y la alineación de
  resultados entre ambas fuentes de datos.

## [0.3.8] - 2025-09-29
### Added
- Integración con Yahoo Finance para descargar históricos, indicadores técnicos y
  métricas fundamentales visibles en la pestaña de portafolio.
- Nuevos paneles con métricas fundamentales y ranking ESG del portafolio basados
  en los datos enriquecidos de Yahoo Finance.
### Changed
- Caché configurable para las consultas de Yahoo Finance mediante los TTL
  `CACHE_TTL_YF_*`, documentados en la configuración.
### Fixed
- Fallback automático al stub `infrastructure/cache/ta_fallback.csv` cuando la
  API de Yahoo Finance devuelve errores, con trazabilidad en el healthcheck.

## [0.3.7] - 2025-09-28
### Added
- Se agregó la pestaña beta de "Empresas con oportunidad" junto con su stub inicial para explorar la integración futura.

## [0.3.6] - 2025-09-17
### Removed
- Se eliminó la referencia obsoleta a `TimeProvider.now().moment` para evitar invocaciones inexistentes.
### Fixed
- Se corrigió el uso de `bearer_time` asegurando que utilice la clave actualizada.

## [0.3.5] - 2025-09-17
### Fixed
- Se configuró `fileWatcherType = "poll"` en Streamlit para evitar bloqueos del recargador
  en entornos con sistemas de archivos basados en red.

## [0.3.4] - 2025-09-17
### Fixed
- Se corrigió la incompatibilidad aware/naive al comparar las marcas de tiempo.
- Se actualizó `bearer_time` a naive en el cliente legacy para alinear el formato de fechas.

## [0.3.3] - 2025-09-21
### Fixed
- Se corrigió `shared.time_provider.TimeProvider` para garantizar que los timestamps y objetos `datetime`
  generados compartan la misma zona horaria y formato.
### Changed
- Se unificó la API de `TimeProvider` documentando explícitamente `now()` y `now_datetime()` para
  elegir entre cadena formateada u objeto `datetime` según la necesidad.

## [0.3.2] - 2025-09-20
### Changed
- Se unificó el manejo de hora mediante `shared.time_provider.TimeProvider` para mantener
  timestamps consistentes en formato `YYYY-MM-DD HH:MM:SS` (UTC-3).

## [0.3.1] - 2025-09-19
### Changed
- El healthcheck del sidebar ahora expone la versión actual de la aplicación y se movió al final para concentrar en un único bloque el estado de los servicios monitoreados.

## [0.3.0] - 2025-09-18
### Added
- El bloque de seguridad del login ahora muestra dinámicamente la versión actual de la aplicación.

## [0.2.1] - 2025-09-17
### Added
- Se incorporó un timestamp argentino en el footer para reflejar la hora local
  de manera consistente.

## [0.2.0] - 2025-09-16
### Added
- Centralized cache TTL configuration in `shared/settings` and documented the
  new environment keys for quote and FX caches.
- Added a sidebar health-check indicator so operators can quickly confirm
  downstream service availability from the navigation.
- Jerarquía de errores compartida (PR1) para estandarizar cómo controllers y
  services reportan incidencias recuperables.
### Changed
- Refactored the Streamlit session and routing logic to reuse authentication and
  page-loading flows between the main application and auxiliary entry points.
### Fixed
- Successful login now marks the session as authenticated to access the main page.
- Fixed: los paneles ahora se recargan automáticamente después de logout/login sin requerir refresco manual.
- Se corrigieron los tests de logout para reflejar la nueva firma y el comportamiento de la función.
- Se corrigieron pruebas fallidas en ta_service, portfolio_controller y
  portfolio_service_utils para alinear expectativas de tests con la
  implementación real.
- Deployment stable on Streamlit Cloud.

### Security
- Removed passwords from `session_state`; authentication now relies solely on local variables and tokens.

### Removed
- Removed deprecated `use_container_width` parameter (Streamlit ≥ 1.30).

### Tests
- Nuevas pruebas de TTL, health sidebar y propagación de errores (PR2).

## [2025-09-13]
### Tests
- Se agregaron pruebas de cobertura para UI, controllers, servicios, application, infrastructure y shared.

## v0.6.3-part1 — Created predictive_engine package and migrated predictive/adaptive core logic.

