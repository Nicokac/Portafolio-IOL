# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

### Documentation

### Tests

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
- El controlador de oportunidades combina la información sectorial proveniente de FRED (o del fallback configurado) con los resultados del screening, agregando la columna `macro_outlook` y notas contextuales. ([`controllers/opportunities.py`](controllers/opportunities.py))
### Documentation
- README actualizado con los pasos para habilitar la integración macro, variables de entorno requeridas y consideraciones de failover. ([`README.md`](README.md#datos-macro-y-sectoriales-fred--fallback))
### Tests
- Cobertura específica para los flujos de fallback del controlador frente a la dependencia macro, asegurando la continuidad del screener. ([`controllers/test/test_opportunities_macro.py`](controllers/test/test_opportunities_macro.py))

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
  desde el panel de salud. ([`services/health.py`](services/health.py), [`controllers/opportunities.py`](controllers/opportunities.py))
### Documentation
- Se incorporó documentación multimedia (capturas y clips) que guía la interpretación del mini-dashboard y la navegación por la
  nueva telemetría histórica. ([`README.md`](README.md#caché-del-screener-de-oportunidades))

## [0.3.20] - 2025-10-04
### Added
- Mini-dashboard en el healthcheck que expone la duración previa y cacheada de los screenings de oportunidades, permitiendo
  comparar tiempos desde la UI. ([`controllers/opportunities.py`](controllers/opportunities.py), [`services/health.py`](services/health.py),
  [`ui/health_sidebar.py`](ui/health_sidebar.py))
### Changed
- Telemetría extendida para registrar aciertos de caché y variaciones de filtros del screener, dejando trazabilidad directa en el
  panel de salud. ([`services/health.py`](services/health.py), [`ui/health_sidebar.py`](ui/health_sidebar.py))
### Tests
- Casos que validan *cache hits* e invalidaciones al cambiar filtros del screener de oportunidades. ([`tests/controllers/test_opportunities_controller.py`](tests/controllers/test_opportunities_controller.py))
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
- El stub de oportunidades ahora genera notas de telemetría con severidades `ℹ️/⚠️` según el tiempo de ejecución y deja trazabilidad de los descartes aplicados para facilitar la observabilidad durante los failovers. ([`application/screener/opportunities.py`](application/screener/opportunities.py))
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
- Universo automático de oportunidades generado con `list_symbols_by_markets` y la configuración `OPPORTUNITIES_TARGET_MARKETS` para alinear los emisores con los mercados habilitados en cada sesión.
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

