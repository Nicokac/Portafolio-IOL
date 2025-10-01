# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Caché en `generate_opportunities_report` reutilizando screenings por combinación de filtros y registrando métricas de duración previa vs. cacheada en el healthcheck. ([`controllers/opportunities.py`](controllers/opportunities.py), [`services/health.py`](services/health.py), [`ui/health_sidebar.py`](ui/health_sidebar.py))
### Tests
- Casos que validan *cache hits* e invalidaciones al cambiar filtros del screener de oportunidades. ([`tests/controllers/test_opportunities_controller.py`](tests/controllers/test_opportunities_controller.py))
### Documentation
- README actualizado con la estrategia de cacheo y las métricas expuestas en el sidebar. ([`README.md`](README.md#caché-del-screening-de-oportunidades))

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

