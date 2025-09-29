# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

