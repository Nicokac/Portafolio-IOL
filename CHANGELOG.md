# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Centralized cache TTL configuration in `shared/settings` and documented the
  new environment keys for quote and FX caches.
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

## [2025-09-13]
### Tests
- Se agregaron pruebas de cobertura para UI, controllers, servicios, application, infrastructure y shared.

