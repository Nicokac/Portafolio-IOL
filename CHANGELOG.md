# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Fixed
- Successful login now marks the session as authenticated to access the main page.
- Fixed: los paneles ahora se recargan automáticamente después de logout/login sin requerir refresco manual.
- Se corrigieron los tests de logout para reflejar la nueva firma y el comportamiento de la función.

### Security
- Removed passwords from `session_state`; authentication now relies solely on local variables and tokens.

### Tests
- Added tests verifying login reruns for valid, invalid, and expired sessions.

