# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Security
- Improved authentication flow and strengthened token handling.
- Removed password from in-memory session state; authentication relies solely on local variables and tokens.
### Fixed
- Successful login now marks the session as authenticated to access the main page.
### Tests
- Added tests verifying login reruns for valid, invalid, and expired sessions.
