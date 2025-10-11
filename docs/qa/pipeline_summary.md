# Resumen del pipeline QA

## Validaciones automáticas
- Linting: `flake8`
- Type-checking: `mypy`
- Tests: `pytest --cov --cov-report=term-missing`
- Seguridad: `bandit -r application controllers services shared` y `pip-audit`

## Tiempos de referencia
- Linting: ~20 segundos
- Type-checking: ~35 segundos
- Tests: ~2 minutos (dependiendo de APIs externas)
- Seguridad: ~45 segundos

## Dependencias clave
- Python 3.11+
- `requirements.txt` para dependencias de runtime
- `requirements-dev.txt` para herramientas de desarrollo
- `nox` para orquestar las sesiones
