# Guía operativa para Portafolio IOL

## Índice
- [1. Visión general breve](#1-visión-general-breve)
- [2. Requisitos del sistema](#2-requisitos-del-sistema)
- [3. Guía rápida](#3-guía-rápida)
- [4. Estructura del repo](#4-estructura-del-repo)
- [5. Política para agentes (Codex Cloud)](#5-política-para-agentes-codex-cloud)
- [6. Convenciones de ramas y PRs (con checklist)](#6-convenciones-de-ramas-y-prs-con-checklist)
- [7. Makefile/Pre-commit sugeridos](#7-makefilepre-commit-sugeridos)
- [8. Comandos de desarrollo](#8-comandos-de-desarrollo)
- [9. Variables de entorno](#9-variables-de-entorno)
- [10. CI/CD](#10-cicd)
- [11. Troubleshooting](#11-troubleshooting)
- [12. Apéndices](#12-apéndices)
- [13. Fuentes consultadas](#13-fuentes-consultadas)
- [14. Epígrafe de validación](#14-epígrafe-de-validación)

## 1. Visión general breve
- **Proyecto:** Portafolio IOL v0.7.0. Aplicación Streamlit para analizar carteras de inversión apoyada por backend FastAPI y un motor predictivo desacoplado.
- **Stack principal:** Python 3.10, Streamlit, FastAPI/Uvicorn, pandas, numpy, scipy, statsmodels, Plotly y servicios auxiliares de caché/telemetría.
- **Componentes clave:** UI (`app.py`, `ui/`), API (`api/main.py`, `run_api.sh`), motor adaptativo (`predictive_engine/`), servicios compartidos (`services/`, `shared/`), bootstrap común (`bootstrap/`).

## 2. Requisitos del sistema
- **Python:** 3.10 (imagen base `python:3.10-slim`).
- **Instalación mínima:** `pip install -r requirements.txt`; para QA agregar `-r requirements-dev.txt`.
- **Herramientas opcionales:** `nox`, `bandit`, `pip-audit`, `pytest-cov`, `ruff`, `mypy`.
- **Contenedores/Procfile:** `scripts/start.sh` establece `UV_USE_REQUIREMENTS=1` y `PYTHONDONTWRITEBYTECODE=1` antes de iniciar Streamlit.

## 3. Guía rápida

# AGENTS.md

## Setup
- Instalar deps: `pip install -r requirements.txt`
- Tests: `pytest -q`
- Lint: `ruff check .`
- Types: `mypy --ignore-missing-imports .`

## Convenciones
- Formato: `ruff format`
- Pre-commit pasa en CI; arregla antes de commitear.

## Cómo validar cambios
1) `pytest -q`
2) `ruff check .`
3) `mypy ...`
4) Si falla, proponer el **cambio mínimo** para que (1)-(3) pasen.

## PRs
- Título: `[módulo] corto y claro`
- Mensaje: por qué, cómo, pruebas

## 4. Estructura del repo
- `bootstrap/`: Inicialización compartida para UI/API, con perfiles y precarga de servicios.
- `application/`: Servicios de dominio (recomendaciones, riesgo, perfiles) y lógica de negocio.
- `controllers/`: Orquestadores UI/API que coordinan servicios y state de Streamlit.
- `domain/`: Modelos y entidades de soporte para la lógica de negocio.
- `services/`: Adaptadores (caché, telemetría, notificaciones, mantenimiento) reutilizables.
- `shared/`: Utilidades comunes (configuración, seguridad, logging, proveedores de tiempo).
- `ui/`: Componentes Streamlit, tabs, paneles de monitoreo, tablas y helpers.
- `predictive_engine/`: Motor de pronósticos con adapters, almacenamiento y cálculos estadísticos.
- `api/`: Aplicación FastAPI con routers, esquemas y configuración de Uvicorn.
- `scripts/`: Utilitarios CLI (`start.sh`, `export_analysis.py`, `generate_mock_data.py`, etc.).
- `tests/`: Suite Pytest con stubs de Streamlit y casos por capa.
- `docs/`: Guías de desarrollo, pruebas, seguridad, operaciones y reportes QA.

## 5. Política para agentes (Codex Cloud)
- Prioriza el **cambio mínimo**: modifica solo lo necesario para resolver el issue o mejorar la documentación/process.
- Respeta los estilos existentes; evita reordenar imports o aplicar formateos masivos si no se justifica.
- Limita el alcance de cada PR a un objetivo acotado y documenta las rutas modificadas.
- Ejecuta las validaciones principales (`pytest`, `ruff`, `mypy`) siempre que sea factible y registra el resultado.
- Sincroniza la documentación relevante cuando cambies flujos (por ejemplo, `docs/dev_guide.md`, `docs/testing.md`).
- Mantén los commits descriptivos en español o inglés, indicando claramente la intención.
- No agregues secretos en el repositorio; utiliza `.env` y validadores (`python -m shared.security_env_validator`).

## 6. Convenciones de ramas y PRs (con checklist)
- **Ramas:** TODO: confirmar convención de nombres (no se encontró política explícita).
- **PRs:** Usa el formato de título `[módulo] corto y claro`. Describe **por qué**, **cómo** y **pruebas** en el cuerpo.
- **Checklist sugerido antes de abrir PR:**
  - [ ] Confirmar que la rama está actualizada respecto a `main` (o rama base) — TODO: confirmar rama principal.
  - [ ] Ejecutar `pytest -q` o documentar la causa de fallo.
  - [ ] Ejecutar `ruff check .` o justificar excepciones.
  - [ ] Ejecutar `mypy --ignore-missing-imports .` cuando aplique.
  - [ ] Actualizar documentación afectada.
  - [ ] Adjuntar resultados de QA en la descripción del PR.
  - [ ] Verificar que no se incluyan archivos generados o secretos.

## 7. Makefile/Pre-commit sugeridos
- No se encontró `Makefile` ni configuración `pre-commit`. TODO: confirmar si existen en ramas privadas.
- **Sugerencia de targets útiles:**
  - `make install`: `pip install -r requirements.txt -r requirements-dev.txt`.
  - `make test`: `pytest -q` (permitir override de `PYTEST_ADDOPTS`).
  - `make lint`: `ruff check .`.
  - `make typecheck`: `mypy --ignore-missing-imports .`.
  - `make format`: `ruff format`.
- **Hook pre-commit recomendado:** ejecutar `ruff check --fix`, `ruff format` y `mypy --ignore-missing-imports .` sobre cambios staged para detectar issues tempranos.

## 8. Comandos de desarrollo
- **Instalación local completa:** `pip install -r requirements.txt -r requirements-dev.txt`.
- **UI Streamlit:** `streamlit run app.py --server.address=0.0.0.0 --server.port=8501` o `bash scripts/start.sh`.
- **Backend FastAPI:** `./run_api.sh` levanta Uvicorn con recarga.
- **QA vía nox:** `nox -s lint`, `nox -s typecheck`, `nox -s tests`, `nox -s security`.
- **Generación de mocks:** `python scripts/generate_mock_data.py --output docs/fixtures/default`.
- **Exportaciones:** `python scripts/export_analysis.py --input <ruta> --output <destino>`.

## 9. Variables de entorno
- Variables base (`.env.example`): `USER_AGENT`, `IOL_USERNAME`, `IOL_PASSWORD`, `CACHE_TTL_*`, `YAHOO_*_TTL`, `QUOTES_HIST_MAXLEN`, `MAX_QUOTE_WORKERS`.
- Seguridad y cifrado: `IOL_TOKENS_KEY` es obligatorio para Fernet, `FASTAPI_TOKENS_KEY` requerido por validadores.
- Configuración adicional usada en scripts/CI: `UV_USE_REQUIREMENTS`, `PYTHONDONTWRITEBYTECODE`, `RUN_LIVE_YF`, `CACHE_SMOKE_REPORT`, `PORT`.
- Documenta en el PR cualquier variable nueva o cambio de comportamiento.

## 10. CI/CD
- Workflow principal `ci.yml`: ejecuta pytest con cobertura, smoke de caché y (cuando aplica) smoke Yahoo Finance; publica artefactos.
- Workflow `dependency-update.yml`: corre `scripts/update_dependencies.sh` programado o manual.
- Workflow `validate_secrets.yml`: valida claves Fernet con `python -m shared.security_env_validator`.
- TODO: confirmar si hay despliegues automatizados fuera de los workflows listados.

## 11. Troubleshooting
- **Pytest falla por `--cov` obligatorio:** la opción proviene de `pyproject.toml`; usa `pytest --override-ini addopts=''` para aislar casos.
- **Streamlit en modo headless:** los tests muestran advertencias `missing ScriptRunContext` y Kaleido deshabilitado; son esperadas.
- **Ruff reporta >400 issues:** prioriza módulos críticos o utiliza `--select` para abordar familias específicas.
- **Mypy:** corrige comentarios `type: ignore` inválidos en `services/cache/market_data_cache.py` y `services/maintenance/sqlite_maintenance.py` antes de habilitar CI estricto.
- **Warmup científico:** mantener `ENABLE_BYTECODE_WARMUP` activo en despliegues para reducir latencia inicial — TODO: confirmar pipeline actual.

## 12. Apéndices
- **Despliegue:** Dockerfile multi-stage basado en `python:3.10-slim`; `scripts/start.sh` sirve como entrypoint.
- **Versionado:** sincronizar `pyproject.toml`, `README.md` y `CHANGELOG.md` en cada release.
- **QA manual:** `docs/testing.md` y `docs/qa/` centralizan reportes y procedimientos offline.
- **Convenciones adicionales:** revisar `docs/dev_guide.md` para flujo offline `_render_for_test()` y buenas prácticas de commits.

## 13. Fuentes consultadas
- README.md
- pyproject.toml
- docs/dev_guide.md
- docs/testing.md
- .env.example
- scripts/start.sh
- scripts/README.md
- scripts/generate_mock_data.py
- scripts/export_analysis.py
- run_api.sh
- app.py
- api/main.py
- noxfile.py
- .github/workflows/ci.yml
- .github/workflows/dependency-update.yml
- .github/workflows/validate_secrets.yml

## 14. Epígrafe de validación (2025-10-21T16:32:59+00:00)
- `pytest -q`: falla inmediatamente porque `pyproject.toml` fuerza `--cov=application --cov=controllers --cov=services` y Pytest no reconoce la opción en este entorno.
- `ruff check .`: detecta 437 errores (E402, F401, F811, F821, F841) distribuidos principalmente en tests y módulos UI.
- `mypy --ignore-missing-imports .`: reporta 3 errores (comentarios `type: ignore` inválidos y sangría inesperada en tests).
