# Guía operativa para Portafolio IOL

## Índice
- [1. Visión general breve](#1-visión-general-breve)
- [2. Requisitos del sistema](#2-requisitos-del-sistema)
- [3. Guía rápida](#3-guía-rápida)
- [4. Estructura del repo](#4-estructura-del-repo)
- [5. Comandos de desarrollo](#5-comandos-de-desarrollo)
- [6. Configuración de agentes](#6-configuración-de-agentes)
- [7. Variables de entorno](#7-variables-de-entorno)
- [8. CI/CD](#8-cicd)
- [9. Troubleshooting](#9-troubleshooting)
- [10. Apéndices](#10-apéndices)

## 1. Visión general breve
- **Proyecto:** Portafolio IOL v0.7.0, aplicación para analizar carteras de inversión con UI Streamlit, backend FastAPI y motor predictivo reutilizable.【F:README.md†L1-L77】【F:api/main.py†L1-L61】
- **Stack principal:** Streamlit ≥1.49.1, FastAPI ≥0.115.0, Uvicorn, pandas ≥2.3, numpy ≥2.3, statsmodels 0.14, scipy ≥1.16, Plotly 6.3, cache y telemetría propios.【F:pyproject.toml†L6-L45】

## 2. Requisitos del sistema
- **Python:** 3.10 (Dockerfile base `python:3.10-slim`; CI usa `python-version: 3.x`).【F:Dockerfile†L1-L21】【F:.github/workflows/ci.yml†L22-L35】
- **Pip:** se usa `pip install -r requirements.txt` (modo root en contenedores; sugerido virtualenv si se trabaja fuera del sandbox).【F:README.md†L86-L108】
- **Dependencias del sistema:** imagen slim instala `curl` para healthcheck; scripts en Procfile exportan `UV_USE_REQUIREMENTS` y `PYTHONDONTWRITEBYTECODE`.【F:Dockerfile†L13-L21】【F:Procfile†L1-L1】
- **Herramientas opcionales:** `nox`, `bandit`, `pip-audit`, `pytest-cov` cuando se ejecuta la QA extendida.【F:README.md†L86-L108】【F:noxfile.py†L33-L65】

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
- `application/`: servicios de dominio y lógica de negocio.【F:docs/dev_guide.md†L7-L30】
- `controllers/`: orquestación de UI y endpoints específicos.【F:docs/dev_guide.md†L7-L30】
- `services/`: adaptadores compartidos (caché, métricas, mantenimiento).【F:docs/dev_guide.md†L7-L30】【F:api/main.py†L1-L61】
- `ui/`: componentes Streamlit, tabs y paneles de diagnóstico.【F:docs/dev_guide.md†L7-L30】【F:README.md†L9-L77】
- `shared/`: helpers (configuración, logging, seguridad).【F:docs/dev_guide.md†L7-L30】
- `predictive_engine/`: motor predictivo desacoplado.【F:README.md†L49-L95】
- `api/`: backend FastAPI con routers y esquemas.【F:api/main.py†L1-L61】
- `tests/`: suite Pytest separada por capas con stubs de Streamlit.【F:docs/dev_guide.md†L12-L25】【F:docs/testing.md†L1-L64】
- `scripts/`: utilitarios (start, export, smoke tests, warmup).【F:scripts/README.md†L1-L20】【F:scripts/start.sh†L1-L14】
- Entry points:
  - UI Streamlit: `streamlit run app.py` (usa `bootstrap` y `ui.orchestrator`).【F:app.py†L1-L64】
  - API FastAPI: `uvicorn api.main:app` (wrapper `run_api.sh`).【F:api/main.py†L1-L61】【F:run_api.sh†L1-L3】
  - CLI utilitarios en `scripts/` (ej. `export_analysis.py`).【F:scripts/README.md†L1-L20】

## 5. Comandos de desarrollo
- **Instalación local:** `pip install -r requirements.txt -r requirements-dev.txt` para cubrir dependencias de QA (pytest, flake8).【F:docs/testing.md†L8-L29】
- **UI local:** `streamlit run app.py --server.address=0.0.0.0 --server.port=8501` (idéntico a `scripts/start.sh`).【F:scripts/start.sh†L1-L14】
- **Backend local:** `./run_api.sh` levanta Uvicorn con recarga automática en `http://localhost:8000`.【F:README.md†L109-L141】【F:run_api.sh†L1-L3】
- **QA con nox:**
  - `nox -s lint` (flake8), `nox -s typecheck` (mypy), `nox -s tests` (pytest con cobertura), `nox -s security` (bandit + pip-audit).【F:noxfile.py†L1-L65】
- **Scripts útiles:**
  - `python scripts/export_analysis.py --input <snapshots> --output <dest>` para reportes.【F:scripts/README.md†L1-L20】
  - `bash scripts/test_smoke_endpoints.sh <salida.json>` para smoke-test de endpoints cacheados (usado en CI).【F:.github/workflows/ci.yml†L47-L67】
  - `bash scripts/update_dependencies.sh` automatiza upgrades y PRs programados.【F:.github/workflows/dependency-update.yml†L1-L23】
- **Estado actual de validaciones (sandbox 2025-03-):**
  - `pytest -q` falla por `ImportError` circular en `tests/api/test_adaptive_utils.py` (`api.schemas.predictive`).【0d2ec6†L1-L104】
  - `ruff check .` reporta 437 errores (imports fuera de orden, redefiniciones, etc.).【0d7f94†L1-L118】
  - `mypy --ignore-missing-imports .` marca 3 errores por `type: ignore` inválido e indentación inesperada en tests.【c93084†L1-L4】
  - Documenta cualquier fix propuesto como cambio mínimo antes de actualizar esta sección.

## 6. Configuración de agentes
- Aplica siempre la regla de **cambio mínimo** y respeta estilos existentes (evitar reordenamientos masivos salvo que el lint lo exija).【F:docs/dev_guide.md†L79-L112】
- Trabaja únicamente dentro del workspace; evita tocar archivos fuera del repo.
- Flujo de cambios:
  1. Actualiza documentación relevante (`docs/dev_guide.md`, `docs/testing.md`, etc.) si cambias procesos.
  2. Ejecuta las validaciones (`pytest`, `ruff`, `mypy`) y anota resultados en el PR (incluye fallas conocidas si no se corrigen).
  3. Commits en español o inglés consistentes, describiendo la intención (ej. `docs: actualizar flujo offline`).【F:docs/dev_guide.md†L79-L112】
  4. PRs deben titularse `[módulo] corto y claro` y detallar **por qué**, **cómo** y **pruebas** (ver bloque obligatorio en sección 3).
- Si CI falla, reproducí localmente el job involucrado (tests, cache-smoke, security) y propone un fix puntual antes de ampliar el alcance.

## 7. Variables de entorno
- Desde `.env.example` (sin valores): `USER_AGENT`, `IOL_USERNAME`, `IOL_PASSWORD`, `CACHE_TTL_PORTFOLIO`, `CACHE_TTL_LAST_PRICE`, `CACHE_TTL_FX`, `CACHE_TTL_QUOTES`, `YAHOO_FUNDAMENTALS_TTL`, `YAHOO_QUOTES_TTL`, `QUOTES_HIST_MAXLEN`, `MAX_QUOTE_WORKERS`, `IOL_TOKENS_FILE`, `IOL_TOKENS_KEY`, `FX_AHORRO_MULTIPLIER`, `FX_TARJETA_MULTIPLIER`, `LOG_LEVEL`, `LOG_FORMAT`, `LOG_USER`, `PORTFOLIO_CONFIG_PATH`, `ASSET_CATALOG_PATH`. Úsalas para personalizar TTLs, logging y rutas.【F:.env.example†L1-L32】
- Seguridad / validadores: `FASTAPI_TOKENS_KEY`, `IOL_TOKENS_KEY`, `APP_ENV`, `APP_PRELOAD_LIBS` (precarga científica), `ENABLE_BYTECODE_WARMUP` (habilita `warmup_bytecode.py`).【F:docs/security_validation.md†L1-L42】【F:docs/operations.md†L1-L84】【F:scripts/start.sh†L1-L14】
- CI adicional: `RUN_LIVE_YF`, `LIVE_YAHOO_SMOKE_SCHEDULE_MODE`, `LIVE_YAHOO_SMOKE_ALLOWED_DAYS`, `LIVE_YAHOO_SMOKE_FORCE_SKIP`, `CACHE_SMOKE_REPORT`, `UV_USE_REQUIREMENTS`, `PYTHONDONTWRITEBYTECODE`, `PORT` (Procfile/Heroku).【F:.github/workflows/ci.yml†L1-L96】【F:Procfile†L1-L1】
- Documenta en el PR si agregas nuevas variables o modificás su uso.

## 8. CI/CD
- **Workflow `ci.yml`:** ejecuta pytest con cobertura, smoke test de caché y smoke opcional de Yahoo Finance; sube artefactos de cobertura y smoke report. Usa `RUN_LIVE_YF` cuando se habilita manualmente o vía cron.【F:.github/workflows/ci.yml†L1-L96】
- **Workflow `dependency-update.yml`:** corre mensualmente o manualmente `scripts/update_dependencies.sh` y abre PR automático si hay cambios.【F:.github/workflows/dependency-update.yml†L1-L23】
- **Workflow `validate_secrets.yml`:** verifica claves Fernet en cada push/PR usando `python -m shared.security_env_validator` con `APP_ENV=prod`.【F:.github/workflows/validate_secrets.yml†L1-L21】
- Sin evidencia de bots de auto-fix; cualquier formateo se maneja manualmente o vía `nox`. TODO: confirmar si existe pre-commit configurado en local.

## 9. Troubleshooting
- **Pytest ImportError (`api.schemas.predictive`):** la suite actual falla por inicialización circular al recolectar `tests/api/test_adaptive_utils.py`. Revisar dependencias en `api/routers/engine.py` y `api/schemas/predictive.py` antes de ejecutar CI.【0d2ec6†L1-L48】
- **Carga pesada de SciPy/statsmodels:** los tests que importan `application.benchmark_service` pueden tardar >1 min al compilar módulos de SciPy; mantener `PYTHONDONTWRITEBYTECODE=0` para reutilizar bytecode precalentado.【0d2ec6†L49-L88】【scripts/start.sh†L1-L14】
- **Ruff check masivo:** `ruff check .` reporta >400 issues (imports E402, redefiniciones, etc.); planifica refactors graduales o limita el alcance del lint con `ruff check path::module`.【0d7f94†L1-L118】
- **Mypy `type: ignore` inválido:** corrige anotaciones en `services/cache/market_data_cache.py` y `services/maintenance/sqlite_maintenance.py` antes de habilitar mypy en CI.【c93084†L1-L4】
- **Kaleido/Chromium warnings:** al no contar con Chromium en el entorno, Plotly emite advertencias y deshabilita exportación a imagen; ignorable para pruebas unitarias.【0d2ec6†L1-L27】
- **Streamlit warnings:** `missing ScriptRunContext` aparece en modo headless durante tests; es benigno según stub de Streamlit.【7248db†L1-L8】

## 10. Apéndices
- **Despliegue:** `Dockerfile` multi-stage instala deps y expone `scripts/start.sh`; `Procfile` y `scripts/start.sh` requieren `IOL_TOKENS_KEY` configurada antes de arrancar.【F:Dockerfile†L1-L21】【F:scripts/start.sh†L1-L14】【F:Procfile†L1-L1】
- **Versionado:** sincroniza `pyproject.toml`, `README.md` y `CHANGELOG.md` al publicar releases.【F:README.md†L1-L24】【F:pyproject.toml†L6-L9】
- **QA manual:** consulta `docs/testing.md` y `docs/qa/` para reportes históricos y procedimientos offline (`_render_for_test`).【F:docs/testing.md†L1-L86】
- **Convenciones de ramas:** TODO: confirmar (no se documenta política explícita).
- **Recursos adicionales:** `docs/cache_management.md`, `docs/security_validation.md`, `docs/operations.md` para procedimientos avanzados.【F:README.md†L77-L106】【F:docs/security_validation.md†L1-L42】【F:docs/operations.md†L1-L120】

