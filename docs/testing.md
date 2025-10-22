# Guía de pruebas

Esta guía resume los prerequisitos y comandos necesarios para ejecutar la suite completa del proyecto, incluyendo verificaciones offline que ejercitan la pestaña de recomendaciones mediante `_render_for_test()`. Consulta `docs/dev_guide.md` para obtener una visión general de la arquitectura y del flujo de QA.

## Prerrequisitos

- Python 3.10 o superior.
- Dependencias de producción y QA instaladas:
  ```bash
  pip install -r requirements.txt -r requirements-dev.txt
  ```
  > `requirements.txt` se sincroniza desde `[project.dependencies]` de `pyproject.toml` con `python scripts/sync_requirements.py`. Ejecutalo si actualizás versiones antes de reinstalar dependencias.
- No es necesario instalar Streamlit. Los tests utilizan el stub definido en `tests/conftest.py`.
- Toda la suite activa vive en `tests/` (incluyendo `tests/legacy/` para referencias históricas); la carpeta `test/` fue
  retirada en la serie 0.7.x para evitar duplicaciones.
- Variables de entorno opcionales para pruebas específicas:
  - `RUN_LIVE_YF=1` habilita los tests que consultan Yahoo Finance en vivo.
  - `FRED_API_KEY` y `FRED_SECTOR_SERIES` permiten validar integraciones macro con datos reales. Por defecto se usan stubs deterministas.

## Suites recomendadas

### Ejecución completa

```bash
pytest
```

### Modo rápido sin opciones adicionales

```bash
pytest --override-ini addopts=''
```

Útil para ciclos de TDD locales o al depurar suites nuevas que no requieren medición de cobertura.

### Smoke test offline `_render_for_test()`

```bash
pytest -q tests/ui/test_render_for_test_smoke.py
```

El smoke test carga `docs/fixtures/default/recommendations_sample.csv`, ejecuta `_render_for_test()` con un estado mínimo y verifica que `st.session_state["_recommendations_state"]` contenga las recomendaciones enriquecidas sin emitir warnings. El objetivo es garantizar que el flujo offline permanezca ejecutable en menos de tres segundos.

### Subconjuntos frecuentes

- `pytest tests/services/test_cache_service.py`
- `pytest tests/application/test_recommendation_service.py`
- `pytest --override-ini addopts='' tests/ui/test_portfolio_ui.py -k risk`

## Pruebas manuales sugeridas (v0.5.x)

1. **Badge del caché predictivo.** Ejecuta `_render_for_test()` y comprueba que el badge exponga ratio de aciertos y TTL coherentes con el fixture.
2. **Simulador de rebalanceo.** Con el modo `diversify`, valida que el simulador compare métricas antes/después y que los toasts de progreso no registren warnings.
3. **Correlaciones sectoriales.** Navega a la pestaña "Correlaciones sectoriales" y verifica que el gráfico genere matrices deterministas con los datos sintéticos.
4. **Exportación adaptativa.** Desde la UI renderizada, dispara la exportación del reporte adaptativo y confirma que el archivo Markdown se cree en `docs/reports/`.

## Generadores aleatorios reproducibles

El módulo `application.risk_service` expone un generador persistente `default_rng`, inicializado con `numpy.random.SeedSequence`, para todas las simulaciones Monte Carlo. Durante las pruebas podés inyectar tu propio generador pasando el parámetro `rng` a `monte_carlo_simulation`, por ejemplo:

```python
from numpy.random import SeedSequence, default_rng

result = monte_carlo_simulation(
    returns,
    weights,
    n_sims=1024,
    horizon=64,
    rng=default_rng(SeedSequence(2024)),
)
```

De esta manera cada test controla explícitamente la semilla sin depender de `numpy.random.seed`, y los escenarios siguen siendo reproducibles incluso en ejecuciones en paralelo.

## Checklist de CI (v0.5.x)

1. **Suite determinista sin legacy.** Ejecuta `pytest --maxfail=1 --disable-warnings -q --ignore=tests/legacy` y verifica que no se recolecten casos desde `tests/legacy/`.
2. **Cobertura de servicios críticos.** Corre `pytest --cov=application --cov=controllers --cov-report=term-missing --cov-report=xml` y sube `coverage.xml` más `htmlcov/` como artefactos.
3. **Auditoría de imports legacy.** Ejecuta `rg "infrastructure\\.iol\\.legacy" application controllers services tests` y marca la corrida como fallida si aparecen coincidencias fuera de `tests/legacy/`.
4. **Exportaciones consistentes.** Invoca `python scripts/export_analysis.py --input ~/.portafolio_iol/snapshots --formats both --output exports/ci` y verifica que se generen los paquetes (`analysis.zip`, `analysis.xlsx`, `summary.csv`, `environment.json`, logs rotados).
5. **Monitoreo y TTLs visibles.** Ejecuta la app en modo headless y captura la pestaña **Monitoreo** para confirmar los TTL configurados en `shared.settings`.
6. **Smoke offline documentado.** Asegura que el resultado del comando `pytest -q tests/ui/test_render_for_test_smoke.py` quede registrado en `docs/qa/` junto con la duración medida.
7. **Puerta de seguridad.** Corre `bandit -r application controllers services` y `pip-audit --requirement requirements.txt --requirement requirements-dev.txt` como parte del pipeline.

## Apéndice: Historial de pruebas legacy (< v0.4.0)

Las notas siguientes corresponden a releases previas a la serie 0.4.x y se conservan como referencia histórica para auditorías específicas.

La release 0.3.4.3 consolida la telemetría dentro de la pestaña Monitoreo, mantiene la rotación automática de `analysis.log` y añade verificaciones visuales sobre el sidebar unificado, el badge global de estado y el nuevo bloque de enlaces del footer, por lo que los tests deben asegurar que los snapshots y los logs comprimidos generados por la app se publiquen como artefactos. La release 0.3.4.4.2 profundiza este trabajo al apilar los controles del sidebar en tarjetas verticales con feedback visual específico por sección, por lo que las verificaciones manuales deben incluir capturas del nuevo layout y la animación de feedback al aplicar filtros. La release 0.3.4.4.5 extiende esta validación al heatmap de riesgo, exigiendo evidencias de que cada tipo de activo se correlaciona únicamente con sus símbolos homogéneos, que los CEDEARs omiten acciones locales (LOMA, YPFD, TECO2) y que existe una pestaña específica para las Acciones locales con su propio tablero de correlaciones. La release 0.3.4.4.4 refuerza el análisis de riesgo alineando el heatmap de correlaciones con la clasificación del portafolio base. Antes de descargar históricos se aplica un mapeo canónico por símbolo, evitando que los CEDEARs compartan matriz con acciones locales y descartando explícitamente tickers como LOMA, YPFD o TECO2 cuando el payload de precios los etiqueta de forma ambigua.

> Las pruebas visuales se deben realizar mediante inspección manual del layout, verificando jerarquía tipográfica, alineación y visibilidad del menú de acciones.

### Suites legacy (deprecated)

La carpeta `tests/legacy/` contiene casos heredados que duplican escenarios ya cubiertos en la suite principal. Se excluye de la recolección estándar para mantener los tiempos de CI y sirve como histórico para comparar comportamientos. Si necesitás auditarlos manualmente, ejecutalos de forma explícita:

```bash
pytest tests/legacy
```

### Stubs de Streamlit y control de fixtures

Las suites de UI y sidebar utilizan un stub definido en `tests/conftest.py` que emula las funciones y componentes de Streamlit (sidebar, formularios, columnas, etc.). Algunas consideraciones para extender o depurar estas pruebas:

- El stub registra cada llamada y expone el helper `streamlit_stub.get_records("tipo")` para inspeccionar los eventos renderizados por los componentes.
- Métodos como `set_button_result`, `set_checkbox_result` y `set_form_submit_result` permiten simular la interacción del usuario desde los tests sin depender de `streamlit.testing`.
- Si se añade nuevo comportamiento en la UI que invoque APIs de Streamlit no cubiertas, ampliá el stub agregando el método correspondiente y registrando su uso.
- Para validar las notificaciones internas (`st.toast`), monkeypatchea la función como en `application/test/test_login_flow.py` y aserta sobre las banderas de `session_state` (`show_refresh_toast`, `logout_done`) o sobre el stub personalizado que definas.

Gracias a esta infraestructura, las suites pueden ejecutarse en entornos mínimos (CI headless, containers livianos) sin requerir dependencias binarias de Streamlit.

Para acotar la ejecución a subconjuntos específicos, podés lanzar `pytest` con rutas o filtros frecuentes:

- `pytest tests/ui/test_portfolio_ui.py -k risk`
- `pytest --override-ini addopts='' tests/controllers/test_risk_filtering.py`
