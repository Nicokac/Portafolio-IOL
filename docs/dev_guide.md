# Guía técnica de desarrollo — v0.5.6

La release v0.5.6 consolida la ejecución offline de la pestaña de recomendaciones y estructura la documentación necesaria para contribuir sin depender de servicios externos. Este documento resume la arquitectura del proyecto, los flujos de QA y los procedimientos de mantenimiento.

## Estructura de carpetas

- `application/`: Servicios de dominio (recomendaciones, perfiles, backtesting, indicadores) y lógica de negocio de alto nivel.
- `services/`: Adaptadores y utilidades compartidas por la UI, como el servicio de caché y el ensamblador de métricas de riesgo.
- `ui/`: Componentes de Streamlit organizados por pestañas (`ui/tabs/`) y módulos auxiliares (`ui/charts/`).
- `shared/`: Helpers reutilizables (configuración, logging, proveedores de tiempo, constantes) diseñados para ejecutarse tanto online como offline.
- `docs/`: Documentación funcional, guías y reportes de QA. Incluye fixtures en `docs/fixtures/` y bitácoras de calidad en `docs/qa/`.
- `tests/`: Suite de Pytest dividida por capa (`application/`, `services/`, `ui/`, etc.) más los stubs y fixtures necesarios para pruebas deterministas.

## Flujo offline con `_render_for_test()`

1. Los datasets sintéticos se cargan desde `docs/fixtures/default/`. El smoke test y los ejemplos manuales usan `recommendations_sample.csv` para simular las sugerencias.
2. `_render_for_test()` inicializa `streamlit.session_state` con posiciones, perfil inversor y métricas de caché predeterminadas. No requiere autenticar contra IOL ni levantar la aplicación completa.
3. Durante la ejecución se redirigen `stdout`/`stderr` a buffers en memoria para evitar ruido en CI.
4. El helper normaliza columnas críticas (`sector`, `predicted_return_pct`, `expected_return`) antes de invocar `render_recommendations_tab()`.
5. Una vez renderizado, el estado persistente queda disponible bajo `st.session_state["_recommendations_state"]`, lo que permite inspeccionar el resultado desde los tests o herramientas de QA.

### Ejemplo de QA manual

```bash
python - <<'PY'
from ui.tabs import recommendations
import pandas as pd
from types import SimpleNamespace

df = pd.read_csv('docs/fixtures/default/recommendations_sample.csv')
state = SimpleNamespace(selected_mode='low_risk')
recommendations._render_for_test(df, state)
PY
```

El stub de Streamlit definido en `tests/conftest.py` se carga automáticamente cuando se ejecutan las pruebas, por lo que este flujo es reproducible en entornos sin interfaz gráfica.

## Ejecución de pruebas

- **Suite rápida:** `pytest -q` ejecuta toda la batería con la configuración de `pyproject.toml`.
- **Sin opciones adicionales:** `pytest --override-ini addopts=''` es útil para depurar casos individuales sin la sobrecarga de cobertura.
- **Smoke test offline:** `pytest -q tests/ui/test_render_for_test_smoke.py` valida que `_render_for_test()` se ejecute en menos de tres segundos y que alimente el estado de Streamlit.

Todos los comandos deben correrse desde la raíz del repositorio. Para aislar subconjuntos, apunta Pytest al directorio o archivo deseado (`pytest tests/services/test_cache_service.py`).

## Gestión de versiones y changelog

- Actualiza la versión visible en `README.md` y sincroniza la entrada correspondiente en `CHANGELOG.md`.
- Cada release debe incluir un reporte en `docs/qa/` describiendo el resultado de los smoke tests offline y cualquier validación manual.
- Cuando se agregan o remueven tests, documenta el impacto en esta guía y en `docs/testing.md`.

## Regeneración de fixtures

Los datasets sintéticos se regeneran con `scripts/generate_mock_data.py`:

```bash
python scripts/generate_mock_data.py --output docs/fixtures/default
```

El script produce perfiles, recomendaciones y snapshots de precios listos para alimentar `_render_for_test()`. Ejecutarlo cada vez que cambie la estructura de las columnas o se necesiten escenarios adicionales.

## Buenas prácticas de commits y PRs

- Mantén los commits enfocados en una sola intención (tests, documentación, ajustes puntuales). Utiliza mensajes en español con verbo en infinitivo (`Agregar smoke test offline`).
- Antes de abrir un PR, ejecuta la suite relevante (`pytest -q tests/ui/test_render_for_test_smoke.py` y los servicios afectados) y adjunta los resultados en la descripción.
- Documenta en el PR las rutas tocadas, los pasos de QA realizados y enlaza el reporte correspondiente de `docs/qa/` cuando aplique.
- Reutiliza el mensaje del PR en follow-ups, actualizándolo únicamente cuando se agreguen cambios significativos.

## Logs y reportes de QA

Los reportes de verificación manual y automatizada residen en `docs/qa/`. Cada smoke test debe registrar duración y estado (éxito o fallo) en un archivo con prefijo de versión (`docs/qa/v0.5.6-smoke-report.md`). Conserva estos reportes para auditar regresiones y documentar la reproducibilidad del flujo offline.
