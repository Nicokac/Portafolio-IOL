# Guía de pruebas

Esta guía resume los prerequisitos y comandos necesarios para ejecutar la suite completa del
proyecto, incluyendo las verificaciones opcionales que dependen de servicios externos.

## Prerrequisitos

- Python 3.10 o superior.
- Dependencias de producción y QA instaladas:
  ```bash
  pip install -r requirements.txt -r requirements-dev.txt
  ```
- No es necesario instalar Streamlit para ejecutar la suite. Los tests incorporan un stub local
  que reemplaza al módulo y provee las APIs utilizadas por la UI.
- Variables de entorno opcionales para pruebas específicas:
  - `RUN_LIVE_YF=1` habilita los tests que consultan Yahoo Finance en vivo.
  - `FRED_API_KEY` y `FRED_SECTOR_SERIES` permiten ejercitar llamadas reales a FRED en entornos
    donde se desea validar la integración macro. La suite estándar utiliza stubs, por lo que no son
    obligatorias.

## Suite completa

Ejecuta todas las pruebas desde la raíz del repositorio:

```bash
pytest
```

El proyecto incorpora `pytest.ini` con marcadores y configuración de logging. La ejecución completa
usa los stubs deterministas para mantener resultados reproducibles.

### Suites legacy

La carpeta `tests/legacy/` contiene casos heredados que duplican escenarios ya cubiertos en la suite principal. Se excluye de la recolección estándar para mantener los tiempos de CI. Si necesitas auditarlos manualmente, ejecútalos de forma explícita:

```bash
pytest tests/legacy
```

### Stubs de Streamlit y control de fixtures

Las suites de UI y sidebar utilizan un stub definido en `tests/conftest.py` que emula las
funciones y componentes de Streamlit (sidebar, formularios, columnas, etc.). Algunas
consideraciones para extender o depurar estas pruebas:

- El stub registra cada llamada y expone el helper `streamlit_stub.get_records("tipo")` para
  inspeccionar los eventos renderizados por los componentes.
- Métodos como `set_button_result`, `set_checkbox_result` y `set_form_submit_result` permiten
  simular la interacción del usuario desde los tests sin depender de `streamlit.testing`.
- Si se añade nuevo comportamiento en la UI que invoque APIs de Streamlit no cubiertas, amplia el
  stub agregando el método correspondiente y registrando su uso.
- Para validar las notificaciones internas (`st.toast`), monkeypatchea la función como en
  `application/test/test_login_flow.py` y aserta sobre las banderas de `session_state`
  (`show_refresh_toast`, `logout_done`) o sobre el stub personalizado que definas.

Gracias a esta infraestructura, las suites pueden ejecutarse en entornos mínimos (CI headless,
containers livianos) sin requerir dependencias binarias de Streamlit.

Para acotar la ejecución a subconjuntos específicos, puedes lanzar `pytest` con rutas o filtros
frecuentes:

- `pytest controllers/test/test_opportunities_controller.py`: ejecuta sólo las pruebas del
  controlador de oportunidades.
- `pytest tests/ui/test_portfolio_ui.py -k risk`: limita la ejecución a los escenarios que cubren
  las visualizaciones de riesgo renderizadas en la UI.

### Validación de snapshots y almacenamiento persistente

La release 0.3.29, orientada a hardening/CI, introduce contadores de snapshots y telemetría de
almacenamiento. Para cubrirlos en QA combina pruebas automáticas y verificaciones manuales:

- `pytest tests/test_sidebar_controls.py -k snapshot`: comprueba que los presets persistan en
  `st.session_state["controls_snapshot"]` y que el estado se limpie correctamente al cerrar sesión.
- `pytest services/test/test_portfolio_view_model.py`: valida que `PortfolioViewSnapshot` reutilice
  el cálculo previo cuando el dataset no cambia, asegurando que el contador de hits aumente en la UI.
- `pytest tests/integration/test_opportunities_flow.py -k snapshot`: smoke test integrado que
  verifica la creación y el consumo de snapshots durante screenings consecutivos.

Para reproducir la telemetría manualmente:

1. Ejecuta `streamlit run app.py` y lanza dos screenings con los mismos filtros. Observa cómo el
   bloque **Snapshots y almacenamiento** aumenta `snapshot_hits` en la segunda corrida.
2. Exporta el análisis enriquecido desde la línea de comandos para validar la paridad con la UI:
   ```bash
   python scripts/export_analysis.py --input ~/.portafolio_iol/snapshots --formats both --output exports/manual_checks
   ```
   Revisa `exports/manual_checks/<snapshot>/kpis.csv` y confirma que la columna `generated_at`
   coincida con el valor mostrado en el health sidebar. El archivo `exports/manual_checks/summary.csv`
   resume los KPI crudos (`raw_value`) para comparar rápidamente contra la UI.

## Pruebas con APIs en vivo

Los tests marcados como `live_yahoo` consultan Yahoo Finance y se consideran opcionales. Para
incluirlos debes exportar la variable y filtrar por el marcador:

```bash
export RUN_LIVE_YF=1
pytest -m live_yahoo
```

Estos escenarios dependen de datos de mercado en tiempo real y pueden variar entre corridas. Se
recomienda ejecutarlos sólo en entornos controlados.

## Flags útiles y depuración

- `pytest -k <expresión>`: corre únicamente los tests cuyo nombre coincide con la expresión.
- `pytest -x`: detiene la ejecución al primer fallo (útil para ciclos cortos).
- `pytest --maxfail=1 --disable-warnings -q`: modo compacto para pipelines CI.
- `pytest --ff`: reejecuta primero los tests fallidos en la corrida previa.

El proyecto utiliza fixtures que escriben artefactos temporales bajo `tmp_path`. Asegúrate de contar
con permisos de escritura en el directorio temporal del sistema.

## Integración en CI/CD

Para mantener las ejecuciones reproducibles en pipelines, establece las variables de entorno que
desactivan integraciones externas y ejecuta la suite en modo compacto. Los siguientes ejemplos
ilustran cómo instalar dependencias, inyectar los mocks por defecto y lanzar `pytest` en los sistemas
de CI más comunes.

### GitHub Actions

```yaml
name: qa

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  tests:
    runs-on: ubuntu-latest
    env:
      RUN_LIVE_YF: "0"           # Fuerza el uso de stubs para Yahoo Finance
      FRED_API_KEY: ""          # Variables vacías para evitar llamadas reales
      WORLD_BANK_API_KEY: ""
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Instalar dependencias
        run: |
          pip install -r requirements.txt -r requirements-dev.txt
      - name: Ejecutar pruebas
        run: pytest --maxfail=1 --disable-warnings -q
      - name: Generar coverage
        run: pytest --cov=application --cov=controllers --cov-report=term --cov-report=html
      - name: Publicar artefacto coverage
        uses: actions/upload-artifact@v4
        with:
          name: htmlcov
          path: htmlcov
```

El artefacto `htmlcov/index.html` queda disponible en la sección "Artifacts" del job. Descárgalo y
abre el archivo en tu navegador para revisar los módulos con menor cobertura.

### GitLab CI

```yaml
stages:
  - test

pytest:
  stage: test
  image: python:3.10-slim
  variables:
    RUN_LIVE_YF: "0"
    FRED_API_KEY: ""
    WORLD_BANK_API_KEY: ""
    PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"
  cache:
    paths:
      - .cache/pip
  script:
    - pip install -r requirements.txt -r requirements-dev.txt
    - pytest --maxfail=1 --disable-warnings -q
    - pytest --cov=application --cov=controllers --cov-report=term-missing --cov-report=xml
  artifacts:
    when: always
    paths:
      - coverage.xml
      - htmlcov/
    reports:
      cobertura: coverage.xml
```

En GitLab, el reporte Cobertura queda adjunto al job y puede visualizarse en la pestaña **CI/CD →
Pipelines → Jobs → Artifacts**. Para el detalle HTML, descarga el directorio `htmlcov` y abre
`index.html` localmente.
