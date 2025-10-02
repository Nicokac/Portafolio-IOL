# Guía de pruebas

Esta guía resume los prerequisitos y comandos necesarios para ejecutar la suite completa del
proyecto, incluyendo las verificaciones opcionales que dependen de servicios externos.

## Prerrequisitos

- Python 3.10 o superior.
- Dependencias de producción y QA instaladas:
  ```bash
  pip install -r requirements.txt -r requirements-dev.txt
  ```
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

Para acotar la ejecución a un paquete o archivo en particular, indica la ruta:

```bash
pytest controllers/test/test_opportunities_controller.py
```

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
