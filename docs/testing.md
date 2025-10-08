# Guía de pruebas

Esta guía resume los prerequisitos y comandos necesarios para ejecutar la suite completa del
proyecto, incluyendo las verificaciones opcionales que dependen de servicios externos.

## Prerrequisitos

- Python 3.10 o superior.
- Dependencias de producción y QA instaladas:
  ```bash
  pip install -r requirements.txt -r requirements-dev.txt
  ```
  > `requirements.txt` se sincroniza desde `[project.dependencies]` de `pyproject.toml` con `python scripts/sync_requirements.py`. Asegurate de correrlo si cambiaste las versiones fijadas antes de reinstalar dependencias.
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

Cuando necesites una corrida rápida sin la sobrecarga de cobertura puedes
invocar Pytest anulando el valor de `addopts` definido en `pyproject.toml`:

```bash
pytest --override-ini addopts=''
```

Esto resulta útil para los ciclos de TDD locales o al depurar suites nuevas
que no requieren medir cobertura.

El proyecto incorpora `pytest.ini` con marcadores y configuración de logging. La ejecución completa
usa los stubs deterministas para mantener resultados reproducibles. La release 0.3.4.3 consolida la
telemetría dentro de la pestaña Monitoreo, mantiene la rotación automática de `analysis.log` y
añade verificaciones visuales sobre el sidebar unificado, el badge global de estado y el nuevo bloque
de enlaces del footer, por lo que los tests deben asegurar que los snapshots y los logs comprimidos
generados por la app se publiquen como artefactos. La release 0.3.4.4.2 profundiza este trabajo al
apilar los controles del sidebar en tarjetas verticales con feedback visual específico por sección,
por lo que las verificaciones manuales deben incluir capturas del nuevo layout y la animación de
feedback al aplicar filtros. La release 0.3.4.4.5 extiende esta validación al heatmap de riesgo,
exigiendo evidencias de que cada tipo de activo se correlaciona únicamente con sus símbolos
homogéneos, que los CEDEARs omiten acciones locales (LOMA, YPFD, TECO2) y que existe una pestaña
específica para las Acciones locales con su propio tablero de correlaciones.
feedback al aplicar filtros. La release 0.3.4.4.4 extiende esta validación al heatmap de riesgo,
exigiendo evidencias de que cada tipo de activo se correlaciona únicamente con sus símbolos
homogéneos y que los CEDEARs omiten acciones locales (LOMA, YPFD, TECO2) incluso cuando las
cotizaciones llegan etiquetadas de forma inconsistente.

> Las pruebas visuales se deben realizar mediante inspección manual del layout, verificando jerarquía tipográfica, alineación y visibilidad del menú de acciones.

### Pruebas manuales sugeridas (0.3.4.3)

1. **Sidebar de controles apilado.** Abrí la aplicación en resoluciones desktop y medianas para validar que las tarjetas de Actualización, Filtros, Moneda, Orden, Gráficos y Acciones se rendericen una debajo de la otra con padding uniforme, chips activos, tooltips cortos y botones de refresco/cierre funcionando.
2. **Pestaña Monitoreo activa.** Navegá a la pestaña **Monitoreo** y confirmá que el healthcheck conserva las secciones de dependencias, snapshots, oportunidades y diagnósticos, registrando TTLs y latencias con la misma profundidad que el antiguo sidebar.
3. **Badge global y footer.** Revisá que bajo el encabezado principal aparezca el badge de estado general y que el footer incluya el bloque de enlaces útiles con contraste reducido en los metadatos.
4. **Heatmap alineado por tipo.** En la pestaña **Riesgo**, filtrá por CEDEARs y confirmá que el heatmap solo muestra tickers del catálogo base, excluyendo LOMA, YPFD y TECO2. Capturá evidencia de la advertencia cuando un tipo no tiene suficientes símbolos para generar la matriz.

### Generadores aleatorios reproducibles

El módulo `application.risk_service` expone un generador persistente `default_rng`, inicializado con
`numpy.random.SeedSequence`, para todas las simulaciones Monte Carlo. Durante las pruebas podés
inyectar tu propio generador pasando el parámetro `rng` a `monte_carlo_simulation`, por ejemplo:

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

De esta manera cada test controla explícitamente la semilla sin depender de `numpy.random.seed`, y
los escenarios siguen siendo reproducibles incluso cuando se ejecutan en paralelo.

## CI Checklist (0.3.4.3)

1. **Suite determinista sin legacy.** Ejecuta `pytest --maxfail=1 --disable-warnings -q --ignore=tests/legacy` y
   verifica que el resumen final no recolecte casos desde `tests/legacy/`.
2. **Cobertura obligatoria con foco en `/Cotizacion`.** Corre `pytest --cov=application --cov=controllers --cov-report=term-missing --cov-report=html --cov-report=xml`
   y asegúrate de subir `coverage.xml` y el directorio `htmlcov/` como artefactos del job, revisando que
   las rutas del endpoint de cotizaciones queden dentro del reporte.
3. **Auditoría de importaciones legacy.** Añade un paso que ejecute
   `rg "infrastructure\.iol\.legacy" application controllers services tests` y marque el pipeline como
   fallido si aparecen coincidencias fuera de `tests/legacy/` o de fixtures destinados a compatibilidad.
4. **Exportaciones consistentes.** Invoca `python scripts/export_analysis.py --input ~/.portafolio_iol/snapshots --formats both --output exports/ci`
 (o reutiliza `tmp_path` en las suites) y revisa que cada snapshot incluya los CSV (`kpis.csv`,
  `positions.csv`, `history.csv`, `contribution_by_symbol.csv`, etc.), el ZIP `analysis.zip`, el Excel
  `analysis.xlsx`, el resumen `summary.csv`, el snapshot de entorno (`environment.json`) y el paquete de logs rotados (`analysis.log` + `.gz`).
5. **TTLs y monitoreo visibles.** Ejecuta la app en modo headless y capturá la pestaña **Monitoreo** para confirmar que cada proveedor muestra el TTL restante configurado en `CACHE_TTL_*` y que el timeline de sesión despliega los hitos (login, screenings, exportaciones) en orden.
6. **Sidebar apilado y badge global.** Capturá el sidebar con los bloques apilados (Actualización, Filtros, Moneda, Orden, Gráficos y Acciones) verificando que cada tarjeta conserve padding uniforme, tooltips cortos y feedback visual al aplicar filtros. Confirmá también que el bloque **⚙️ Configuración general** y el badge global de salud se rendericen sin solaparse con el nuevo layout.
7. **Footer con enlaces útiles.** Acompañá el pipeline con capturas o vídeos que muestren el bloque de enlaces útiles en el footer y el contraste suavizado de los metadatos.
8. **Checklist previa al merge.** Antes de aprobar la release inspecciona los artefactos del pipeline y
  confirma que `htmlcov/`, `coverage.xml`, `analysis.zip`, `analysis.xlsx`, `summary.csv`, el snapshot de entorno y
  los archivos `analysis.log*` rotados (desde `~/.portafolio_iol/logs/`) estén adjuntos. Si falta alguno, la ejecución debe considerarse fallida.
9. **Puerta de seguridad.** Ejecuta `bandit -r application controllers services` para auditar llamadas inseguras
  y `pip-audit --requirement requirements.txt --requirement requirements-dev.txt` para identificar
  dependencias vulnerables. Ambos comandos deben formar parte del pipeline y bloquear el merge ante
  hallazgos críticos.
9. **Verificación del feed live.** Incluye un paso que ejecute `pytest tests/integration/test_quotes_flow.py`
   (o el job equivalente) y aserte que la UI muestre la etiqueta "Observabilidad operativa" con el TTL restante,
   el bloque de **Descargas de observabilidad** habilite la descarga del snapshot de entorno y que `analysis.log`
   registre la rotación correspondiente cuando `/Titulos/Cotizacion` entrega precios en tiempo real.

### Suites legacy (deprecated)

La carpeta `tests/legacy/` contiene casos heredados que duplican escenarios ya cubiertos en la suite
principal. Se excluye de la recolección estándar para mantener los tiempos de CI y sirve como
histórico para comparar comportamientos. Los flujos modernos deben utilizar
`services.portfolio_view.PortfolioViewModelService`, `application.portfolio_viewmodel` y los fixtures
de `tests/conftest.py` (especialmente `streamlit_stub`) en lugar de helpers legacy. Si necesitas
auditarlos manualmente, ejecútalos de forma explícita:
El paquete `infrastructure.iol.legacy` permanece disponible únicamente para mantener compatibilidad con integraciones antiguas; su importación ahora emite una advertencia de deprecación y no participa del flujo principal. La carpeta `tests/legacy/` conserva los escenarios originales para auditorías puntuales y continúa excluida de la recolección estándar para preservar los tiempos de CI. Si necesitás ejecutarlos manualmente, hacelo de forma explícita:

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

La release 0.3.4.0 restablece la bitácora unificada, mantiene el flujo de cotizaciones en vivo, propaga
el indicador de procedencia a `/Titulos/Cotizacion`, añade el país al view-model del portafolio, expone los TTL configurados para cada proveedor dentro del health sidebar y despliega un timeline de sesión con los hitos (login, screenings, exportaciones, fallbacks) asociados a cada ejecución.
Las
pruebas continúan reforzando el fallback jerárquico mientras verifican que el feed live quede etiquetado
correctamente en la UI, que `~/.portafolio_iol/logs/analysis.log` capture cada screening con el `session_tag` correspondiente y que los artefactos por país lleguen a
los exports aun cuando Kaleido no esté instalado (Excel siempre se genera con tablas aunque falten PNG). Para cubrirlos en QA combina pruebas automáticas y verificaciones manuales:

- `pytest tests/test_sidebar_controls.py -k snapshot`: comprueba que los presets persistan en
  `st.session_state["controls_snapshot"]` y que el estado se limpie correctamente al cerrar sesión.
- `pytest services/test/test_portfolio_view_model.py`: valida que `PortfolioViewSnapshot` reutilice
  el cálculo previo cuando el dataset no cambia, asegurando que el contador de hits aumente en la UI.
- `pytest tests/integration/test_opportunities_flow.py -k snapshot`: smoke test integrado que
  verifica la creación y el consumo de snapshots durante screenings consecutivos.

Para reproducir la telemetría manualmente:

1. Ejecuta `streamlit run app.py` y lanza dos screenings con los mismos filtros. Observa cómo el
   bloque **Snapshots y almacenamiento** aumenta `snapshot_hits` en la segunda corrida y cómo el
   timeline de sesión incorpora cada evento con timestamp, origen y `session_tag`.
2. Exporta el análisis enriquecido desde la línea de comandos para validar la paridad con la UI:
   ```bash
   python scripts/export_analysis.py --input ~/.portafolio_iol/snapshots --formats both --output exports/manual_checks
   ```
   Revisa `exports/manual_checks/<snapshot>/kpis.csv` y confirma que la columna `generated_at`
   coincida con el valor mostrado en el health sidebar. El archivo `exports/manual_checks/summary.csv`
   resume los KPI crudos (`raw_value`) para comparar rápidamente contra la UI. Cada subdirectorio
   incluye los CSV de tablas, el ZIP `analysis.zip` con el bundle y el Excel `analysis.xlsx` con las
   hojas y gráficos equivalentes a la UI.

### Validaciones Markowitz

- Ejecuta `pytest tests/application/test_risk_metrics.py -k markowitz` para confirmar que la
  optimización degrada a pesos `NaN` ante matrices singulares sin romper la UI.
- Valida que la pestaña de riesgo no renderice la distribución cuando los pesos no están disponibles;
  puedes simularlo filtrando el portafolio a dos activos idénticos y revisando los logs generados por la
  pestaña.
- La suite `pytest tests/integration/test_portfolio_tabs.py` confirma que las pestañas siguen
  operativas y que los presets mantienen pesos normalizados (suma igual a 1) tras las nuevas
  validaciones.

### Configuración del backend de snapshots en CI

El módulo `services.snapshots` difiere la configuración automática hasta la primera llamada a
`save_snapshot` o `list_snapshots`. Esto permite que `app.py`, scripts o fixtures de `pytest`
inyecten un backend explícito mediante `snapshots.configure_storage(...)` antes de utilizar las
APIs públicas.

En pipelines efímeros se recomienda declarar explícitamente las variables de entorno o fijar la
configuración al inicio de la suite para evitar que los artefactos se escriban en el workspace
compartido:

- `SNAPSHOT_BACKEND`: acepta `json`, `sqlite` o `null`. Usa `null` para habilitar
  `NullSnapshotStorage` y desactivar por completo la persistencia durante los tests.
- `SNAPSHOT_STORAGE_PATH`: ruta absoluta donde se almacenarán los archivos de snapshots cuando se
  utiliza un backend basado en disco. En runners como GitHub Actions puedes apuntar a
  `$RUNNER_TEMP/portfolio_snapshots.json` para generar los archivos en un directorio temporal que se
  limpia automáticamente.
- Para validar exportaciones o flujos multi-proveedor dentro del pipeline, fija
  `SNAPSHOT_STORAGE_PATH` a la carpeta provista por `tmp_path` (por ejemplo, `${{ runner.temp }}`) y
  ejecuta `pytest tests/integration/` completo. La suite ejerce los degradadores de proveedores y
  confirma que las exportaciones generen CSV, ZIP y Excel por snapshot.

Si necesitás desactivar la persistencia de snapshots durante CI, podés forzar el backend nulo en el
hook de `pytest_sessionstart` o en un fixture de alcance global:

```python
from services import snapshots


@pytest.fixture(scope="session", autouse=True)
def disable_snapshots_for_ci():
    snapshots.configure_storage(backend="null")
```

Cuando un test o script requiera un backend efímero sin afectar `_STORAGE`, utilizá el helper
`temporary_snapshot_storage` o creá la instancia manualmente:

```python
with snapshots.temporary_snapshot_storage(backend="json", path=tmp_path) as session:
    record = session.save_snapshot("portfolio", payload={}, metadata={})

storage = snapshots.create_storage(backend="sqlite", path=tmp_path / "snap.db")
storage.save_snapshot("portfolio", payload={}, metadata={})
```

Ambos métodos permiten trabajar con backends aislados sin tocar `_STORAGE` global.

Tras cada ejecución conviene borrar cualquier archivo residual para mantener el entorno limpio:

```bash
rm -f data/snapshots.json data/snapshots.db
find "$RUNNER_TEMP" -maxdepth 1 -type f -name "portfolio_snapshots.*" -delete
```

Las fixtures de `pytest` que dependen de snapshots (por ejemplo, en `tests/services/test_snapshots.py`
y `tests/integration/test_snapshot_export_flow.py`) reconfiguran el backend hacia `tmp_path` o
`NullSnapshotStorage` y restauran la configuración global al finalizar cada caso. Si añadís nuevos
escenarios que persistan snapshots, reutiliza el mismo patrón para no contaminar otros tests.

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
      - name: Generar exportaciones CI
        run: |
          python scripts/export_analysis.py --input ~/.portafolio_iol/snapshots --formats both --output exports/ci
          find exports/ci -maxdepth 2 -type f || true
      - name: Publicar artefacto coverage
        uses: actions/upload-artifact@v4
        with:
          name: htmlcov
          path: htmlcov
      - name: Publicar exportaciones
        uses: actions/upload-artifact@v4
        with:
          name: exports-ci
          path: |
            exports/ci
```

El artefacto `htmlcov/index.html` queda disponible en la sección "Artifacts" del job. Descárgalo y
abre el archivo en tu navegador para revisar los módulos con menor cobertura. El paso
`Generar exportaciones CI` asume que los tests o jobs previos generaron snapshots bajo
`~/.portafolio_iol/snapshots`; si tu pipeline usa rutas temporales diferentes, ajusta los parámetros
`--input` y `--output` o crea snapshots de prueba antes de invocar el script.

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
    - python scripts/export_analysis.py --input ~/.portafolio_iol/snapshots --formats both --output exports/ci
  artifacts:
    when: always
    paths:
      - coverage.xml
      - htmlcov/
      - exports/ci/
    reports:
      cobertura: coverage.xml
```

En GitLab, el reporte Cobertura queda adjunto al job y puede visualizarse en la pestaña **CI/CD →
Pipelines → Jobs → Artifacts**. Para el detalle HTML, descarga el directorio `htmlcov` y abre
`index.html` localmente. El directorio `exports/ci` publicado como artefacto debe contener los CSV,
`analysis.zip`, `analysis.xlsx` y `summary.csv`; si queda vacío, revisa las rutas de snapshots usadas
por la suite.
