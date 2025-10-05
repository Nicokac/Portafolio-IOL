# Guía de troubleshooting

Esta guía resume los síntomas más comunes que reportan usuarios y QA al operar con Portafolio IOL, junto con los pasos recomendados para diagnosticarlos y resolverlos. Cada bloque distingue los incidentes funcionales de portafolio de los técnicos/infraestructura.

## Claves API

> Nota: Esta guía corresponde a la release 0.3.30.10, enfocada en restaurar la bitácora unificada y las
> exportaciones multi-formato tras los incidentes de logging/export. Mantiene las cotizaciones en vivo
> sincronizadas, la procedencia propagada hacia `/Titulos/Cotizacion` y los metadatos de país en el
> portafolio, junto con los refuerzos del fallback jerárquico documentados en la serie 0.3.30.x.

## CI Checklist (0.3.30.10)

- **Suite legacy detectada.** Si el resumen de `pytest` menciona archivos dentro de `tests/legacy/`,
  ajustá el comando (`pytest --ignore=tests/legacy`) o revisá `norecursedirs` en `pyproject.toml` para
  evitar recolectar escenarios duplicados.
- **Importaciones legacy en código activo.** Ejecuta `rg "infrastructure\.iol\.legacy" application controllers services tests`
  y bloqueá el pipeline si aparecen coincidencias fuera de `tests/legacy/` o de pruebas de compatibilidad
  explícitas.
- **Pipelines sin artefactos.** Si un job finaliza sin adjuntar `coverage.xml`, `htmlcov/`, `analysis.zip`,
  `analysis.xlsx`, `summary.csv` o `analysis.log`, márcalo como fallido y reejecuta las etapas de `pytest --cov` y
  `scripts/export_analysis.py`. Los pasos deben apuntar al mismo directorio temporal (`$RUNNER_TEMP` o
  `tmp_path`).
- **Rutas inconsistentes de snapshots.** Cuando `scripts/export_analysis.py` no encuentra archivos,
  revisa la variable `SNAPSHOT_STORAGE_PATH` utilizada durante los tests. Configúrala explícitamente en
  el pipeline y replica la ruta al generar las exportaciones.
- **Cobertura divergente.** Si los reportes de cobertura muestran discrepancias, borra `htmlcov/` antes
  de ejecutar `pytest --cov` para evitar mezclar runs anteriores y confirma que el pipeline suba el
  artefacto actualizado, verificando que el endpoint `/Cotizacion` figure dentro del reporte.

- **Falta una clave y los servicios quedan en `disabled`.**
  - **Síntomas:** El health sidebar indica `disabled` para Alpha Vantage/Polygon/FMP/FRED/World Bank y el log muestra `Missing API key`.
  - **Diagnóstico rápido:** Confirma que las variables `ALPHA_VANTAGE_API_KEY`, `POLYGON_API_KEY`, `FMP_API_KEY`, `FRED_API_KEY` y `WORLD_BANK_API_KEY` estén definidas en `.env`, `config.json` o `secrets.toml`.
  - **Resolución:**
    1. Declara cada clave en el backend correspondiente. Para validar rápidamente qué detecta la aplicación ejecuta:
       ```bash
       python - <<'PY'
       from pprint import pprint
       from shared import settings

       pprint({
           "ALPHA_VANTAGE_API_KEY": settings.alpha_vantage_api_key,
           "POLYGON_API_KEY": settings.polygon_api_key,
           "FMP_API_KEY": settings.fmp_api_key,
           "FRED_API_KEY": settings.fred_api_key,
           "WORLD_BANK_API_KEY": settings.world_bank_api_key,
       })
       PY
       ```
    2. Si una clave no aplica (por ejemplo, sin Polygon), elimina al proveedor del orden configurado (`OHLC_PRIMARY_PROVIDER`, `OHLC_SECONDARY_PROVIDERS`, `MACRO_API_PROVIDER`) para evitar intentos fallidos.
    3. Reinicia la app y verifica en el health sidebar que el proveedor cambie a `success` o `fallback` según corresponda.

- **Claves inconsistentes entre entornos.**
  - **Síntomas:** En local la app funciona pero en CI aparece `unauthorized`.
  - **Diagnóstico rápido:** Revisa el pipeline para confirmar que se estén inyectando los secretos y que los jobs exporten las variables antes de lanzar la app o los tests.
  - **Resolución:**
    1. Replica la configuración de `.env` en el servicio de secretos de CI (GitHub Actions, GitLab, etc.).
    2. Asegura que los pasos de despliegue establezcan las variables antes de ejecutar `streamlit run app.py` o `pytest`.

## Health sidebar y resiliencia

- **El timeline de resiliencia no persiste tras un rerun.**
  - **Síntomas:** Luego de presionar **⟳ Refrescar**, el bloque **Resiliencia de proveedores** se vacía.
  - **Diagnóstico rápido:** Verifica que estés en la release 0.3.30.10 o superior, que `analysis.log` se regenere tras cada screening y que no haya código externo reescribiendo `st.session_state["resilience_timeline"]`.
  - **Resolución:**
    1. Actualiza el repositorio y reinstala dependencias si trabajas con un build antiguo.
    2. Comprueba que el stub de tests (`tests/conftest.py`) conserve los datos de sesión entre llamadas; limpia `st.session_state` solo al finalizar las aserciones.

- **La etiqueta "Logging y exports restaurados" no aparece en el sidebar.**
  - **Síntomas:** El banner superior muestra la versión `0.3.30.10`, pero el bloque de salud no adjunta el mensaje de logging/export y el feed live indica degradación aun cuando `/Titulos/Cotizacion` responde.
  - **Diagnóstico rápido:** Ejecuta `python tests/helpers/check_live_quotes.py` (o el script equivalente) para confirmar que el proveedor activo devuelve `last = price` y que `shared.version.DEFAULT_VERSION` coincide con la release actual.
  - **Resolución:**
    1. Revisa los logs de `services.quotes.live_quotes_flow` y verifica que `source="titulos"` llegue al `quotes_store`.
    2. Si estás en modo offline, habilita el flag `LIVE_QUOTES_ENABLED=1` y vuelve a iniciar la app para forzar la consulta en vivo.
    3. Comprueba que no existan interceptores sobrescribiendo `st.session_state["live_quotes_status"]`; en caso de encontrarlos, elimínalos o actualízalos para reflejar el nuevo flujo.

- **El bloque "Snapshots y almacenamiento" aparece vacío o en error.**
  - **Síntomas:** El health sidebar muestra `snapshot_hits = 0` pese a ejecutar screenings consecutivos, o aparece un mensaje "Ruta de snapshots inaccesible".
  - **Diagnóstico rápido:** Ejecuta el siguiente snippet para validar la ruta configurada y los permisos:
    ```bash
    python - <<'PY'
    import os
    from pathlib import Path
    from shared import settings

    snapshot_dir = getattr(settings, "SNAPSHOT_STORAGE_PATH", Path.home() / ".portafolio_iol" / "snapshots")
    print("ruta", snapshot_dir)
    print("existe", snapshot_dir.exists())
    print("permite escritura", os.access(snapshot_dir, os.W_OK))
    PY
    ```
  - **Resolución:**
    1. Crea manualmente el directorio (`mkdir -p ~/.portafolio_iol/snapshots`) y asigna permisos de escritura al usuario que ejecuta Streamlit.
    2. Asegura que ningún job de CI limpie el directorio entre corridas si necesitas comparar métricas persistentes; monta un volumen dedicado al ejecutar contenedores.
    3. Reinicia la app y lanza dos screenings con los mismos filtros. El contador `snapshot_hits` debería incrementarse en la segunda corrida y, al exportar con `scripts/export_analysis.py`, el archivo `kpis.csv` mostrará los KPI actualizados con la marca temporal correspondiente.

- **La jerarquía de fallback no coincide con las notas del screening.**
  - **Síntomas:** El health sidebar marca como último éxito un proveedor distinto del mostrado en las notas (`Datos macro (World Bank)` etc.).
  - **Diagnóstico rápido:** Ejecuta nuevamente el flujo descrito en el README y habilita `LOG_HEALTH_DEBUG=1` para registrar cada escalón de la degradación.
  - **Resolución:**
    1. Forza un escenario de fallo (deshabilita la clave primaria) y valida que el log muestre `primary=error`, `secondary=success`.
    2. Si el orden no se respeta, revisa `config.json` y los valores de `MACRO_API_PROVIDER`, `OHLC_PRIMARY_PROVIDER` y `OHLC_SECONDARY_PROVIDERS` para confirmar la jerarquía.
    3. Reporta el caso adjuntando el log y una captura del health sidebar si la inconsistencia persiste.

## Portafolio y datos de mercado

- **La app falla con `NameError: name 'apply_filters' is not defined`.**
  - **Síntomas:** Al iniciar sesión, Streamlit detiene la ejecución y
    muestra el error anterior en `controllers/portfolio/portfolio.py`.
  - **Diagnóstico rápido:** Confirmá que estés ejecutando la versión
    `0.3.24.2` o superior, donde el controlador delega la construcción
    del view-model al servicio cacheado.
  - **Resolución:**
    1. Actualizá el repo a la release más reciente.
    2. Si mantenés un fork, reemplazá llamadas directas a
       `apply_filters` por `PortfolioViewModelService.get_portfolio_view`
       y reutilizá `build_portfolio_viewmodel` con el snapshot devuelto.

- **El endpoint `/Cotizacion` responde HTTP 500 pese a la degradación automática.**
  - **Síntomas:** Las llamadas a `/Cotizacion` (UI, scripts o integraciones) devuelven `500 Internal Server Error` y la telemetría registra múltiples intentos de fallback sin éxito.
  - **Diagnóstico rápido:** Revisa los logs del servicio (`services/quotes` o equivalente) para confirmar si el error proviene del proveedor upstream y habilita `LOG_HEALTH_DEBUG=1` para ver la secuencia completa.
  - **Resolución:**
    1. Valida que las claves de los proveedores configurados estén activas y que la red permita las llamadas; luego relanza el flujo.
    2. Si el proveedor primario continúa fallando, fuerza el uso de un secundario o del snapshot disponible ajustando las variables `OHLC_PRIMARY_PROVIDER`/`SECONDARY` y reiniciando la app.
    3. Ejecuta `pytest tests/services` (o la suite específica del endpoint) para confirmar que la prueba de cobertura recién añadida siga pasando y que los mocks manejen los 500 como se espera.

- **No se puede guardar el token de IOL y la aplicación se cierra.**
  - **Síntomas:** Streamlit termina inmediatamente con un mensaje que indica que falta la clave Fernet o que no se permiten tokens sin cifrar.
  - **Diagnóstico rápido:** Revisa las variables `IOL_TOKENS_KEY` y `IOL_ALLOW_PLAIN_TOKENS` definidas en el `.env` o en `secrets.toml`.
  - **Resolución:**
    1. Genera una clave Fernet y declárala en `IOL_TOKENS_KEY`.
       ```bash
       python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
       ```
    2. Reinicia la app para que el token se guarde cifrado. En entornos de prueba puedes habilitar temporalmente `IOL_ALLOW_PLAIN_TOKENS=1`, pero vuelve a deshabilitarlo al finalizar.

- **La nota `📈 Yahoo Finance` queda en `fallback` o `error` de forma persistente.**
  - **Síntomas:** El health sidebar marca `fallback` para Yahoo, los screenings muestran severidad `⚠️` y las métricas se sirven desde el stub.
  - **Diagnóstico rápido:** Verifica conectividad de red, cuota de `yfinance` y que los TTL de caché (`CACHE_TTL_YF_*`, `YAHOO_FUNDAMENTALS_TTL`, `YAHOO_QUOTES_TTL`) no estén excedidos.
  - **Resolución:**
    1. Asegúrate de ejecutar la app con red estable y sin proxys que bloqueen Yahoo.
    2. Si sospechas de datos corruptos, elimina los archivos de caché bajo `infrastructure/cache/` y reinicia la app para forzar una descarga limpia.
    3. Confirma que las variables de TTL no estén fijadas en valores extremos que impidan la actualización.
- **Los pesos Markowitz aparecen como `NaN` o el gráfico queda vacío.**
  - **Síntomas:** La pestaña **Riesgo** no muestra la distribución de pesos y las exportaciones dejan columnas vacías para los pesos optimizados.
  - **Diagnóstico rápido:** Ejecuta `pytest tests/application/test_risk_metrics.py -k markowitz` para validar que la degradación controlada funciona e inspecciona que el preset no concentre todos los activos en un solo símbolo.
  - **Resolución:**
    1. Amplía el histórico (`period=1y` o superior) y repite el screening para garantizar una matriz de covarianzas invertible.
    2. Ajusta el preset para diversificar pesos antes de exportar o recalcula la optimización desde la UI.
    3. En CI, ejecuta `pytest tests/integration/` completo para regenerar snapshots con datos válidos y revisar el health sidebar en busca del estado de validación Markowitz.

- **El screening devuelve menos de 10 resultados o la tabla queda vacía.**
  - **Síntomas:** La telemetría del barrido muestra `universe final < 10` con severidad `⚠️`.
  - **Diagnóstico rápido:** Revisa los filtros activos, presets aplicados y disponibilidad de datos en Yahoo/stub.
  - **Resolución:**
    1. Relaja temporalmente el `score` mínimo (`MIN_SCORE_THRESHOLD`) o reduce filtros técnicos exigentes.
    2. Valida que las variables de entorno `OPPORTUNITIES_TARGET_MARKETS` y presets personalizados no estén recortando en exceso el universo.
    3. Si los datos provienen del stub, confirma que no se hayan alterado los archivos deterministas (`run_screener_stub`, `ta_fallback.csv`).

## Plataforma técnica y despliegue

- **Streamlit falla al iniciar con `ModuleNotFoundError`.**
  - **Síntomas:** La ejecución `streamlit run app.py` detiene con módulos ausentes (`streamlit`, `controllers`, etc.).
  - **Diagnóstico rápido:** Comprueba que la virtualenv esté activa y que las dependencias estén instaladas.
  - **Resolución:**
    1. Crea/activa el entorno virtual (`python -m venv .venv && source .venv/bin/activate`).
    2. Instala dependencias de producción (y opcionalmente QA) y relanza la app.
       ```bash
       pip install -r requirements.txt -r requirements-dev.txt
       streamlit run app.py
       ```
       > Si modificaste las dependencias en `pyproject.toml`, sincroniza la lista plana con `python scripts/sync_requirements.py` antes de reinstalar.

- **Las notificaciones internas no aparecen tras refrescar el dashboard.**
  - **Síntomas:** El menú **⚙️ Acciones** ejecuta `⟳ Refrescar`, pero no se muestra el toast "Proveedor primario restablecido" ni el mensaje de cierre de sesión.
  - **Diagnóstico rápido:** Verifica que la versión visible indique `0.3.30.10` en el header/footer, que el banner mencione "Logging y exports restaurados" y que `st.toast` no esté sobreescrito en el entorno (suele ocurrir en notebooks o shells sin UI).
  - **Resolución:**
    1. Ejecuta la app en Streamlit 1.32+ (requerido para `st.toast`) o, en suites headless, garantiza que el stub defina el método antes de lanzar la UI.
    2. Confirma que `st.session_state["show_refresh_toast"]` y `st.session_state["logout_done"]` no queden fijados en `False` permanente por código externo; limpia la sesión (`st.session_state.clear()`) y vuelve a probar.
    3. Si trabajas con el stub de pruebas, revisa `tests/conftest.py` y asegura que exponga un logger o impresión equivalente para simular la notificación.

- **El contenedor Docker termina con error de configuración.**
  - **Síntomas:** Los logs muestran fallos de variables obligatorias (`IOL_TOKENS_KEY` ausente) o problemas de permisos con `tokens_iol.json`.
  - **Diagnóstico rápido:** Inspecciona el archivo `.env` que usas con `docker run --env-file` y confirma rutas montadas.
  - **Resolución:**
    1. Revisa que `.env` contenga todas las claves obligatorias y monta volúmenes con permisos restrictivos (`chmod 600 tokens_iol.json`).
    2. Reconstruye la imagen si actualizaste dependencias (`docker build -t portafolio-iol .`) y vuelve a ejecutar `docker run` con el archivo corregido.

- **`scripts/export_analysis.py` falla al generar la exportación.**
  - **Síntomas:** El script termina con `FileNotFoundError` o `PermissionError` al escribir en `exports/`.
  - **Diagnóstico rápido:** Comprueba que la ruta indicada en `--output` exista y que el usuario tenga permisos de escritura. Ejecuta:
    ```bash
    python scripts/export_analysis.py --input ~/.portafolio_iol/snapshots --formats csv --output /tmp/probe
    ```
    para descartar un problema con rutas relativas. Verifica que se generen los CSV, el ZIP `analysis.zip`
    (cuando `--formats` incluya `csv`), el archivo `analysis.xlsx` y el resumen `summary.csv` dentro del
    subdirectorio del snapshot o en la raíz del directorio de exportaciones según corresponda.
- **Kaleido ausente y sin PNG en las exportaciones.**
  - **Síntomas:** `pytest` marca `kaleido no disponible` o la UI muestra una advertencia al generar el Excel.
  - **Diagnóstico rápido:** Ejecuta `python -c "import kaleido"`; si falla, instala la dependencia incluida en `requirements.txt`.
  - **Resolución:**
    1. Instala Kaleido en el entorno activo (`pip install -r requirements.txt` o `pip install kaleido`).
    2. Repite la exportación: el Excel incorporará los PNG y las suites de tests dejarán de saltar o advertir.
    3. Si prefieres continuar sin la librería (por ejemplo, en CI minimalista), el flujo seguirá generando el ZIP de CSV y mostrará la advertencia para informar del fallback.
- **Los snapshots persisten entre jobs en CI.**
  - **Síntomas:** Un job reutiliza datos de un pipeline anterior y la telemetría marca `snapshot_hits` altos aunque se espera un entorno limpio.
  - **Diagnóstico rápido:** Revisa si las variables `SNAPSHOT_BACKEND` y `SNAPSHOT_STORAGE_PATH` están configuradas en el pipeline.
  - **Resolución:**
    1. Fija `SNAPSHOT_BACKEND=null` para desactivar la persistencia en jobs que no necesitan exportar artefactos.
    2. Cuando debas validar exportaciones, apunta `SNAPSHOT_STORAGE_PATH` a la ruta temporal del runner (por ejemplo, `$RUNNER_TEMP`) y limpia el directorio al finalizar.
    3. Ejecuta `pytest tests/integration/` para asegurarte de que la suite multi-proveedor funciona con la configuración elegida y de que se generan los CSV, ZIP y Excel esperados por snapshot.
  - **Resolución:**
    1. Crea el directorio de destino (`mkdir -p exports`) o usa una ruta absoluta accesible.
    2. Verifica que `pandas` esté instalado en el entorno (`pip install -r requirements.txt`).
    3. Revisa que `analysis.zip`, `analysis.xlsx`, `summary.csv` y los CSV (`kpis.csv`, `positions.csv`, etc.) se generen dentro del subdirectorio del snapshot; si falta alguna columna en el resultado, confirma que `run_screener_stub` siga intacto y que no se hayan modificado los nombres esperados.

- **Los tests con Yahoo (`pytest -m live_yahoo`) fallan por rate limiting.**
  - **Síntomas:** `pytest` reporta `HTTPError` o `Timeout` al consultar Yahoo Finance.
  - **Diagnóstico rápido:** Comprueba si `RUN_LIVE_YF=1` está habilitado y si se lanzó el job nocturno en paralelo.
  - **Resolución:**
    1. Deshabilita temporalmente el job live en CI (`LIVE_YAHOO_SMOKE_FORCE_SKIP=true`) o ejecuta los tests fuera de la ventana nocturna.
    2. Reintenta con una red diferente o limita los símbolos bajo prueba para reducir la carga.
