# Guía de troubleshooting

Esta guía resume los síntomas más comunes que reportan usuarios y QA al operar con Portafolio IOL, junto con los pasos recomendados para diagnosticarlos y resolverlos. Cada bloque distingue los incidentes funcionales de portafolio de los técnicos/infraestructura.

## Claves API

> Nota: Esta guía corresponde a la release 0.3.29, enfocada en hardening/CI para reforzar los
> pipelines automáticos y las verificaciones de integridad sin alterar los flujos funcionales
> documentados en la serie 0.3.29.

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
  - **Diagnóstico rápido:** Verifica que estés en la release 0.3.29 o superior y que no haya código externo reescribiendo `st.session_state["resilience_timeline"]`.
  - **Resolución:**
    1. Actualiza el repositorio y reinstala dependencias si trabajas con un build antiguo.
    2. Comprueba que el stub de tests (`tests/conftest.py`) conserve los datos de sesión entre llamadas; limpia `st.session_state` solo al finalizar las aserciones.

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

- **Las notificaciones internas no aparecen tras refrescar el dashboard.**
  - **Síntomas:** El menú **⚙️ Acciones** ejecuta `⟳ Refrescar`, pero no se muestra el toast "Proveedor primario restablecido" ni el mensaje de cierre de sesión.
  - **Diagnóstico rápido:** Verifica que la versión visible indique `0.3.29` en el header/footer y que `st.toast` no esté sobreescrito en el entorno (suele ocurrir en notebooks o shells sin UI).
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
    python scripts/export_analysis.py --input ~/.portafolio_iol/snapshots --formats csv --output /tmp/exports_probe
    ```
    para descartar un problema con rutas relativas.
  - **Resolución:**
    1. Crea el directorio de destino (`mkdir -p exports`) o usa una ruta absoluta accesible.
    2. Verifica que `pandas` esté instalado en el entorno (`pip install -r requirements.txt`).
    3. Si necesitas ampliar el contenido, utiliza `--metrics help` o `--charts help` para revisar las claves disponibles y vuelve a ejecutar el comando con las opciones deseadas.

- **Los tests con Yahoo (`pytest -m live_yahoo`) fallan por rate limiting.**
  - **Síntomas:** `pytest` reporta `HTTPError` o `Timeout` al consultar Yahoo Finance.
  - **Diagnóstico rápido:** Comprueba si `RUN_LIVE_YF=1` está habilitado y si se lanzó el job nocturno en paralelo.
  - **Resolución:**
    1. Deshabilita temporalmente el job live en CI (`LIVE_YAHOO_SMOKE_FORCE_SKIP=true`) o ejecuta los tests fuera de la ventana nocturna.
    2. Reintenta con una red diferente o limita los símbolos bajo prueba para reducir la carga.
