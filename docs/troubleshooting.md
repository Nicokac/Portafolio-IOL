# Guía de troubleshooting

Esta guía resume los síntomas más comunes que reportan usuarios y QA al operar con Portafolio IOL, junto con los pasos recomendados para diagnosticarlos y resolverlos. Cada bloque distingue los incidentes funcionales de portafolio de los técnicos/infraestructura.

## Portafolio y datos de mercado

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

- **El contenedor Docker termina con error de configuración.**
  - **Síntomas:** Los logs muestran fallos de variables obligatorias (`IOL_TOKENS_KEY` ausente) o problemas de permisos con `tokens_iol.json`.
  - **Diagnóstico rápido:** Inspecciona el archivo `.env` que usas con `docker run --env-file` y confirma rutas montadas.
  - **Resolución:**
    1. Revisa que `.env` contenga todas las claves obligatorias y monta volúmenes con permisos restrictivos (`chmod 600 tokens_iol.json`).
    2. Reconstruye la imagen si actualizaste dependencias (`docker build -t portafolio-iol .`) y vuelve a ejecutar `docker run` con el archivo corregido.

- **Los tests con Yahoo (`pytest -m live_yahoo`) fallan por rate limiting.**
  - **Síntomas:** `pytest` reporta `HTTPError` o `Timeout` al consultar Yahoo Finance.
  - **Diagnóstico rápido:** Comprueba si `RUN_LIVE_YF=1` está habilitado y si se lanzó el job nocturno en paralelo.
  - **Resolución:**
    1. Deshabilita temporalmente el job live en CI (`LIVE_YAHOO_SMOKE_FORCE_SKIP=true`) o ejecuta los tests fuera de la ventana nocturna.
    2. Reintenta con una red diferente o limita los símbolos bajo prueba para reducir la carga.
