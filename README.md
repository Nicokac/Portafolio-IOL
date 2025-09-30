# Portafolio IOL

Aplicación Streamlit para consultar y analizar carteras de inversión en IOL.

> Nota: todos los timestamps visibles provienen de `shared.time_provider.TimeProvider` y se muestran
> en formato `YYYY-MM-DD HH:MM:SS` (UTC-3). El footer de la aplicación se actualiza en cada
> renderizado con la hora de Argentina.

## Uso del proveedor de tiempo

Para generar fechas consistentes en toda la aplicación, importa la clase `TimeProvider`:

```python
from shared.time_provider import TimeProvider

timestamp = TimeProvider.now()          # "2025-09-21 10:15:42"
moment = TimeProvider.now_datetime()    # datetime consciente de zona (UTC-3)
```

- `TimeProvider.now()` devuelve la representación en texto lista para mostrar en la interfaz.
- `TimeProvider.now_datetime()` expone el mismo instante como un objeto `datetime` con zona horaria de Buenos Aires.

El método `TimeProvider.now_datetime()` retorna un `datetime` consciente de zona en UTC-3, lo que garantiza que el `tzinfo` se mantenga alineado con Buenos Aires para todo cálculo dentro de la aplicación. Algunos integradores —como el cliente legacy de IOL— deben convertirlo explícitamente a naive (`tzinfo=None`) para continuar siendo compatibles con librerías que no gestionan zonas horarias.

Ambos métodos apuntan al mismo reloj centralizado, por lo que los valores son intercambiables según si necesitas una cadena o un
`datetime` para cálculos adicionales.

Desde Streamlit 1.30 se reemplazó el parámetro `use_container_width` y se realizaron ajustes mínimos de diseño.

### Empresas con oportunidad (beta)

La vista beta evoluciona hacia un universo dinámico que se recalcula en cada sesión combinando:

- Tickers provistos manualmente por el usuario en la interfaz cuando existen; si no hay input manual, se utiliza `YahooFinanceClient.list_symbols_by_markets` parametrizada mediante la variable de entorno `OPPORTUNITIES_TARGET_MARKETS`.
- Un conjunto determinista de respaldo basado en el stub local (`run_screener_stub`) para garantizar resultados cuando no hay configuración externa ni datos remotos, o cuando Yahoo Finance no está disponible.

El ranking final pondera criterios técnicos y fundamentales alineados con los parámetros disponibles en el backend. Los filtros actualmente soportados corresponden a los argumentos `max_payout`, `min_div_streak`, `min_cagr`, `min_market_cap`, `max_pe`, `min_revenue_growth`, `min_eps_growth`, `min_buyback`, `include_latam`, `sectors` e `include_technicals`, combinando métricas de dividendos, valuación, crecimiento y cobertura geográfica.

Cada oportunidad obtiene un **score normalizado en escala 0-100** que promedia aportes de payout, racha de dividendos, CAGR, recompras, RSI y MACD. Esta normalización permite comparar emisores de distintas fuentes con un criterio homogéneo. Los resultados que queden por debajo del umbral configurado se descartan automáticamente para reducir ruido.

Los controles disponibles en la UI permiten ajustar esos filtros sin modificar código:

- Multiselect de sectores para recortar el universo devuelto por la búsqueda.
- Checkbox **Incluir indicadores técnicos** para agregar RSI y medias móviles al resultado.
- Inputs dedicados a crecimiento mínimo de EPS y porcentaje mínimo de recompras (`buybacks`).
- Sliders y number inputs para capitalización, payout, P/E, crecimiento de ingresos, racha/CAGR de dividendos e inclusión de Latinoamérica.

El umbral mínimo de score y el recorte del **top N** de oportunidades son parametrizables mediante las variables `MIN_SCORE_THRESHOLD` (valor por defecto: `80`) y `MAX_RESULTS` (valor por defecto: `20`). La interfaz utiliza ese valor centralizado como punto de partida en el selector "Máximo de resultados" para reflejar cualquier override definido en la configuración. Puedes redefinirlos desde `.env`, `secrets.toml` o `config.json` para adaptar la severidad del filtro o ampliar/restringir el listado mostrado en la UI. La cabecera del listado muestra notas contextuales cuando se aplican estos recortes y sigue diferenciando la procedencia de los datos con un caption que alterna entre `yahoo` y `stub`, manteniendo la trazabilidad de la fuente durante los failovers.


Las notas del listado utilizan iconos para indicar la severidad del mensaje:

- `:warning:` señala datos simulados o problemas de disponibilidad remota.
- `:information_source:` destaca mensajes de escasez o recordatorios operativos.
- Las notas sin prefijo se muestran con formato neutro.

## Integración con Yahoo Finance

La aplicación consulta [Yahoo Finance](https://finance.yahoo.com/) mediante la librería `yfinance` para enriquecer la vista de portafolio con series históricas, indicadores técnicos y métricas fundamentales/ESG. La barra lateral de healthcheck refleja si la última descarga provino de Yahoo o si fue necesario recurrir a un respaldo local, facilitando la observabilidad de esta dependencia externa.

### Indicadores técnicos y fallback local

La función `fetch_with_indicators` descarga OHLCV y calcula indicadores (SMA, EMA, MACD, RSI, Bollinger, ATR, Estocástico e Ichimoku). Los resultados se almacenan en caché durante el intervalo definido por `CACHE_TTL_YF_INDICATORS` (predeterminado: 900 segundos) para evitar llamadas redundantes. Cuando `yfinance` produce un `HTTPError` o `Timeout`, la aplicación recurre automáticamente a `infrastructure/cache/ta_fallback.csv` como stub hasta que el servicio se restablezca.

### Métricas fundamentales y ranking del portafolio

`get_fundamental_data` obtiene valuaciones básicas (PE, P/B, márgenes, ROE, deuda, dividend yield depurado) y respeta el TTL de `CACHE_TTL_YF_FUNDAMENTALS` (6 horas por defecto). Para el ranking consolidado se utiliza `portfolio_fundamentals`, que agrega métricas y puntajes ESG por símbolo y persiste los resultados según `CACHE_TTL_YF_PORTFOLIO_FUNDAMENTALS` (4 horas por defecto). Ambos bloques se muestran en la pestaña principal del portafolio, con mensajes claros cuando los datos no están disponibles.

### Históricos y monitoreo

`get_portfolio_history` construye series ajustadas para todos los símbolos y las conserva durante `CACHE_TTL_YF_HISTORY` (valor inicial: 3600 segundos). El healthcheck `📈 Yahoo Finance` indica si la última consulta provino de la API, de la caché o del stub, junto con detalles del símbolo involucrado.

## Seguridad de credenciales

### 🔒 Seguridad de tus credenciales

- Cifrado de tokens con [Fernet](https://cryptography.io/en/latest/fernet/)
- Almacenamiento de secretos con [Streamlit Secrets](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/secrets-management)
- Tokens guardados en archivos cifrados locales (no en la nube)
- Limpieza inmediata de contraseñas en `session_state`

Tus credenciales nunca se almacenan en servidores externos. El acceso a IOL se realiza de forma segura mediante tokens cifrados, protegidos con clave Fernet y gestionados localmente por la aplicación.

El bloque de login muestra la versión actual de la aplicación con un mensaje como "Estas medidas de seguridad aplican a la versión X.Y.Z".

El sidebar finaliza con un bloque de **Healthcheck (versión X.Y.Z)** que lista el estado de los servicios monitoreados, de modo que puedas validar de un vistazo la disponibilidad de las dependencias clave antes de operar.

## Requisitos de sistema

- Python 3.10 o superior
- `pip` y recomendablemente `venv` o `virtualenv`

## Instalación

1. Clonar el repositorio y crear un entorno virtual (opcional).
2. Instalar dependencias de producción:

```bash
pip install -r requirements.txt
```

Para un entorno de desarrollo con herramientas de linting y pruebas:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Configuración del entorno

Crear un archivo `.env` en la raíz del proyecto con los ajustes necesarios (las credenciales se ingresan desde la interfaz de la aplicación):

```env
USER_AGENT="Portafolio-IOL/1.0"
# Ruta opcional del archivo de tokens
IOL_TOKENS_FILE="tokens_iol.json"
# Clave para cifrar el archivo de tokens (Fernet). Debe definirse en producción
IOL_TOKENS_KEY="..."
# Permite guardar tokens sin cifrar (NO recomendado)
IOL_ALLOW_PLAIN_TOKENS=0
# Otros ajustes opcionales
CACHE_TTL_PORTFOLIO=20
CACHE_TTL_LAST_PRICE=10
CACHE_TTL_QUOTES=8
CACHE_TTL_FX=60
CACHE_TTL_YF_INDICATORS=900
CACHE_TTL_YF_HISTORY=3600
CACHE_TTL_YF_FUNDAMENTALS=21600
CACHE_TTL_YF_PORTFOLIO_FUNDAMENTALS=14400
YAHOO_FUNDAMENTALS_TTL=3600
YAHOO_QUOTES_TTL=300
MIN_SCORE_THRESHOLD=80
MAX_RESULTS=20
ASSET_CATALOG_PATH="/ruta/a/assets_catalog.json"
# Nivel de los logs ("DEBUG", "INFO", etc.; predeterminado: INFO)
LOG_LEVEL="INFO"
# Formato de los logs: "plain" o "json" (predeterminado: plain)
LOG_FORMAT="plain"
# Usuario opcional incluido en los logs
LOG_USER="usuario"
```
Los parámetros `CACHE_TTL_YF_*` ajustan cuánto tiempo se reutiliza cada respuesta de Yahoo Finance antes de volver a consultar la API (indicadores técnicos, históricos, fundamentales individuales y ranking del portafolio, respectivamente). Las variables `YAHOO_FUNDAMENTALS_TTL` (3600 segundos por defecto) y `YAHOO_QUOTES_TTL` (300 segundos por defecto) controlan el TTL de la caché específica para fundamentales y cotizaciones de Yahoo; puedes redefinir estos valores en el `.env` o en `secrets.toml` según tus necesidades. Ambos parámetros también se exponen con alias en minúsculas (`yahoo_fundamentals_ttl` y `yahoo_quotes_ttl`) para facilitar su lectura desde `st.secrets`, y cualquier alias o nombre en mayúsculas puede sobrescribirse indistintamente mediante variables de entorno, archivos `.env` o secretos.

`MIN_SCORE_THRESHOLD` (80 por defecto) define el puntaje mínimo aceptado para que una empresa aparezca en el listado beta, mientras que `MAX_RESULTS` (20 por defecto) determina cuántas filas finales mostrará la UI tras aplicar filtros y ordenar el score normalizado. Ambos valores pueden sobreescribirse desde el mismo `.env`, `secrets.toml` o `config.json` si necesitás afinar la agresividad del recorte.
También puedes definir estos valores sensibles en `secrets.toml`,
el cual `streamlit` expone a través de `st.secrets`. Los valores en
`secrets.toml` tienen prioridad sobre las variables de entorno.

Ejemplo de `.streamlit/secrets.toml`:

```toml
IOL_USERNAME = "tu_usuario"
IOL_PASSWORD = "tu_contraseña"
IOL_TOKENS_KEY = "clave"
IOL_TOKENS_FILE = "tokens_iol.json"
```

`LOG_LEVEL` controla la verbosidad de los mensajes (`DEBUG`, `INFO`, etc.). Evita usar `DEBUG` u otros niveles muy verbosos en producción, ya que pueden revelar información sensible y generar un volumen excesivo de datos. `LOG_FORMAT` puede ser `plain` para un formato legible o `json` para registros estructurados, útil cuando se integran sistemas de logging centralizado o se requiere auditoría. Si `LOG_LEVEL` o `LOG_FORMAT` no están definidos, la aplicación utiliza `INFO` y `plain` como valores por defecto. El valor de `LOG_USER` se incluye en los registros si está definido.

Las credenciales de IOL se utilizan para generar un token de acceso que se guarda en `tokens_iol.json` (o en la ruta indicada por `IOL_TOKENS_FILE`). Si `IOL_TOKENS_KEY` no está configurada y `IOL_ALLOW_PLAIN_TOKENS` no está habilitado, la aplicación registrará un error y se cerrará con código 1 para evitar guardar el archivo sin cifrar. Se puede forzar este comportamiento (solo para entornos de prueba) estableciendo `IOL_ALLOW_PLAIN_TOKENS=1`. Puedes generar una clave con:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Este archivo es sensible: **manténlo fuera del control de versiones** (ya está incluido en `.gitignore`) y con permisos restringidos, por ejemplo `chmod 600`. Si el token expira o se desea forzar una nueva autenticación, borra dicho archivo.

## Ejecución local

```bash
streamlit run app.py
```

## Despliegue

En entornos de producción es obligatorio definir la variable `IOL_TOKENS_KEY` para que el archivo de tokens se almacene cifrado. Si falta y `IOL_ALLOW_PLAIN_TOKENS` no está habilitado, la aplicación registrará el problema y se cerrará.

### Docker

1. Construir la imagen:

```bash
docker build -t portafolio-iol .
```

2. Ejecutar el contenedor (requiere un archivo `.env` con las variables descritas en la sección anterior):

```bash
docker run --env-file .env -p 8501:8501 portafolio-iol
```

La imagen define un `HEALTHCHECK` que consulta `http://localhost:8501/_stcore/health` para comprobar la disponibilidad del servicio durante el despliegue.

Para conservar los tokens generados por la aplicación, se puede montar un volumen:

```bash
mkdir -p tokens
docker run --env-file .env -p 8501:8501 -v $(pwd)/tokens:/app/tokens portafolio-iol
```

Al usar un volumen, define en `.env` la ruta del archivo:

```env
IOL_TOKENS_FILE=/app/tokens/tokens_iol.json
```

### Streamlit Cloud

1. Subir el repositorio a GitHub.
2. En [Streamlit Cloud](https://streamlit.io/cloud), crear una nueva aplicación apuntando a `app.py`.
   La carpeta `.streamlit` ya incluye un `config.toml` con `fileWatcherType = "poll"` para evitar el límite de inotify en la plataforma.
3. En el panel de la aplicación, abre el menú **⋮** y selecciona **Edit secrets** para mostrar la pestaña **Secrets**.
4. Completa el editor con un `secrets.toml` mínimo:
   ```toml
   USER_AGENT = "Portafolio-IOL/1.0"
   IOL_TOKENS_FILE = "tokens_iol.json"
   IOL_TOKENS_KEY = "..."
   IOL_ALLOW_PLAIN_TOKENS = 0
   CACHE_TTL_PORTFOLIO = 20
   CACHE_TTL_LAST_PRICE = 10
   CACHE_TTL_QUOTES = 8
   CACHE_TTL_FX = 60
   CACHE_TTL_YF_INDICATORS = 900
   CACHE_TTL_YF_HISTORY = 3600
   CACHE_TTL_YF_FUNDAMENTALS = 21600
   CACHE_TTL_YF_PORTFOLIO_FUNDAMENTALS = 14400
   ASSET_CATALOG_PATH = "/ruta/a/assets_catalog.json"
   LOG_LEVEL = "INFO"
   LOG_FORMAT = "plain"
   LOG_USER = "usuario"
   ```
   Estos valores coinciden con las variables del ejemplo `.env`.
5. Guarda los cambios y despliega la aplicación.

Para más detalles, consulta la [documentación oficial de Streamlit sobre gestión de secrets](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/secrets-management).

## Post-deploy notes

- **Revisar métricas de performance**:
  - En despliegues locales o Docker, usa `docker stats <container>` para monitorear CPU/memoria y `docker logs <container>` para tiempos de respuesta.
  - En Streamlit Cloud, abre el menú **⋮** y selecciona **View app logs** para ver métricas de la instancia.
- **Prueba de login multiusuario**:
  1. Abre dos navegadores distintos o ventanas en modo incógnito.
  2. Inicia sesión en cada uno con credenciales válidas.
  3. Verifica que cada sesión opere de forma independiente; cerrar sesión en una no debe afectar a la otra.

## Pruebas

Con las dependencias de desarrollo instaladas, ejecutar la suite completa de pruebas:

```bash
pytest
```

Para ejecutar solo un subconjunto por carpeta, indica la ruta deseada:

```bash
pytest application/test
```

Para habilitar las pruebas que consultan Yahoo Finance en vivo, exporta la
variable `RUN_LIVE_YF=1` y ejecuta la etiqueta dedicada. Estas verificaciones
descargan datos reales y, por tratarse de información del mercado en tiempo
real, pueden arrojar resultados no deterministas entre corridas.

```bash
RUN_LIVE_YF=1 pytest -m live_yahoo
```

Para verificar el estilo del código:

```bash
flake8
```

## Tiempos de referencia

Los siguientes tiempos se observan en condiciones normales (aprox. 20 posiciones):

| Paso                | Tiempo objetivo | Detalles |
|---------------------|-----------------|----------|
| `login`             | < 1 s           | `auth_service.login` |
| `fetch_portfolio`   | < 600 ms        | ~20 posiciones |
| `fetch_quotes_bulk` | < 1 s           | 20 símbolos |

Si algún paso supera estos valores, considera reducir llamadas redundantes, ajustar los TTL de cache en `shared/settings.py` o incrementar `MAX_QUOTE_WORKERS` cuando existan muchas posiciones.

## Fallback de análisis técnico

Si ocurre un `HTTPError` o un `Timeout` al descargar datos con `yfinance`,
la función `fetch_with_indicators` recurre al archivo local
`infrastructure/cache/ta_fallback.csv`. Este archivo contiene datos
de respaldo con formato OHLCV utilizados para generar los indicadores.

Para actualizarlo con información reciente, ejecuta el servicio cuando
tengas conexión y guarda el resultado en la misma ruta:

```bash
python - <<'PY'
from application.ta_service import fetch_with_indicators
import pandas as pd
df = fetch_with_indicators('AAPL')  # o el símbolo deseado
df.to_csv('infrastructure/cache/ta_fallback.csv')
PY
```


## Actualización de dependencias

Las versiones de las dependencias están fijadas en `requirements.txt`. Para actualizarlas de forma segura:

```bash
bash scripts/update_dependencies.sh
```

El script actualiza los paquetes a sus últimas versiones, ejecuta las pruebas y, si todo pasa, escribe las nuevas versiones en `requirements.txt`. Este proceso también se ejecuta mensualmente mediante [GitHub Actions](.github/workflows/dependency-update.yml).

## Políticas de sesión y manejo de tokens

Cada sesión de usuario genera un identificador aleatorio almacenado en `st.session_state["session_id"]`, que debe mantenerse constante para aislar los recursos cacheados.

Los tokens de autenticación se guardan en `tokens_iol.json` (o en la ruta indicada por `IOL_TOKENS_FILE`) y deben almacenarse cifrados mediante `IOL_TOKENS_KEY`. Este archivo no debe versionarse y conviene mantenerlo con permisos restringidos (por ejemplo `chmod 600`). Para renovar los tokens:

1. Eliminar el archivo de tokens.
2. Volver a ejecutar la aplicación para que se generen nuevamente.

## Licencia

Este proyecto se distribuye bajo la licencia MIT.
