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

El stub local expone un universo determinista de 37 emisores que cubre múltiples sectores (Technology, Healthcare, Industrials, Financial Services, Consumer Defensive, Consumer Cyclical, Consumer, Financials, Utilities, Energy, Real Estate, Communication Services y Materials) con métricas fundamentales completas. Cada sector crítico —Technology, Energy, Industrials, Consumer, Healthcare, Financials, Utilities y Materials— cuenta con al menos tres emisores para ejercitar filtros exigentes sin perder diversidad. Las cifras se calibraron para que los filtros de payout, racha, CAGR, EPS, buybacks y fundamentals críticos dispongan siempre de datos consistentes y se puedan ejercitar escenarios complejos de QA aun cuando Yahoo Finance no esté disponible.

| Ticker | Sector | Payout % | Racha (años) | CAGR % | EPS trailing | EPS forward | Buyback % | Market cap (M USD) | P/E | Revenue % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AAPL | Technology | 18.5 | 12 | 14.2 | 6.1 | 6.6 | 1.8 | 2,800,000 | 30.2 | 7.4 |
| MSFT | Technology | 28.3 | 20 | 11.7 | 9.2 | 9.8 | 1.1 | 2,450,000 | 33.5 | 14.8 |
| GOOGL | Communication Services | 0.0 | 0 | 0.0 | 5.2 | 6.1 | 2.3 | 1,750,000 | 27.6 | 9.8 |
| KO | Consumer Defensive | 73.0 | 61 | 7.5 | 2.3 | 2.4 | 0.3 | 260,000 | 24.7 | 4.3 |
| PEP | Consumer Defensive | 68.5 | 51 | 8.9 | 6.9 | 7.3 | 1.5 | 250,000 | 25.4 | 6.2 |
| JNJ | Healthcare | 51.2 | 59 | 6.9 | 8.5 | 8.7 | 0.6 | 415,000 | 21.4 | 3.1 |
| ABBV | Healthcare | 42.3 | 10 | 12.4 | 6.8 | 7.4 | 1.2 | 262,000 | 21.1 | 5.6 |
| NUE | Materials | 33.4 | 49 | 9.8 | 18.4 | 18.6 | 0.0 | 42,000 | 8.9 | -6.2 |
| UNP | Industrials | 45.1 | 16 | 12.2 | 10.5 | 11.2 | 2.9 | 130,000 | 22.5 | 5.1 |
| HON | Industrials | 41.4 | 12 | 9.1 | 8.0 | 8.8 | 1.7 | 130,000 | 24.8 | 3.9 |
| V | Financial Services | 21.6 | 14 | 17.3 | 8.7 | 9.6 | 2.8 | 495,000 | 29.5 | 11.2 |
| JPM | Financial Services | 32.5 | 13 | 9.9 | 13.9 | 14.3 | 1.9 | 440,000 | 10.9 | 8.4 |
| NEE | Utilities | 56.2 | 28 | 10.8 | 3.1 | 3.5 | 0.0 | 160,000 | 25.7 | 7.1 |
| DUK | Utilities | 73.4 | 16 | 5.8 | 5.1 | 5.4 | 0.0 | 73,000 | 18.6 | 2.9 |
| UTLX | Utilities | 61.5 | 19 | 6.7 | 3.1 | 3.3 | 0.0 | 58,600 | 19.2 | 4.6 |
| XOM | Energy | 41.8 | 40 | 4.4 | 10.1 | 9.7 | 3.4 | 460,000 | 11.4 | 9.6 |
| CVX | Energy | 37.2 | 35 | 6.4 | 12.2 | 11.9 | 2.7 | 300,000 | 12.8 | 11.5 |
| PLD | Real Estate | 63.5 | 12 | 9.4 | 3.6 | 3.9 | 0.0 | 115,000 | 28.9 | 8.7 |
| MELI | Consumer Cyclical | 0.0 | 0 | 0.0 | 4.8 | 6.2 | 0.0 | 72,000 | 76.4 | 31.5 |
| BBD | Financial Services | 28.0 | 6 | 7.1 | 1.6 | 1.8 | 1.0 | 47,000 | 9.5 | 12.4 |
| FNCL1 | Financials | 29.4 | 16 | 10.8 | 4.8 | 5.2 | 2.4 | 96,500 | 17.6 | 9.4 |
| FNCL2 | Financials | 34.1 | 12 | 8.7 | 3.5 | 3.8 | 1.6 | 73,400 | 15.8 | 6.3 |
| FNCL3 | Financials | 26.7 | 9 | 11.5 | 5.6 | 6.0 | 2.9 | 128,900 | 18.9 | 11.1 |
| MTRL | Materials | 36.5 | 11 | 8.2 | 4.2 | 4.6 | 1.4 | 68,000 | 19.4 | 6.3 |
| MATX | Materials | 31.8 | 14 | 10.1 | 4.9 | 5.3 | 1.9 | 52,300 | 17.2 | 9.1 |
| CYCX | Consumer Cyclical | 22.5 | 8 | 13.1 | 4.6 | 5.5 | 2.5 | 78,000 | 27.1 | 15.7 |
| RSPR | Real Estate | 70.2 | 9 | 7.4 | 2.9 | 3.2 | 0.0 | 32,000 | 18.9 | 5.2 |
| ENRGX | Energy | 38.7 | 18 | 5.6 | 5.5 | 5.8 | 1.9 | 95,000 | 13.6 | 8.9 |
| SOLR | Energy | 24.1 | 5 | 16.8 | 1.5 | 2.1 | 0.5 | 26,000 | 35.2 | 22.4 |
| LATC | Consumer Cyclical | 31.7 | 7 | 9.9 | 2.0 | 2.3 | 1.3 | 18,500 | 17.8 | 12.1 |
| CNMR1 | Consumer | 48.6 | 15 | 10.2 | 3.1 | 3.4 | 0.9 | 42,500 | 19.6 | 7.9 |
| CNMR2 | Consumer | 36.9 | 11 | 12.6 | 3.9 | 4.4 | 1.7 | 58,200 | 22.4 | 10.5 |
| CNMR3 | Consumer | 41.2 | 13 | 9.4 | 2.8 | 3.1 | 1.0 | 37,800 | 18.9 | 6.7 |
| FNSH | Consumer Defensive | 55.4 | 14 | 6.1 | 3.0 | 3.3 | 0.9 | 54,000 | 20.3 | 4.7 |
| INFR | Industrials | 29.8 | 11 | 10.4 | 4.9 | 5.4 | 2.2 | 67,000 | 21.7 | 9.5 |
| DATA | Technology | 15.2 | 4 | 18.7 | 3.8 | 4.9 | 1.6 | 125,000 | 38.1 | 24.6 |
| HLTH | Healthcare | 34.9 | 9 | 11.4 | 3.9 | 4.4 | 1.1 | 58,000 | 23.4 | 8.8 |

Cada registro respeta principios de la estrategia Andy: payout y P/E en rangos saludables, rachas y CAGR positivos, EPS forward superiores al trailing, buybacks y crecimiento de ingresos presentes cuando corresponde. El dataset se utiliza tanto para fallback como para pruebas end-to-end, garantizando que la aplicación conserve diversidad sectorial, métricas completas y comportamiento determinista durante los failovers.

Durante los failovers la UI etiqueta el origen como `stub` y continúa respetando los filtros configurados. Los tests automatizados utilizan este dataset extendido para comprobar diversidad sectorial y completitud de fundamentals, por lo que cualquier ajuste debe mantener la cobertura y las columnas documentadas.

#### Telemetría del barrido

Además de la etiqueta, la UI muestra una nota informativa con la telemetría del barrido cuando se usa el stub o Yahoo Finance. El helper `shared.ui.notes.format_note` renderiza este mensaje con severidad `ℹ️` para que destaque sin generar alertas falsas mientras todo se encuentre dentro de los parámetros esperados. Ejemplos típicos:

```
ℹ️ Yahoo procesó 128 símbolos • elapsed: 5.8 s • discarded: 12% fundamentals / 6% técnicos
ℹ️ Stub sweep • elapsed: 2.4 s • universe: 37 tickers • discarded: 18% fundamentals / 10% técnicos
```

Cuando el sistema detecta anomalías (por ejemplo, universo < 10 o ratios > 35 % durante varias corridas), la misma nota escala a severidad `⚠️` para llamar la atención del operador.

- **Elapsed/elapsed time:** duración total del barrido, útil para detectar degradaciones repentinas (si sube por encima de ~3 s en el stub o supera los 8-9 s en Yahoo, suele indicar latencias externas o tareas adicionales).
- **Universe/universe size:** cantidad de símbolos analizados en la corrida actual; cambios abruptos respecto al universo habitual (37 para el stub o el número que devuelva Yahoo según `OPPORTUNITIES_TARGET_MARKETS`) señalan filtros mal configurados o fallos de descarga.
- **Discarded/discard ratios:** porcentaje de candidatos eliminados por falta de fundamentals o de señales técnicas; valores sostenidos por encima del 25 % ameritan revisar la fuente de datos o los umbrales configurados.

| Métrica | Cómo interpretarla | Severidades posibles |
| --- | --- | --- |
| `elapsed` / `elapsed time` | Duración del barrido. Valores estables indican salud; picos puntuales pueden deberse a IO o throttling. | `ℹ️` cuando está dentro de los rangos esperados; `⚠️` si excede los umbrales configurados (p. ej., >3 s en stub o >9 s en Yahoo).
| `universe` / `symbols processed` | Cantidad de tickers evaluados en la corrida. Debe mantenerse alineada con el origen (37 en stub, `YahooFinanceClient` o input manual). | `ℹ️` mientras se mantenga estable; `⚠️` si cae de forma abrupta (universo < 10, vacíos inesperados).
| `discarded fundamentals/tech` | Ratios de descarte por falta de fundamentals o señales técnicas. | `ℹ️` cuando los descartes se mantienen <25 %; `⚠️` si superan ese valor de forma consistente.

Las notas siempre incluyen tanto los porcentajes de descarte fundamental como técnico; cuando alguno de los dos no aplica, el stub reporta explícitamente `0%` para preservar la consistencia del formato y evitar falsos positivos en los tests automatizados.

El ranking final pondera criterios técnicos y fundamentales alineados con los parámetros disponibles en el backend. Los filtros actualmente soportados corresponden a los argumentos `max_payout`, `min_div_streak`, `min_cagr`, `min_market_cap`, `max_pe`, `min_revenue_growth`, `min_eps_growth`, `min_buyback`, `include_latam`, `sectors` e `include_technicals`, combinando métricas de dividendos, valuación, crecimiento y cobertura geográfica.

Cada oportunidad obtiene un **score normalizado en escala 0-100** que promedia aportes de payout, racha de dividendos, CAGR, recompras, RSI y MACD. Esta normalización permite comparar emisores de distintas fuentes con un criterio homogéneo. Los resultados que queden por debajo del umbral configurado se descartan automáticamente para reducir ruido.

Los controles disponibles en la UI permiten ajustar esos filtros sin modificar código, y la interfaz incluye un glosario interactivo [¿Qué significa cada métrica?](#qué-significa-cada-métrica) con ejemplos numéricos para alinear la interpretación de payout, EPS, CAGR, buybacks y score entre la documentación y la aplicación:

- Multiselect de sectores para recortar el universo devuelto por la búsqueda.
- Checkbox **Incluir indicadores técnicos** para agregar RSI y medias móviles al resultado.
- Inputs dedicados a crecimiento mínimo de EPS y porcentaje mínimo de recompras (`buybacks`).
- Sliders y number inputs para capitalización, payout, P/E, crecimiento de ingresos, racha/CAGR de dividendos e inclusión de Latinoamérica.
- Selector **Perfil recomendado** para aplicar presets preconfigurados según el tipo de oportunidad que se quiera priorizar:
  - **Dividendos defensivos**: favorece emisores consolidados con payout moderado, más de 10 años de dividendos, crecimiento estable y foco en sectores defensivos (``Consumer Defensive`` y ``Utilities``).
  - **Crecimiento balanceado**: combina expansión de ingresos/EPS de dos dígitos con payout controlado y sesgo hacia ``Technology`` y ``Healthcare`` para captar historias de crecimiento rentable.
  - **Recompras agresivas**: apunta a compañías con recompras netas relevantes, valuaciones razonables e inclusión de indicadores técnicos para reforzar el timing, con foco en ``Financial Services``, ``Technology`` e ``Industrials``.

El umbral mínimo de score y el recorte del **top N** de oportunidades son parametrizables mediante las variables `MIN_SCORE_THRESHOLD` (valor por defecto: `80`) y `MAX_RESULTS` (valor por defecto: `20`). La interfaz utiliza ese valor centralizado como punto de partida en el selector "Máximo de resultados" para reflejar cualquier override definido en la configuración. Puedes redefinirlos desde `.env`, `secrets.toml` o `config.json` para adaptar la severidad del filtro o ampliar/restringir el listado mostrado en la UI. La cabecera del listado muestra notas contextuales cuando se aplican estos recortes y sigue diferenciando la procedencia de los datos con un caption que alterna entre `yahoo` y `stub`, manteniendo la trazabilidad de la fuente durante los failovers.


#### ¿Qué significa cada métrica?

- **Payout:** porcentaje de las ganancias que se reparte como dividendo. Ejemplo: con un payout del 60 %, una empresa distribuye US$0,60 por cada dólar de utilidad.
- **EPS (Earnings Per Share):** ganancias por acción. Si una firma genera US$5 millones y tiene 1 millón de acciones, su EPS es US$5.
- **Crecimiento de ingresos:** variación interanual de ventas. Un aumento de US$100 a US$112 implica un crecimiento del 12 %.
- **Racha de dividendos:** cantidad de años consecutivos pagando dividendos. Una racha de 7 significa pagos sin interrupciones durante siete ejercicios.
- **CAGR de dividendos:** crecimiento anual compuesto del dividendo. Pasar de US$1 a US$1,50 en cinco años implica un CAGR cercano al 8 %.
- **Buybacks:** recompras netas que reducen el flotante. Un buyback del 2 % indica que la empresa retiró 2 de cada 100 acciones en circulación.
- **Score compuesto:** puntaje de 0 a 100 que combina valuación, crecimiento, dividendos y técnicos; por ejemplo, un score de 85 señala atributos superiores al umbral típico de 80.


### Notas del listado y severidades

Las notas del listado utilizan una clasificación estandarizada para transmitir la severidad del mensaje. Cada nivel comparte prefijos visibles en el texto bruto (útiles en pruebas o fixtures) y un icono renderizado al pasar por `shared.ui.notes.format_note`:

| Severidad | Prefijos esperados | Icono renderizado | Uso típico |
| --- | --- | --- | --- |
| `warning` | `⚠️` | `:warning:` | Avisar que los datos provienen de un stub, que el universo está recortado o que hubo fallbacks. |
| `info` | `ℹ️` | `:information_source:` | Recordatorios operativos o mensajes neutrales relacionados con disponibilidad de datos. |
| `success` | `✅` | `:white_check_mark:` | Confirmar procesos completados o resultados satisfactorios. |
| `error` | `❌` | `:x:` | Indicar fallas irrecuperables que el usuario debe revisar. |

Siempre que sea posible prefija el contenido con el emoji correspondiente para que el helper lo clasifique correctamente. El siguiente ejemplo mínimo muestra cómo centralizar el formato en la UI:

```python
from shared.ui.notes import format_note

format_note("⚠️ Solo se encontraron 3 tickers con datos recientes.")
# ":warning: **Solo se encontraron 3 tickers con datos recientes.**"
```

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

El bloque de login muestra la versión actual de la aplicación con un mensaje como "Estas medidas de seguridad aplican a la versión 0.3.16".

El sidebar finaliza con un bloque de **Healthcheck (versión 0.3.16)** que lista el estado de los servicios monitoreados, de modo que puedas validar de un vistazo la disponibilidad de las dependencias clave antes de operar.

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

### Job opcional de Yahoo Finance en CI

El workflow `CI` incluye un job opcional que ejecuta `pytest -m live_yahoo`
con `RUN_LIVE_YF=1` para validar la integración real con Yahoo Finance.
Este smoke-test **no se ejecuta automáticamente** porque consume datos en
tiempo real y los resultados pueden variar entre corridas. Para activarlo:

1. Ingresa a **Actions → CI → Run workflow**.
2. Habilita el toggle **Run live Yahoo Finance smoke-test**.
3. Ejecuta el workflow manualmente.

Al hacerlo, GitHub Actions exportará `RUN_LIVE_YF=1` antes de invocar el
marcador `live_yahoo`. Usa este job sólo cuando necesites verificar la
integración en vivo o validar incidentes relacionados con Yahoo Finance.

### Barrido prolongado del stub y presets en CI

Adicionalmente, el workflow programa un job nocturno `stub-fallback-sweep`
que se ejecuta todos los días a las **03:00 UTC** para ejercitar el stub y
los presets recomendados. Este barrido ejecuta la batería prolongada de
pruebas de fallback sobre el stub (incluye validaciones de notas, presets y
consistencia entre corridas) y registra métricas de duración junto con los
totales de `passed/failed/errors/skipped`.

Para detonarlo manualmente:

1. Ingresa a **Actions → CI → Run workflow**.
2. Habilita el toggle **Run prolonged stub fallback & preset sweep**.
3. Ejecuta el workflow.

Al finalizar, revisa el resumen del job en GitHub Actions o descarga el
artefacto `stub-sweep-logs`, que incluye `stub_sweep.log` y
`stub_sweep_metrics.json` con las métricas necesarias para seguimiento de QA. Allí se registran el `elapsed_time`, el tamaño del universo evaluado y los porcentajes de descartes de fundamentals/técnicos que muestra la nota de telemetría (ver [guía de interpretación](#telemetria-del-barrido) para detalles). En los monitoreos nocturnos consideramos saludable que el stub termine en menos de 3 segundos, que el universo se mantenga estable (37 símbolos) y que las tasas de descarte se mantengan por debajo del 25 %; desvíos persistentes disparan revisiones manuales o ajustes en los presets.

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
