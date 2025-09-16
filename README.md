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

Ambos métodos apuntan al mismo reloj centralizado, por lo que los valores son intercambiables según si necesitas una cadena o un
`datetime` para cálculos adicionales.

Desde Streamlit 1.30 se reemplazó el parámetro `use_container_width` y se realizaron ajustes mínimos de diseño.

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
ASSET_CATALOG_PATH="/ruta/a/assets_catalog.json"
# Nivel de los logs ("DEBUG", "INFO", etc.; predeterminado: INFO)
LOG_LEVEL="INFO"
# Formato de los logs: "plain" o "json" (predeterminado: plain)
LOG_FORMAT="plain"
# Usuario opcional incluido en los logs
LOG_USER="usuario"
```
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
