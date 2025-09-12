# Portafolio IOL

Aplicación Streamlit para consultar y analizar carteras de inversión en IOL.

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
ASSET_CATALOG_PATH="/ruta/a/assets_catalog.json"
LOG_LEVEL="INFO"
# Formato JSON opcional para logs
LOG_JSON=0
# Usuario opcional incluido en los logs
LOG_USER="usuario"
```

`LOG_LEVEL` controla la verbosidad de los mensajes (`DEBUG`, `INFO`, etc.). Si se establece `LOG_JSON=1`, los registros se emitirán en formato JSON e incluirán el nombre del módulo y el valor de `LOG_USER` si está definido.

Las credenciales de IOL se utilizan para generar un token de acceso que se guarda en `tokens_iol.json` (o en la ruta indicada por `IOL_TOKENS_FILE`). Si `IOL_TOKENS_KEY` no está configurada, la aplicación se detendrá para evitar guardar el archivo en texto plano. Se puede forzar este comportamiento (solo para entornos de prueba) estableciendo `IOL_ALLOW_PLAIN_TOKENS=1`. Puedes generar una clave con:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Este archivo es sensible: **manténlo fuera del control de versiones** (ya está incluido en `.gitignore`) y con permisos restringidos, por ejemplo `chmod 600`. Si el token expira o se desea forzar una nueva autenticación, borra dicho archivo.

## Ejecución local

```bash
streamlit run app.py
```

### Identificador de sesión

Al comenzar una nueva interacción, la aplicación genera un ID aleatorio y lo
guarda en `st.session_state["session_id"]`. Este identificador debe
mantenerse sin cambios durante toda la sesión de un usuario para garantizar el
aislamiento de los recursos cacheados.

## Despliegue

En entornos de producción es obligatorio definir la variable `IOL_TOKENS_KEY` para que el archivo de tokens se almacene cifrado. Si falta, la inicialización fallará salvo que se active `IOL_ALLOW_PLAIN_TOKENS`, lo cual no es recomendable.

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
3. Definir en *Secrets* las variables de entorno descritas en `.env`.

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

## Mantenimiento de tokens

El token de autenticación se guarda en `tokens_iol.json` (cifrado si se configuró `IOL_TOKENS_KEY`). Este archivo no debe versionarse y debe mantenerse con permisos restringidos (por ejemplo `chmod 600`). Para renovar los tokens:

1. Eliminar el archivo `tokens_iol.json` o el indicado por `IOL_TOKENS_FILE`.
2. Volver a ejecutar la aplicación para que se generen nuevamente.

## Licencia

Este proyecto se distribuye bajo la licencia MIT.
