# Portafolio IOL

Aplicación Streamlit para consultar y analizar carteras de inversión en IOL.

## Requisitos de sistema

- Python 3.10 o superior
- `pip` y recomendablemente `venv` o `virtualenv`

## Instalación

1. Clonar el repositorio y crear un entorno virtual (opcional).
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Configuración del entorno

Crear un archivo `.env` en la raíz del proyecto con las credenciales y ajustes necesarios:

```env
IOL_USERNAME="usuario"
IOL_PASSWORD="secreto"
USER_AGENT="Portafolio-IOL/1.0"
# Ruta opcional del archivo de tokens
IOL_TOKENS_FILE="tokens_iol.json"
# Otros ajustes opcionales
CACHE_TTL_PORTFOLIO=20
CACHE_TTL_LAST_PRICE=10
ASSET_CATALOG_PATH="/ruta/a/assets_catalog.json"
```

Las credenciales de IOL se utilizan para generar un token de acceso que se guarda en `tokens_iol.json` (o en la ruta indicada por `IOL_TOKENS_FILE`). Si el token expira o se desea forzar una nueva autenticación, borrar dicho archivo.

## Ejecución local

```bash
streamlit run app.py
```

## Despliegue

### Docker

1. Construir la imagen:

```bash
docker build -t portafolio-iol .
```

2. Ejecutar el contenedor:

```bash
docker run --env-file .env -p 8501:8501 portafolio-iol
```

### Streamlit Cloud

1. Subir el repositorio a GitHub.
2. En [Streamlit Cloud](https://streamlit.io/cloud), crear una nueva aplicación apuntando a `app.py`.
3. Definir en *Secrets* las variables de entorno descritas en `.env`.

## Pruebas

Ejecutar la suite de pruebas automatizadas:

```bash
pytest application/test
```

## Mantenimiento de tokens

El token de autenticación se guarda en `tokens_iol.json`. Para renovar los tokens:

1. Eliminar el archivo `tokens_iol.json` o el indicado por `IOL_TOKENS_FILE`.
2. Volver a ejecutar la aplicación para que se generen nuevamente.

## Licencia

Este proyecto se distribuye bajo la licencia MIT.
