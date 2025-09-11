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
# Clave opcional para cifrar el archivo de tokens (Fernet)
IOL_TOKENS_KEY=""
# Otros ajustes opcionales
CACHE_TTL_PORTFOLIO=20
CACHE_TTL_LAST_PRICE=10
ASSET_CATALOG_PATH="/ruta/a/assets_catalog.json"
```

Las credenciales de IOL se utilizan para generar un token de acceso que se guarda en `tokens_iol.json` (o en la ruta indicada por `IOL_TOKENS_FILE`). Si `IOL_TOKENS_KEY` está definido, el archivo se cifra mediante [Fernet](https://cryptography.io/en/latest/fernet/) con esa clave. Puedes generar una clave con:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Este archivo es sensible: **manténlo fuera del control de versiones** (ya está incluido en `.gitignore`) y con permisos restringidos, por ejemplo `chmod 600`. Si el token expira o se desea forzar una nueva autenticación, borra dicho archivo.

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
