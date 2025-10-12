# Validación automática de claves de seguridad

Este proyecto valida automáticamente las claves utilizadas para cifrar tokens antes de iniciar la UI de Streamlit o el backend de FastAPI. El proceso cubre tanto los entornos locales como el pipeline de CI/CD.

## Variables monitoreadas

- `FASTAPI_TOKENS_KEY`
- `IOL_TOKENS_KEY`

Ambas deben contener claves Fernet válidas (32 bytes codificados en base64) y no pueden compartir el mismo valor.

## Reglas de validación

1. **Presencia obligatoria**: si falta alguna clave la aplicación se detiene con un error descriptivo.
2. **Formato**: cada clave debe representarse en base64 y, una vez decodificada, medir exactamente 32 bytes.
3. **Claves distintas**: se rechaza la reutilización de la misma clave para ambos servicios.
4. **Entropía mínima**: en `APP_ENV=prod` se emiten *warnings* cuando una clave decodificada posee muy pocos bytes únicos (indicio de baja entropía).

## Ejecución local

La validación ocurre automáticamente al importar `app.py` o `api/main.py`. Para verificar manualmente las claves podés ejecutar:

```bash
python -m shared.security_env_validator
```

## Pipeline de CI/CD

El workflow [`validate_secrets.yml`](../.github/workflows/validate_secrets.yml) ejecuta la validación antes del deploy. Configurá los secretos `FASTAPI_TOKENS_KEY` e `IOL_TOKENS_KEY` en el repositorio o en el entorno de la organización para que la verificación se ejecute correctamente.

## Generación de claves nuevas

Utilizá el script incluido para obtener claves Fernet válidas:

```bash
python generate_key.py
```

Repetí el proceso para generar dos claves distintas y guardalas en tu administrador de secretos antes de lanzar el pipeline.
