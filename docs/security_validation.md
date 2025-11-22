# Validación de variables de seguridad

La aplicación requiere dos claves Fernet distintas para cifrar tokens internos:

- `FASTAPI_TOKENS_KEY`
- `IOL_TOKENS_KEY`

Generá claves nuevas con:

```bash
python generate_key.py
```

### Entornos de desarrollo vs. producción

- **Producción (`APP_ENV=prod`)**: si falta alguna clave o es inválida, el arranque se bloquea para evitar configuraciones inseguras.
- **Desarrollo (`APP_ENV=dev`, `development` o `local`)**: cuando faltan las claves, el validador degrada a un modo relajado, registra una advertencia y permite continuar para facilitar el trabajo local. Configurá las claves cuanto antes para recuperar el cifrado.

Definí siempre `APP_ENV` para que el validador aplique la política correcta.
