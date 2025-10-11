# Lista de verificación QA

## Pruebas automatizadas
- [ ] `nox -s lint`
- [ ] `nox -s typecheck`
- [ ] `nox -s tests`
- [ ] `nox -s security` (incluye `bandit` y `pip-audit`)

## Pruebas manuales sugeridas
- [ ] Verificar flujo de login en entorno de staging.
- [ ] Confirmar generación de reportes PDF.
- [ ] Revisar carga de datos de IOL para cuentas activas.

## Pruebas de smoke
- [ ] `streamlit run app.py` y validar carga inicial.
- [ ] Ejecutar consulta de portafolio y confirmar que no hay errores en consola.
- [ ] Validar conexión con API sandbox de IOL.
