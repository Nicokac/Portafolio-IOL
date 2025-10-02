# Guía de pruebas

Esta guía resume los prerequisitos y comandos necesarios para ejecutar la suite completa del
proyecto, incluyendo las verificaciones opcionales que dependen de servicios externos.

## Prerrequisitos

- Python 3.10 o superior.
- Dependencias de producción y QA instaladas:
  ```bash
  pip install -r requirements.txt -r requirements-dev.txt
  ```
- Variables de entorno opcionales para pruebas específicas:
  - `RUN_LIVE_YF=1` habilita los tests que consultan Yahoo Finance en vivo.
  - `FRED_API_KEY` y `FRED_SECTOR_SERIES` permiten ejercitar llamadas reales a FRED en entornos
    donde se desea validar la integración macro. La suite estándar utiliza stubs, por lo que no son
    obligatorias.

## Suite completa

Ejecuta todas las pruebas desde la raíz del repositorio:

```bash
pytest
```

El proyecto incorpora `pytest.ini` con marcadores y configuración de logging. La ejecución completa
usa los stubs deterministas para mantener resultados reproducibles.

Para acotar la ejecución a un paquete o archivo en particular, indica la ruta:

```bash
pytest controllers/test/test_opportunities_controller.py
```

## Pruebas con APIs en vivo

Los tests marcados como `live_yahoo` consultan Yahoo Finance y se consideran opcionales. Para
incluirlos debes exportar la variable y filtrar por el marcador:

```bash
export RUN_LIVE_YF=1
pytest -m live_yahoo
```

Estos escenarios dependen de datos de mercado en tiempo real y pueden variar entre corridas. Se
recomienda ejecutarlos sólo en entornos controlados.

## Flags útiles y depuración

- `pytest -k <expresión>`: corre únicamente los tests cuyo nombre coincide con la expresión.
- `pytest -x`: detiene la ejecución al primer fallo (útil para ciclos cortos).
- `pytest --maxfail=1 --disable-warnings -q`: modo compacto para pipelines CI.
- `pytest --ff`: reejecuta primero los tests fallidos en la corrida previa.

El proyecto utiliza fixtures que escriben artefactos temporales bajo `tmp_path`. Asegúrate de contar
con permisos de escritura en el directorio temporal del sistema.
