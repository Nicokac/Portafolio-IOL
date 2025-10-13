# SQLite maintenance jobs

A partir de `v0.6.6-patch3` se ejecuta un proceso automático que mantiene las
bases SQLite utilizadas por `MarketDataCache` y `performance_store`. El
objetivo es mantener bajo control el tamaño de los archivos, eliminar datos
expirados y exponer métricas operativas.

## Qué hace el mantenimiento

* **Poda por TTL**
  * `MarketDataCache`: elimina filas cuyo `expires_at` ya pasó.
  * `performance_store`: borra métricas con antigüedad mayor a la ventana de
    retención configurada.
* **VACUUM**: compacta la base para recuperar espacio en disco.
* **Métricas Prometheus** (si están habilitadas):
  * `cache_cleanup_total{database,reason}`: filas eliminadas por ciclo.
  * `vacuum_duration_seconds{database,reason}`: duración de cada VACUUM.
* **Logging**: cada ejecución informa tamaño antes/después, filas removidas y
  duración del VACUUM. Si un archivo supera el umbral configurado se emite una
  advertencia previa y posterior al ciclo.

## Programación y disparadores

El scheduler corre en un thread en segundo plano y puede dispararse por dos
motivos:

1. Se supera el intervalo configurado (`SQLITE_MAINTENANCE_INTERVAL_HOURS`).
2. Alguna base excede el tamaño máximo permitido (`SQLITE_MAINTENANCE_SIZE_THRESHOLD_MB`).

Cuando ambos valores son cero se considera deshabilitado y no se inicia el
thread. Cada iteración consulta ambas condiciones y ejecuta el ciclo una sola
vez por disparo.

## Configuración

| Variable | Descripción | Default |
| --- | --- | --- |
| `SQLITE_MAINTENANCE_INTERVAL_HOURS` | Intervalo entre ejecuciones automáticas. | `6.0` |
| `SQLITE_MAINTENANCE_SIZE_THRESHOLD_MB` | Tamaño máximo antes de disparar una corrida inmediata. | `256.0` |
| `PERFORMANCE_STORE_TTL_DAYS` | Ventana de retención para `performance_store`. | Igual a `LOG_RETENTION_DAYS` |
| `PERFORMANCE_DB_PATH` | Ubicación del archivo de métricas de performance. | `logs/performance/performance_metrics.db` |
| `MARKET_DATA_CACHE_BACKEND` / `MARKET_DATA_CACHE_PATH` | Deben apuntar a `sqlite` para que el ciclo incluya `MarketDataCache`. | `sqlite` / `data/market_cache.db` |

> **Nota:** Cambiar cualquiera de estas variables requiere reiniciar la app o
> el backend para que el scheduler vuelva a inicializarse con la nueva
> configuración.

## Ejecución manual

Es posible forzar un ciclo sin esperar al scheduler:

```bash
python -c "from services.maintenance import run_sqlite_maintenance_now; print(run_sqlite_maintenance_now(reason='manual'))"
```

El resultado es una lista de diccionarios con los tamaños (en bytes), filas
borradas y duración del VACUUM por base.

## Notas de implementación

Desde `v0.6.6-patch3b` los imports del scheduler se resuelven de forma diferida
para evitar dependencias circulares al inicializar `app.py` o `api/main.py`.
Los ajustes mantienen la inicialización automática del thread y las métricas de
Prometheus, pero ahora el módulo sólo crea el scheduler cuando alguna función
lo solicita.

## Buenas prácticas

* Revisá los logs `sqlite-maintenance` al desplegar nuevas versiones para
  asegurar que los tamaños se mantienen dentro del umbral.
* Supervisá los dashboards de Prometheus/Grafana para detectar aumentos
  progresivos en `cache_cleanup_total` o duraciones de VACUUM.
* Ajustá los TTLs de caché o la retención de performance si la cantidad de
  eliminaciones es muy alta en cada ciclo.
