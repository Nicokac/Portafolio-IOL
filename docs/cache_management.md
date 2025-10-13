# Guía operativa de la gestión de caché

Esta guía describe cómo operar y monitorear los endpoints de administración del caché de mercado expuestos por la API. Se incluyen ejemplos prácticos y recomendaciones para entornos productivos.

## Endpoints disponibles

### `GET /cache/status`
Entrega un resumen consolidado del caché en memoria y persistente. El payload devuelve la cantidad total de entradas, ratio de aciertos, TTL promedio (en segundos) y el tamaño estimado en MB. Requiere autenticación Bearer.

### `POST /cache/invalidate`
Permite invalidar entradas del caché utilizando un patrón glob (`pattern`) o una lista explícita de claves (`keys`). La operación devuelve cuántos registros fueron eliminados y el tiempo total invertido. Requiere autenticación Bearer.

### `POST /cache/cleanup`
Limpia registros expirados y entradas huérfanas detectadas en los backends in-memory y SQLite. Retorna cuántos registros fueron removidos por categoría y la duración de la tarea. Requiere autenticación Bearer.

## Ejemplos de uso

> ℹ️ Generá un token Fernet válido desde la UI o con `python -m services.auth generate-token`. Asignalo a la variable `TOKEN` antes de ejecutar los ejemplos.

### Via `curl`

```bash
# Estado del caché (autenticado)
curl \
  -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/cache/status

# Invalidación por patrón
data='{ "pattern": "TECH_*" }'
curl \
  -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "$data" \
  http://localhost:8000/cache/invalidate

# Limpieza manual
curl \
  -X POST \
  -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/cache/cleanup
```

### Via `requests` en Python

```python
import os
import requests

BASE_URL = os.getenv("CACHE_API_URL", "http://localhost:8000")
TOKEN = os.environ["TOKEN"]
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

status = requests.get(f"{BASE_URL}/cache/status", headers=HEADERS, timeout=10)
status.raise_for_status()
print(status.json())

invalidate_payload = {"keys": ["GGAL", "YPFD"]}
invalidated = requests.post(
    f"{BASE_URL}/cache/invalidate",
    headers={**HEADERS, "Content-Type": "application/json"},
    json=invalidate_payload,
    timeout=10,
)
invalidated.raise_for_status()
print(invalidated.json())

cleanup = requests.post(f"{BASE_URL}/cache/cleanup", headers=HEADERS, timeout=10)
cleanup.raise_for_status()
print(cleanup.json())
```

## Recomendaciones para producción

- **Ciclos de limpieza manual:** programá limpiezas programadas (por ejemplo cada 6 horas) para evitar acumulación de registros expirados en SQLite cuando el tráfico baja.
- **TTL sugeridos:** mantené `default_ttl` entre 120 y 360 minutos para históricos y fundamentales; ajustá a 6 horas (`21600 s`) para predicciones si tu instancia sirve reportes recurrentes.
- **Monitoreo:** exportá las métricas Prometheus `cache_*` y `cache_operation_duration_seconds` para detectar degradaciones. Añadí alarmas sobre ratio de errores y tiempos promedio por operación.
- **Backups del backend persistente:** verificá periódicamente el tamaño de `market_data_cache.sqlite` y respalda antes de upgrades mayores.

## Advertencias y límites

- **Máximo de claves:** la lista `keys` admite hasta `500` entradas por solicitud. El endpoint devolverá `400 Bad Request` si se excede el límite (`"max keys exceeded"`).
- **Patrones válidos:** los patrones vacíos o compuestos sólo por espacios se rechazan con `400` y mensaje `"invalid pattern"`.
- **Autenticación obligatoria:** cualquier request sin cabecera `Authorization` válida obtiene `401 Unauthorized`.
- **Errores comunes:**
  - `El servicio de caché no está disponible` (`500`): backend caído o sin inicializar.
  - `Timeout ...` (`500`): la operación superó el tiempo de espera máximo.
  - `Error inesperado ...` (`500`): revisar logs estructurados (`logs/performance/structured.log`) para detalles.
- **Plan de contingencia:** ante bloqueos de SQLite (`database is locked`), ejecutá `POST /cache/cleanup` tras liberar procesos y validá que los jobs batch no compartan el mismo archivo.
