# Operaciones y Observabilidad

Este documento describe el flujo operativo para monitorear y mantener la
aplicación de Portafolio IOL con foco en la visibilidad expuesta en el panel
**"🔍 Estado del Sistema"**.

## Panel de estado

La UI de Streamlit ofrece un panel dedicado con las siguientes secciones:

* **Performance:** métricas Prometheus de tiempos de ejecución, latencia y
  volumen de solicitudes.
* **Seguridad:** estado del token de autenticación emitido por la UI y contadores
  de refrescos, fallas y revocaciones.
* **Caché:** indicadores de eficiencia del caché predictivo, incluyendo hit ratio
  y eventos de invalidación.

Se recomienda revisar periódicamente los indicadores clave expuestos en los
cards superiores del panel:

| Indicador | Descripción |
| --- | --- |
| **Uptime** | Tiempo transcurrido desde el último arranque del backend instrumentado por Prometheus. |
| **Refresh tokens** | Cantidad acumulada de renovaciones exitosas del token backend. |
| **Hit ratio caché** | Porcentaje de aciertos sobre el caché predictivo. Valores menores al 70 % requieren investigación. |

## Gestión del token de autenticación

La UI emite un token Fernet con TTL máximo configurado por `FASTAPI_AUTH_TTL`
(por defecto 15 minutos). El panel muestra:

* Usuario asociado (con ofuscación básica).
* Timestamp de emisión y expiración calculados con la zona horaria
  `America/Argentina/Buenos_Aires`.
* Tiempo restante de vida (TTL restante).

### Refresh manual

Usá el botón **"🔄 Refrescar token"** cuando:

1. Estés ejecutando workflows prolongados y desees evitar la expiración del
   token en medio del proceso.
2. Detectes advertencias de proximidad a la expiración en los logs o en el
   backend.

Al refrescar de manera manual se emite un nuevo token y se actualiza el
timestamp visible en la UI. Si el refresh falla, consultá los logs del backend
(`analysis.log`) o el historial de observabilidad en la sección de performance.

## Alertas y troubleshooting

* **Latencias elevadas:** el tablero de performance incluye percentiles y
  filtros por bloque para identificar cuellos de botella. Revisá los registros
  exportables (`performance_metrics.csv`/`.json`).
* **Cache hit ratio bajo:** verificá que los jobs de precarga estén activos y
  que la caducidad (`TTL`) sea acorde. Revisá `docs/cache_management.md` para
  estrategias de warmup.
* **Errores de autenticación:** inspeccioná los contadores `auth_*` y
  confirmá el estado del token. En caso de revocación manual, generá un nuevo
  token desde la pantalla de login.

Consultá también `docs/troubleshooting.md` para guías específicas de resolución
de incidentes.

