# Operaciones y Observabilidad

Este documento describe el flujo operativo para monitorear y mantener la
aplicación de Portafolio IOL con foco en la visibilidad expuesta en el panel
**"🔍 Estado del Sistema"** y los diagnósticos automáticos.

## Monitoreo de rendimiento

Un job en segundo plano ejecuta benchmarks periódicos sobre los endpoints
críticos (`/predictive_compute`, `/quotes_refresh`, `/apply_filters` y otros
instrumentados) y registra los resultados en ``logs/system_diagnostics.log``.
Cada ciclo calcula el promedio de latencia reciente y lo compara con la media
histórica. Si la métrica actual duplica (o supera 2×) la media previa, el panel
marca la condición como degradada.

Además del promedio de latencias, el snapshot incluye:

* Estado del caché predictivo (hits, misses, hit ratio, TTL efectivo y TTL
  restante).
* Validación de claves Fernet obligatorias (`FASTAPI_TOKENS_KEY` e
  `IOL_TOKENS_KEY`) con un *fingerprint* seguro para identificar cambios.
* Información del entorno de ejecución (APP_ENV, zona horaria, versión de
  Python y plataforma).

Para acceder a estos datos desde la UI abrí **"🔎 Diagnóstico del sistema"** en
la barra lateral. El panel muestra las latencias promedio, posibles
degradaciones y un resumen del caché y las claves Fernet. El archivo de log
permite auditar los ciclos históricos o integrarlo a pipelines externos.

### Métrica `ui_total_load_ms` (v0.6.6-patch10b)

La versión `v0.6.6-patch10b` alinea la visibilidad del tiempo total de carga de
la UI en los tres canales operativos principales:

* **Panel de Streamlit:** la tabla de *Métricas instrumentadas* muestra
  `total_load_ms`, alimentada directamente por `st.session_state`. Útil para
  validar mejoras sin abandonar la vista de diagnóstico.
* **Endpoint `/metrics`:** el backend expone un gauge `ui_total_load_ms` dentro
  del registro Prometheus compartido. El valor se actualiza al finalizar cada
  render exitoso y publica `NaN` en ejecuciones headless donde la sesión de UI no
  existe.
* **logs/app_startup.log:** se agrega una línea JSON con los campos
  `{metric, value_ms, version, timestamp}` al completarse la primera renderización.
  Esto permite correlacionar startups lentos con despliegues o migraciones.

Objetivos de operación sugeridos:

* **< 10 000 ms:** escenario nominal.
* **10 000–15 000 ms:** advertencia, revisar latencias de dependencias.
* **> 15 000 ms:** crítico, disparar alerta y escalar al equipo de backend.

### Secuencia de arranque y métrica `ui_startup_load_ms`

El arranque inicial sigue un orden estricto para maximizar el *time-to*
*interactive* del login:

1. **Validación de seguridad:** `shared.security_env_validator.validate_security_environment`
   corre una única vez, marca `_security_validated` en `st.session_state` y detiene la
   ejecución ante claves inválidas.
2. **Preload asincrónico:** `services.preload_worker.start_preload_worker` inicia un
   hilo *daemon* que importa `pandas`, `plotly` y `statsmodels` mientras la UI sigue
   respondiendo.
3. **Login interactivo:** `ui.login.render_login_page` renderiza la pantalla y
   persiste `ui_startup_load_ms` con el tiempo transcurrido desde `_TOTAL_LOAD_START`.
4. **Inicialización post-auth:** una vez autenticado el usuario, `app._schedule_post_login_initialization`
   prepara métricas, mantenimiento SQLite y diagnósticos en segundo plano.

El valor de `ui_startup_load_ms` queda visible en el panel **"🔎 Diagnóstico del sistema"**
junto a `ui_total_load_ms`, y se publica en Prometheus como gauge homónimo.
Para consultarlo manualmente:

* **Prometheus:** solicitá `/metrics` y buscá la línea `ui_startup_load_ms <valor>`.
* **UI:** abrí la sección "🕒 Tiempos de arranque" dentro del panel de diagnóstico para ver
  el último registro en milisegundos.

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

### Worker predictivo asincrónico

Las simulaciones sectoriales ahora se ejecutan a través de un worker en segundo
plano definido en `application/predictive_jobs`. Cada solicitud publica un job
identificado por `job_id`; la UI muestra el último resultado cacheado y un
spinner con el estado (`pending`, `running`, `failed`).

* El TTL del resultado se sincroniza con `MarketDataCache.resolve_prediction_ttl`
  para evitar drift entre el cache y el worker.
* Consultá el estado en caliente mediante la función
  `application.predictive_service.predictive_job_status(job_id)`.
* Cuando el job finaliza, las métricas de latencia (`predictive_job_latency`) se
  registran junto con los contadores de hits/misses existentes.

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

