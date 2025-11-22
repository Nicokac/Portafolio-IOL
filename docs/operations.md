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
   corre una única vez, marca `_security_validated` y aborta si faltan secretos.
2. **Preload pausado:** `_render_login_phase()` arranca
   `start_preload_worker(paused=True)`, marca `scientific_preload_ready=False` y
   muestra la pantalla de login sin dependencias pesadas.
3. **Login interactivo:** `ui.login.render_login_page` registra
   `ui_startup_load_ms` al quedar visible la pantalla.
4. **Reanudación científica:** `_schedule_scientific_preload_resume()` se invoca
   inmediatamente después de renderizar el login para reanudar el worker con
   `resume_preload_worker(delay_seconds=0.0)`. Las vistas de análisis llaman a
   `ui.helpers.preload.ensure_scientific_preload_ready`, que muestra un *spinner*
   corto hasta que el worker termina.
5. **Inicialización post-auth:** `app._schedule_post_login_initialization`
   prepara métricas, mantenimiento SQLite y diagnósticos en segundo plano.

El valor de `ui_startup_load_ms` queda visible en el panel **"🔎 Diagnóstico del sistema"**
junto a `ui_total_load_ms`, y se publica en Prometheus como gauge homónimo.
Para consultarlo manualmente:

* **Prometheus:** solicitá `/metrics` y buscá `ui_startup_load_ms`,
  `preload_total_ms` y las métricas por librería
  (`preload_pandas_ms`, `preload_plotly_ms`, `preload_statsmodels_ms`).
* **UI:** abrí la sección "🕒 Tiempos de arranque" dentro del panel de diagnóstico para ver
 el último registro en milisegundos junto al estado de la precarga.

**Configurar la lista científica:** el worker lee `APP_PRELOAD_LIBS` (coma
separada) si se necesita ampliar o acotar la precarga; de lo contrario usa el
trío `pandas`, `plotly`, `statsmodels`. En despliegue se fuerza
`APP_PRELOAD_LIBS=pandas,plotly` para reducir la precarga científica. Evitá
añadir `application.predictive_service` o `controllers.portfolio.charts`, que
continúan importándose bajo demanda vía `importlib.import_module`.

**API de reanudación (`resume_preload_worker`):**

* **Firma:** `resume_preload_worker(delay_seconds=0.0, libs_override=None)`.
* **Quién la llama:** `ui.orchestrator._schedule_scientific_preload_resume` al
  terminar de renderizar la pantalla de login. Otros orquestadores pueden
  reusarla si necesitan repetir la precarga con un conjunto custom.
* **Cuándo:** inmediatamente después del login exitoso o con un `delay_seconds`
  acotado cuando se desea posponer la reanudación sin bloquear la UI.
* **Parámetros:**
  * `libs_override`: lista opcional de módulos a precargar (ej. `("pandas",)`);
    si no se indica, se usa `APP_PRELOAD_LIBS` o el trío por defecto.
  * `delay_seconds`: demora opcional antes de disparar el `Event` que despierta
    al hilo. Útil para coordinar con otras tareas de arranque.
* **Invariantes:** el hilo `preload_worker` es el único responsable de importar
  librerías pesadas; el hilo principal solo programa la reanudación y consulta
  `get_preload_metrics()` para leer el último resultado.

**Métricas estructuradas de precarga:** cada import registra en
`logs/app_startup.log` un JSON con `{event, module_name, duration_ms, status,
timestamp}` y un resumen final `{event:"preload_total", resume_delay_ms,
libraries}`. Estas líneas permiten auditar cuánto demoró cada módulo y cuándo se
disparó la reanudación desde la UI.

### Fase A / Fase B y alarmas

* **Fase A:** va desde `TOTAL_LOAD_START` hasta que se renderiza el login.
  El evento `login_screen_rendered` agrega `startup_ms` y `phase_a_status`
  (`ok` si < 500 ms, `alert` en caso contrario).
* **Fase B:** va desde la validación de credenciales hasta que el worker de
  precarga marca `preload_ready=True`. El evento `startup_phase_timings` incluye
  `phase_b_ms` y `phase_b_status` (`ok` si < 1 s tras el login).
* **Inicio y fin del preload:** el evento `preload_worker_started` registra
  timestamp, librerías y si quedó pausado; el cierre se refleja en
  `preload_total` con `status` y `resume_delay_ms`.
* **Análisis renderizados:** `analysis_screen_rendered` se emite una sola vez
  por pestaña (portafolio, recomendaciones, comparativa, monitoreo) con el
  tiempo de arranque acumulado.
* **Arranque de la app:** `app_start` agrega una marca de tiempo inicial para
  correlacionar Fase A con el tiempo de proceso.

**Objetivos operativos:** Fase A < 500 ms y Fase B < 1000 ms tras el login. Si
se usan métricas centralizadas (Prometheus o logs parseados), sugerimos:

* Panel: gráfico de barras apilado por `phase_a_ms` y `phase_b_ms` filtrando
  por `phase_*_status="alert"` para detectar regresiones.
* Alerta: en Prometheus, un `alert` sobre `max_over_time(ui_startup_load_ms[5m])`
  > 500 o `max_over_time(preload_total_ms[5m]) > 1000` puede encender una
  notificación (Slack/Email). Con logs estructurados, agregá una regla que
  cuente eventos `phase_*_status=alert` en ventanas de 15 minutos.

**Extender `APP_PRELOAD_LIBS`:**

1. Editá la variable de entorno (Procfile o deployment) y añadí los módulos
   separados por coma: `APP_PRELOAD_LIBS=pandas,plotly,statsmodels,seaborn`.
2. Confirmá que los nuevos imports son **puros** (sin side-effects de red) para
   que el worker no se bloquee. Si son pesados, inicializalos vía
   `importlib.import_module` dentro de `services/preload_worker.py` para que
   queden instrumentados.
3. Documentá el motivo en `docs/operations.md` y, si aplica, actualizá las
   pantallas científicas que dependan de la librería para mantener la lista
   sincronizada.

**Snapshot de bytecode:** durante el arranque `scripts/start.sh` ejecuta
`scripts/warmup_bytecode.py` (controlado por `ENABLE_BYTECODE_WARMUP`, habilitado
por defecto) para generar `.pyc` y reducir los costos de importación en frío.

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

