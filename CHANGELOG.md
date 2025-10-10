# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## v0.5.4 — Auditoría y QA de la serie 0.5.x
### Added
- `tests/application/test_regression_v054.py` ejecuta el flujo adaptativo completo con fixtures,
  valida MAE adaptativo < MAE estático y asegura ratio de cache ≥ 45 % con reporte Markdown.
- `shared/logging_utils.silence_streamlit_warnings` centraliza filtros de logging/warnings para
  ejecuciones offline sin ruido de Streamlit.
- Reporte de QA en `docs/qa/v0.5.4-validation-report.md` con métricas de cache, tiempos de render
  y sumario de validaciones.

### Changed
- `application.predictive_service` ahora expone snapshots de caché con % de hits y marca temporal
  normalizada, corrigiendo imports faltantes y formalizando el docstring del módulo.
- `services.cache.CacheService` formatea `last_updated` en `YYYY-MM-DD HH:MM:SS`, agrega método
  `stats()` e incrementa la trazabilidad de hits/misses.
- `ui/tabs/recommendations.py` muestra el ratio de hits en porcentaje, última actualización y usa
  el helper de logging compartido para suprimir warnings en modo bare.
- Fixtures de recomendaciones incluyen columna `sector` para consolidar la preparación histórica.

### Fixed
- Se sincronizaron exports de `application.__init__` para evitar importaciones implícitas y ciclos.
- El flujo adaptativo reutiliza caches dedicados con TTL estable, respetando el formato ISO en los
  reportes y evitando residuos tras los tests de regresión.

## v0.5.3 — Métricas extendidas del motor adaptativo
### Added
- `simulate_adaptive_forecast` ahora calcula `beta_shift_avg`, `sector_dispersion` y genera un resumen legible con metadatos de caché.
- Nuevo `export_adaptive_report` produce un reporte Markdown con resumen global, tabla temporal e interpretación de métricas.
- Pestaña **Correlaciones sectoriales** incorpora mini-card de β-shift/σ, botón de exportación y metadata de caché.

### Changed
- `CacheService` registra `hit_ratio` y `last_updated` en formato HH:MM:SS, reutilizados por la UI adaptativa.
- Logging del motor adaptativo reducido a nivel DEBUG para evitar ruido en consola.

## v0.5.2 — Aprendizaje adaptativo y correlaciones dinámicas
### Added
- `application.adaptive_predictive_service` introduce un estado persistente con TTL de 12 horas, cálculo de correlaciones adaptativas vía EMA y simulaciones históricas que reportan MAE, RMSE y bias.
- `tests/application/test_adaptive_predictive_service.py` cubre la evolución temporal del modelo, la persistencia de estado y la reducción de error frente a las predicciones originales.
- Nuevo tab **Correlaciones sectoriales** en `ui/tabs/recommendations.py` con matrices histórica/rolling/adaptativa, resumen de β promedio y dispersión sectorial más las métricas del motor adaptativo.
- `ui/charts/correlation_matrix.py` genera la visualización β-shift y se documenta el flujo en `docs/adaptive_learning_overview.md`.

### Changed
- Insight automático y `_render_for_test()` incorporan los datos adaptativos para exponer β-shift promedio y la correlación dinámica junto al resto de métricas.

## v0.5.1 — Forecasting y Retornos Proyectados
### Added
- `application.predictive_service.predict_sector_performance` con suavizado EMA,
  penalización por correlaciones intrasectoriales y métricas de confianza.
- Columna **Predicted Return (%)** y toggle *Incluir predicciones* en la pestaña
  de recomendaciones, además del contador de hits/misses del caché predictivo.
- Tests unitarios para el motor predictivo y la integración de retornos
  proyectados en `RecommendationService`.

### Changed
- Insight automático recalculado con promedios ponderados por asignación y
  racional extendido que destaca la predicción sectorial.

## v0.5.0-dev — Inicio del ciclo de consolidación predictiva
- Se incorpora `application.backtesting_service.BacktestingService`, reutilizando indicadores de `ta_service` y datos de fixtures para ejecutar backtests sin depender de la API de IOL.
- Nueva `CacheService` con TTL configurable en `services/cache.py` para cachear precios históricos, resultados simulados y adaptadores offline durante la transición a la serie 0.5.x.
- Fixtures offline en `docs/fixtures/default/` (precios con indicadores, perfil base y recomendaciones de ejemplo) que alimentan `_render_for_test()` y las pruebas unitarias.
- Versión sincronizada a `0.5.0-dev` en `pyproject.toml`, `shared/version.py`, README y CHANGELOG manteniendo la compatibilidad funcional de la release 0.4.4.

## v0.4.4 — Perfil inversor persistente y comparativas con benchmarks
- Nuevo `ProfileService` con almacenamiento cifrado que sincroniza tolerancia al riesgo, horizonte e
  inclinación estratégica entre `session_state`, `config.json` y `st.secrets`.
- La pestaña de recomendaciones permite ajustar el perfil mediante selectores dedicados, mostrando un
  badge con el perfil activo y aplicando sesgos en `RecommendationService.recommend()`.
- Bloque comparativo frente a Merval, S&P 500 y Bonos que resume ΔRetorno, ΔBeta y Tracking Error
  usando el nuevo `compute_benchmark_comparison()`.
- Documentación y versión actualizadas para la release 0.4.4, junto con pruebas unitarias de perfil y
  benchmarking.

## v0.4.3 — Recomendaciones exportables y explicadas al instante
- Incorporadas descargas "📤 Exportar CSV" y "📥 Exportar XLSX" con promedios finales de retorno y beta.
- Añadido racional extendido que cuantifica aporte al retorno, impacto en beta y diversificación sectorial.
- Insight automático enriquecido con la detección del sector dominante dentro de las sugerencias.

## v0.4.2 — Simulador de Inversión Inteligente
- Refinado algoritmo de recomendación con límites de peso y balanceo sectorial.
- Agregada visualización gráfica de distribuciones sugeridas (Pie y Barras).
- Implementado simulador de impacto con métricas Antes/Después (valor total, retorno, beta).
- Tests unitarios y lógicos validados por script en entorno QA.
- Pendiente: validar renderizado visual completo cuando el mock de API IOL esté disponible.

## v0.4.0 — Factor & Benchmark Analysis (Dec 2025)

**Fecha:** 2025-12-05

**Novedades principales:**
- Incorporado el módulo de *Análisis de Factores y Benchmark* con métricas de Tracking Error,
  Active Return e Information Ratio directamente en el tablero de riesgo.
- Nuevo servicio `application.benchmark_service` para centralizar cálculos de seguimiento y
  regresiones multi-factoriales con soporte para factores macroeconómicos opcionales.
- Visualización de betas por factor con indicación de R², más exportaciones CSV/XLSX desde el
  controlador de riesgo.
- Cobertura de pruebas unitarias e integradas para los cálculos y la nueva UI, junto con
  documentación actualizada en README y guías de testing.

**QA Check:**
✅ Verificar que el panel “Análisis de Factores y Benchmark” renderice correctamente.
✅ Confirmar coherencia entre Tracking Error y Information Ratio.
⚠️ Cuando no haya datos de benchmark, mostrar aviso de datos insuficientes.

## v0.3.4.4.6 — Clasificación y visualización completa por tipo de activo (Nov 2025)

### Summary
- El heatmap de riesgo ahora genera pestañas para cada tipo de activo detectado en el portafolio
  (CEDEAR, Acciones locales, Bonos, Letras, FCI, ETFs y Otros) aun cuando no existan suficientes
  símbolos para calcular correlaciones, mostrando advertencias contextuales cuando corresponde.
- Se amplió el mapeo canónico de tipos (`_TYPE_ALIASES`) para contemplar variantes frecuentes como
  "Bonos Dólar", "Letras del Tesoro" o fondos money market, manteniendo etiquetas visuales
  estandarizadas.
- Nuevas pruebas en `tests/controllers/test_risk_filtering.py` cubren la presencia de todas las
  pestañas y las advertencias asociadas; README y documentación de testing actualizados junto con el
  incremento de versión a 0.3.4.4.6.

## v0.3.4.4.5 — Local Equity Tab in Risk Heatmap (Nov 2025)

### Summary
- El análisis de correlaciones crea una pestaña dedicada para **Acciones locales**, reutilizando la
  clasificación del catálogo base para separar claramente CEDEARs y renta variable doméstica.
- Se preserva la exclusión de tickers locales al seleccionar el grupo de CEDEARs, evitando que
  LOMA, YPFD o TECO2 aparezcan en matrices cruzadas con instrumentos del exterior.
- Documentación, guías de prueba y materiales de comunicación actualizados para reflejar el
  comportamiento del nuevo heatmap junto con el incremento de versión a 0.3.4.4.5.

## v0.3.4.4.4 — Asset Type Alignment in Risk Analysis (Nov 2025)

### Summary
- El cálculo de correlaciones ahora se apoya exclusivamente en la clasificación del portafolio
  base antes de solicitar históricos, aplicando un mapeo canónico por símbolo para evitar que
  instrumentos de distintos tipos se mezclen en el heatmap.
- Los CEDEARs filtran explícitamente los tickers locales (LOMA, YPFD, TECO2) aunque el payload
  de precios o cotizaciones los etiquete erróneamente, manteniendo matrices homogéneas por
  categoría.
- Se añadieron pruebas de controlador que validan el filtro corregido y la asignación de tipos
  desde el catálogo maestro, junto con documentación y materiales de release actualizados para la
  versión 0.3.4.4.4.

## v0.3.4.4.3 — Risk Heatmap Polishing Pass (Nov 2025)

### Summary
- Elimina del cálculo de correlaciones a los activos con rendimientos de varianza nula o indefinida,
  evitando coeficientes erráticos y matrices singulares.
- Los heatmaps de correlación ahora muestran títulos contextualizados por tipo de activo (por
  ejemplo, "Matriz de Correlación — CEDEARs"), lo que refuerza la segmentación aplicada en los
  filtros del análisis de riesgo.
- README y materiales de release actualizados para documentar el descarte de columnas sin
  movimiento y el nuevo etiquetado por grupo.

## v0.3.4.4.2 — Vertical Sidebar Layout (Nov 2025)

### Summary
- Reorganiza los grupos de controles de la barra lateral en tarjetas apiladas verticalmente, manteniendo títulos, captions y tooltips consistentes.
- Mejora la lectura de filtros y acciones al asignar una fila completa a cada bloque (Actualización, Filtros, Moneda, Orden, Gráficos y Acciones) con padding uniforme.
- Conserva el feedback visual al aplicar filtros, resaltando únicamente la sección afectada sin alterar la lógica del formulario.

## v0.3.4.4.1 – Header Centering & Cleanup Hotfix (Nov 2025)

### Summary
- Centra el hero principal del dashboard y elimina el bloque redundante de "Enlaces útiles" del encabezado, manteniendo el bloque únicamente en el footer.
- Refina la composición visual inicial para que el título, subtítulo y resumen FX queden alineados sin alterar datos ni microinteracciones previas.

## v0.3.4.4 — UX Consistency & Interaction Pass (Nov 2025)

### Summary
- Consolidación de microinteracciones en la barra lateral y los formularios clave: estados _hover_, enfoque visible y tooltips sincronizados con los controles de presets y filtros.
- Confirmaciones in-app homogéneas: toasts, banners y contadores sincronizados entre el panel principal y la pestaña **Monitoreo** para que cada acción de screening muestre feedback inmediato.
- Ajustes de ritmo visual y tiempos de carga: skeletons y spinners consistentes en dashboards, exportaciones y healthcheck para reducir saltos al cambiar de contexto.

### Added
- Puerta de calidad de seguridad en CI que ejecuta `bandit` sobre el código crítico y `pip-audit`
  sobre los requirements para bloquear vulnerabilidades antes del merge.

### Documentation
- `docs/testing.md` actualizado con los comandos oficiales de auditoría (`bandit` y `pip-audit`) y
  la exigencia de cobertura configurada por defecto en `pytest`.

### Tests
- Configuración de `pytest` actualizada para imponer cobertura sobre `application`, `controllers` y
  `services` en cada ejecución, alineada con la nueva puerta de seguridad de CI.

## v0.3.4.3 — Layout Consolidation & Sidebar Unification (Nov 2025)

### Summary
- Se creó la pestaña **Monitoreo** para alojar el healthcheck completo y se añadió un badge global de estado en la cabecera.
- Todos los controles del portafolio, el panel de control y las preferencias de apariencia se reubicaron en la barra lateral bajo un contenedor colapsable.
- La vista principal del portafolio aprovecha el ancho completo con espaciado uniforme tras retirar el panel superior.
- El footer incorpora un bloque de enlaces útiles con acceso directo a documentación y soporte.

### Documentation
- `README.md`, `docs/testing.md` y `banners/README` describen el nuevo flujo con sidebar unificado y la pestaña de Monitoreo.
- La versión de la aplicación se actualizó a 0.3.4.3 en código y materiales de release.

## v0.3.4.2 — Visual Polish Pass (Nov 2025)

### Summary
- Incremento de padding y márgenes clave para asegurar el respiro visual del panel superior y las
  tarjetas de KPIs en resoluciones medianas.
- Tarjetas contrastadas y tipografía reajustada para reforzar la jerarquía de información en los
  indicadores del dashboard.
- Alineación central consistente de los bloques del header y filtros, evitando saltos laterales en el
  selector de riesgo.
- Ajustes en el footer: espaciado, alineación de enlaces y consistencia con la narrativa “Observabilidad
  operativa”.

## v0.3.4.1 — Layout y Filtros de Análisis de Riesgo (Nov 2025)

### Summary
- El panel superior del dashboard de análisis se reposicionó como una franja horizontal fija, sobre
  la grilla de contenido, liberando espacio lateral para los gráficos.
- Se adoptó un layout de ancho completo en la vista principal para priorizar la lectura del heatmap
  de riesgo y los indicadores asociados.
- Los filtros del heatmap incorporan un selector por tipo de instrumento que permite acotar el
  análisis sin depender de la antigua barra lateral.

### Documentation
- `README.md`, `docs/testing.md` y `banners/README` reflejan el nuevo layout horizontal y la
  liberación del sidebar para controles.

## v0.3.4.0 — UI Experience Refresh (Oct 2025)

### Summary
Consolidación del roadmap UX/UI iniciado en la release 0.3.30.13, con foco en accesibilidad, jerarquía visual y coherencia narrativa dentro del panel de usuario.  
La versión 0.3.4.0 representa una evolución estética y funcional del dashboard, manteniendo la estabilidad del backend y el enfoque en observabilidad operativa.

### Highlights
- **Refinamiento visual (Fase 1):** Reestructuración del encabezado en formato hero de dos columnas, nuevo resumen FX y reorganización del bloque de seguridad en la pantalla de login.  
- **Experiencia interactiva (Fase 2):** Conversión del menú de acciones en panel persistente con tooltips, layout de doble columna con control fijo y health sidebar expandible con secciones delimitadas.  
- **Personalización guiada (Fase 3):** Chips visuales para filtros activos, vista previa de exportaciones con métricas y fallbacks Kaleido reforzados, además de badges dinámicos por pestaña.  
- Unificación de estilos, tamaños de fuente y espaciado entre secciones clave.  
- Coherencia entre encabezado, footer y panel lateral bajo la narrativa “Observabilidad operativa”.

### Testing
- Validado con `python -m compileall` en módulos UI y layout actualizados.  
- Ejecución parcial de `pytest --override-ini addopts=''` confirmando integridad de componentes.  
- CI visual manual en entorno Streamlit (QA offline).

## [0.3.30.13] — Observabilidad reforzada en ejecución

### Added
- Telemetría de entorno con snapshot automático de variables críticas (Python, Streamlit, Kaleido y
  binarios del sistema) visible desde la UI y embebida en `analysis.log` para acelerar diagnósticos
  remotos.
- Rotación automática de logs con compresión diaria y retención configurable que evita que `~/.portafolio_iol/logs`
  crezca sin control en estaciones con screenings intensivos.
- Controles de dependencias al inicio que advierten por UI y CLI cuando falta Kaleido, faltan
  binarios de exportación o la versión de Python está fuera del rango soportado.

### Changed
- Barra lateral y pantalla de login muestran un bloque de "Observabilidad" con accesos rápidos para
  descargar snapshots de entorno y el paquete de logs rotados.
- Documentación de descarga guiada dentro de la UI para educar a los analistas sobre cómo compartir
  snapshots, logs y artefactos de exportación al escalar incidentes.

### Fixed
- Se evitó la sobrescritura silenciosa de `analysis.log` cuando el proceso se relanza en entornos con
  permisos restringidos, delegando la rotación en un handler tolerante a fallas.

## [0.3.30.12.1] — Hotfix: diagnóstico de inicio resiliente

### Fixed
- Se restauró el registro de diagnósticos de inicio para tolerar snapshots mal formados,
  conservar la telemetría en el health sidebar y evitar que el flujo de login falle.

## [0.3.30.12] — Estabilización y Monitoreo de Sesión

### Added
- Timeline de sesión en el health sidebar con `session_tag`, timestamps y origen de cada hito (login,
  screenings, exportaciones) para diagnosticar degradaciones y rebotes de UI sin revisar logs crudos.
- Etiquetas de sesión en `analysis.zip`, `analysis.xlsx` y `summary.csv` para rastrear qué ejecución
  generó los artefactos y correlacionarlos con los eventos registrados en `analysis.log`.

### Changed
- Banners de login/sidebar actualizados para resaltar "Estabilización y monitoreo de sesión" y el nuevo
  badge de timeline visible para QA.
- README, guías de testing y troubleshooting ajustadas para reflejar el monitoreo de sesión, los TTL
  en vivo y los pasos de verificación asociados en pipelines.

### Fixed
- Normalización del `session_tag` almacenado en `st.session_state` para evitar duplicados tras reruns
  y asegurar que los contadores de resiliencia conserven la trazabilidad de cada sesión.

## [0.3.30.11] — Mantenimiento, observabilidad y optimización de logs/cache.

### Changed
- TTL de caché revisado para mantener los paneles cálidos sin sacrificar consistencia ni forzar
  rehidrataciones innecesarias en los screenings nocturnos.
- Panel de health actualizado con métricas de observabilidad que enlazan directamente con
  `analysis.log`, facilitando el seguimiento de degradaciones y alertas proactivas.

### Fixed
- Limpieza del pipeline de logging para eliminar archivos huérfanos y entradas duplicadas en
  `analysis.log`, reduciendo ruido operativo y facilitando auditorías.

## [0.3.30.10.2] - Robust Excel export

### Fixed
- Reforzada la generación de `analysis.xlsx` para reintentar exportes con hojas vacías y conservar
  el archivo dentro de `analysis.zip` aun cuando la primera iteración falle.
- Normalizados los nombres de hojas y encabezados para evitar errores de `ExcelSheetNameError` en
  portafolios con símbolos extensos o caracteres especiales.
- Alineada la conversión de tipos mixtos en columnas numéricas para impedir que se descarten
  registros al aplicar formatos durante la exportación.

## [0.3.30.10.1] - Hotfix entorno Kaleido

### Changed
- Limpieza y resincronización de dependencias en `pyproject.toml` y los requirements planos
  para evitar paquetes redundantes en CI/CD y entornos mínimos.

### Fixed
- Restaurado el fallback de exportación cuando `kaleido` no está disponible: la aplicación
  mantiene los artefactos CSV/Excel, etiqueta el estado en los banners y registra la ausencia
  de PNG para los pipelines.

### Documentation
- README, guías de testing y troubleshooting actualizadas con la release 0.3.30.10.2, el hotfix
  de Kaleido y el mensaje visible en los banners.

## [0.3.30.10] - 2025-10-15

### Fixed
- Se restableció la tubería de logging para que todos los flujos de screening y exportación vuelvan a
  registrar eventos en `analysis.log`, incluyendo los `snapshot_hits`, degradaciones controladas y la
  procedencia de los datos consumidos por la UI.
- Los exports (`analysis.zip`, `analysis.xlsx`, `summary.csv`) vuelven a generarse con el set completo
  de archivos, preservan los timestamps de ejecución y adjuntan la bitácora consolidada en los artefactos
  de CI.

### Documentation
- README, guías de troubleshooting y banners actualizados para recalcar la release 0.3.30.10 y los
  fixes de logging/export que devuelven la trazabilidad a pipelines y operadores.

## [0.3.30.9] - 2025-10-10

### Fixed
- Se reparó el flujo de cotizaciones en vivo: `/Titulos/Cotizacion` vuelve a sincronizarse con
  `/Cotizacion`, respeta el fallback jerárquico y expone el origen real de cada precio en la UI.
- Se corrigió el sidebar para mostrar el estado actualizado del feed live, la versión `0.3.30.9` y la
  salud de los proveedores sin mensajes inconsistentes.

### Added
- Integración del país de origen en el portafolio para habilitar filtros, dashboards y exports
  multi-país en los análisis de cartera.

### Documentation
- README, guías de testing y troubleshooting actualizadas para destacar la release 0.3.30.9, las
  cotizaciones en vivo restauradas y las verificaciones necesarias en banners y pipelines.

## [0.3.30.8] - 2025-10-06

### Added
- Sesiones legacy cacheadas para reutilizar credenciales válidas y reducir latencia al restaurar
  contextos degradados.
- Rate limiting integrado en los clientes de datos para proteger los umbrales de APIs externas y
  evitar bloqueos al ejecutar pipelines intensivos.
- Recuperación automática de valorizaciones recientes cuando la fuente primaria falla, garantizando
  que la UI y los reportes mantengan cifras consistentes.

## [0.3.30.7] - 2025-10-05

### Fixed
- Corrección del fallback jerárquico que perdía el escalón secundario cuando el proveedor primario
  devolvía credenciales inválidas, garantizando que la degradación continúe hasta el snapshot
  persistido.
- Sincronización del banner de login y del health sidebar para reflejar la procedencia real de los
  datos servidos durante la degradación, evitando mensajes inconsistentes.
- Ajuste del contador `snapshot_hits` para propagar correctamente los resultados recuperados por el
  fallback endurecido y mantener la telemetría alineada en dashboards y exportaciones.

### Documentation
- README, guías de testing y troubleshooting actualizadas con la release 0.3.30.7 y los pasos para
  validar los fixes del fallback jerárquico.

## [0.3.30.5] - 2025-10-04

### Fixed
- Se normalizó la publicación de cotizaciones nulas para evitar excepciones en telemetría y dashboards.
- El backend de snapshots ahora se auto-configura en inicializaciones en frío, evitando estados parciales.
- Se restauró el fallback legacy para consultas de mercado cuando el proveedor principal no responde.
- Se reactivó la valorización de portafolios tras interrupciones de caché, garantizando cifras consistentes.

## [0.3.30.4] - 2025-10-04

### Added
- Nuevo endpoint `/Cotizacion` que publica cotizaciones normalizadas para los consumidores internos y externos.

### Fixed
- Manejo reforzado de errores HTTP 500 provenientes de upstream para evitar caídas en dashboards y telemetría.

### Tests
- Prueba de cobertura dedicada que valida los flujos de cotización bajo escenarios de error y resiliencia.

## [0.3.30.3] - 2025-10-04

### Fixed
- Corrección definitiva del backend de snapshots para asegurar que `_ensure_configured()` se ejecute
  antes de cualquier lectura en dashboards o pipelines CI, evitando inicializaciones incompletas.
- Normalización de la firma `IOLClient.get_quote()` y de los flujos de cotizaciones para aceptar
  `(market, symbol, panel)` sin romper la telemetría ni los consumidores existentes.

## [0.3.30.2] - 2025-10-04

### Fixed
- Agregado `_ensure_configured()` en `services/snapshots.py` para evitar errores de inicialización.
- Corregida la firma de `IOLClient.get_quote()` para aceptar `(market, symbol, panel)`.
- Validación completa de persistencia de snapshots y consultas de mercado sin errores.

## [0.3.30.1] - 2025-12-01

### Changed
- Limpieza de escenarios duplicados y migración final de controladores/servicios fuera de
  `infrastructure.iol.legacy`, consolidando el uso de `IOLClientAdapter` y
  `PortfolioViewModelService` como fuentes únicas para la UI y los scripts.
- Ajuste de los pipelines para auditar importaciones legacy con `rg` y reforzar que `pytest` sólo
  recolecte suites modernas.

### Documentation
- README, guía de pruebas y troubleshooting actualizados con la versión 0.3.30.1, instrucciones de
  migración (helpers reemplazados, stub oficial de Streamlit) y comandos para ejecutar suites sin
  módulos legacy.

### Tests
- Checklist de CI actualizada para exigir `pytest --ignore=tests/legacy`, auditorías de importaciones
  legacy y verificación de artefactos (`coverage.xml`, `htmlcov/`, `analysis.zip`, `analysis.xlsx`,
  `summary.csv`).

## [0.3.29.2] - 2025-11-24

### Changed
- Hardening de CI/cobertura alineado con los hitos [CI resiliente 0.3.29.2](https://github.com/Portafolio-IOL/portafolio-iol/milestone/43)
  y [Cobertura exportaciones 0.3.29.2](https://github.com/Portafolio-IOL/portafolio-iol/milestone/44), incorporando validaciones
  cruzadas entre `pytest`, `coverage.xml` y los artefactos de exportación (CSV, ZIP y Excel) para bloquear merges sin evidencia
  de reportes completos.
- El pipeline ahora normaliza la recolección de artefactos (`htmlcov/`, `summary.csv`, `analysis.zip`, `analysis.xlsx`) y marca como
  fallidos los jobs que no adjuntan cobertura o exportaciones esperadas.

### Documentation
- README, guía de pruebas y troubleshooting actualizados para la release 0.3.29.2 con la nueva sección **CI Checklist** y ejemplos
  de exportación alineados a los artefactos `analysis.zip`, `analysis.xlsx` y `summary.csv`.

### Tests
- Checklist de CI incorporada en la documentación para garantizar que `pytest --cov` publique `htmlcov/` y `coverage.xml`, y que las
  suites de exportación validen la presencia de CSV, ZIP y Excel antes de dar por válidos los pipelines.

## [0.3.29.1] - 2025-11-22

### Changed
- Hardening de arquitectura y exportaciones: las validaciones de Markowitz ahora bloquean presets
  inconsistentes y sincronizan la telemetría con los contadores de resiliencia para evitar falsos
  positivos en screenings cooperativos.
- Refuerzo de CI para escenarios multi-proveedor, ejecutando la suite de integración completa y
  asegurando que los pipelines configuren el backend de snapshots en modo temporal (`Null`/`tmp_path`).

### Documentation
- README, guía de pruebas y troubleshooting alineados con la versión 0.3.29.1, con comandos de
  exportación que detallan parámetros `--input`, artefactos generados (CSV, ZIP y Excel) y los pasos
  para forzar escenarios multi-proveedor en CI.
- Documentación de las nuevas validaciones Markowitz y de la configuración recomendada para el
  backend de snapshots en pipelines efímeros.

### Tests
- Recordatorios en CI para ejecutar `pytest tests/integration/` completo y validar degradaciones
  multi-proveedor antes de publicar artefactos.

## [0.3.29] - 2025-11-20

### Changed
- Sincronización del versionado 0.3.29 entre `pyproject.toml`, `shared.version` y las superficies
  visibles para mantener la trazabilidad durante el hardening de CI.

### Documentation
- README, guías de pruebas y troubleshooting alineados con la numeración 0.3.29 y con ejemplos de
  exportación actualizados (`--input`, `--formats`, directorios de salida) que reflejan el
  comportamiento real de `scripts/export_analysis.py`.

### Tests
- Recordatorios de ejecución en CI y validaciones manuales actualizados para utilizar la versión
  0.3.29 al verificar banners y reportes exportados.

## [0.3.28.1] - 2025-11-18

### Changed
- Hardening de pipelines CI: sincronización de versionado entre `pyproject.toml`, `shared.version`
  y superficies visibles, más validaciones adicionales de telemetría para detectar desalineaciones
  en los contadores persistentes.

### Documentation
- README, guías de pruebas y troubleshooting actualizadas para reflejar la release 0.3.28.1 como
  parche de hardening/CI y mantener vigentes los flujos de snapshots, exportaciones y observabilidad.

### Tests
- Recordatorios de ejecución en CI ajustados para garantizar que las suites utilicen la numeración
  0.3.28.1 en banners, stubs y verificaciones de versionado.

## [0.3.28] - 2025-11-15

### Added
- Script `scripts/export_analysis.py` para generar exportaciones enriquecidas del screening con
  resúmenes agregados y notas de telemetría.
- Métricas de almacenamiento y contadores de snapshots visibles en el health sidebar para rastrear
  recuperaciones desde el almacenamiento persistente.

### Changed
- Persistencia de snapshots del portafolio y de los presets del sidebar para acelerar screenings
  consecutivos y dejar trazabilidad en la telemetría.

### Documentation
- README actualizado con la narrativa de la release (snapshots persistentes, exportaciones
  enriquecidas, observabilidad extendida) e instrucciones paso a paso para `scripts/export_analysis.py`.
- Guías de pruebas y troubleshooting extendidas con escenarios específicos para validar el nuevo
  almacenamiento y depurar métricas de observabilidad.

### Tests
- Nuevas recomendaciones de QA para ejecutar suites y escenarios manuales que ejercitan los contadores
  de snapshots y las rutas de fallback persistente.

## [0.3.27.1] - 2025-11-07

### Changed
- Persistencia del health sidebar reforzada para conservar la última secuencia de degradación y los
  contadores de resiliencia aun después de recargar la sesión, evitando inconsistencias entre la UI
  y la telemetría de backend.

### Documentation
- Se documentó la configuración de claves (Alpha Vantage, Polygon, FMP, FRED y World Bank) y los
  pasos para validar el fallback jerárquico desde el health sidebar, alineando README y guías de
  troubleshooting con la nueva release.

### Tests
- Se estabilizaron las suites que validan la degradación multinivel (`tests/test_version_display.py`
  y escenarios macro) con fixtures de claves deterministas para asegurar la cobertura de
  resiliencia en CI.

## [0.3.27] - 2025-11-05

### Added
- Monitor de resiliencia en el health sidebar que expone el último proveedor exitoso, la secuencia de
  degradación (`primario → secundario → fallback`) y las insignias de recuperación asociadas.

### Changed
- Centralización de timeouts, backoff y códigos de error para los clientes de APIs macro y de
  portafolio, asegurando que los fallback registrados en telemetría conserven la procedencia y la
  latencia de cada intento.
- Notificaciones internas (`st.toast`) actualizadas para informar cuando un proveedor externo vuelve
  a estar disponible tras un incidente, manteniendo trazabilidad directamente en la UI.

### Documentation
- README y guías alineadas con la release 0.3.27: quick-start renovado, escenarios de resiliencia
  multi-API, fecha de publicación y recordatorios para verificar la versión visible en header/footer.

## [0.3.26.1] - 2025-10-26

### Added
- Notificaciones internas en la UI basadas en `st.toast` para confirmar refrescos y cierres de sesión,
  consolidando feedback inmediato para los analistas que operan desde el dashboard.

### Changed
- Sincronización del número de versión 0.3.26.1 entre `pyproject.toml`, `shared.version` y las superficies
  visibles (header, footer, sidebar y tests) para mantener la trazabilidad de la release.

### Documentation
- README, guías y quick-start alineados con la release 0.3.26.1, detallando el flujo de notificaciones
  internas y los pasos para validar la numeración visible.

## [0.3.26] - 2025-10-19

### Changed
- El login y el dashboard principal reutilizan el helper `shared.version` para mostrar "Versión 0.3.26"
  con la hora actualizada por `TimeProvider`, garantizando que el encabezado y el footer compartan
  la misma metadata visible.
- El health sidebar consolida la cronología de screenings con badges de cache hit/miss y métricas de
  fallback sincronizadas con los contadores globales, evitando discrepancias entre la vista tabular y
  los totales expuestos en la parte superior del panel.

### Fixed
- `ui.ui_settings.apply_settings` ahora verifica la disponibilidad de `st.set_page_config` antes de
  invocarlo, permitiendo ejecutar suites locales con stubs de Streamlit que no exponen ese método.
- `app.py` define stubs de compatibilidad (`st.stop`, `st.container`, `st.columns`) cuando la API de
  Streamlit no los ofrece, destrabando los tests que importan la app en entornos fuera de Streamlit.

### Documentation
- README actualizado con el quick-start de la release 0.3.26, incluyendo instrucciones para verificar
  la versión visible en header/footer y resúmenes renovados de telemetría.

### Tests
- Las suites `tests/test_version_display.py` y `tests/test_version_sync.py` se mantienen alineadas con
  la numeración 0.3.26 para validar el helper de versión y la visibilidad en la UI.

## [0.3.25.1] - 2025-10-03

### Fixed
- Se corrigió la función `drawdown_series` en `application/risk_service.py` para manejar correctamente series vacías y calcular drawdowns acumulados, eliminando el `IndentationError` que impedía iniciar la aplicación.
- Se corrigió un `IndentationError` en `application/risk_service.py` causado por un bloque `if` sin cuerpo en la función `drawdown_series`.
- La función ahora retorna un `pd.Series` vacío cuando no hay datos de entrada, previniendo bloqueos en inicialización y permitiendo flujos consistentes en métricas de riesgo.
- La app vuelve a iniciar correctamente tras el reboot con la release 0.3.25.

### Tests
- Cobertura extendida para validar el manejo de series vacías en `drawdown_series`.

## [0.3.24.2] - 2025-10-10

### Fixed
- Se corrigió el `NameError` en `render_portfolio_section` al eliminar la referencia
  obsoleta a `apply_filters` y delegar la construcción del view-model al servicio
  cacheado de portafolio.
- `record_macro_api_usage` vuelve a registrar la última ejecución macro sin depender
  de variables temporales inexistentes, evitando el `NameError latest_entry` y
  propagando correctamente las métricas hacia el sidebar de salud.

### Changed
- `build_portfolio_viewmodel` ahora recibe un `PortfolioViewSnapshot` en lugar de
  ejecutar filtros manualmente, alineando la nueva capa de cache con los
  controladores.

### Tests
- Se actualizaron las suites de portafolio para simular el servicio de view-model
  cacheado y validar el flujo completo tras el refactor.
- Los tests de métricas de salud se adaptaron al nuevo contrato de macro
  (intentos normalizados + entrada más reciente) para cubrir el fix.

## [0.3.24.1] - 2025-10-09

### Tests
- La suite de CI recuperó su estabilidad tras ajustar los timeouts intermitentes y sincronizar los entornos de ejecución.

### Changed
- Los mocks de proveedores externos fueron alineados con los contratos vigentes para evitar desfasajes durante las pruebas integradas.

### Fixed
- La persistencia de favoritos ahora conserva los emisores marcados entre sesiones, incluso al alternar entre vistas y filtros derivados.

### Documentation
- Guías actualizadas describiendo la estabilidad recuperada, los mocks vigentes y el flujo persistente de favoritos para el release 0.3.24.1.

## [0.3.24] - 2025-10-08

### Changed
- Refactor del módulo de portafolio para simplificar dependencias internas y facilitar futuras extensiones en la UI y los controladores.

### Fixed
- Ajustes en los cacheos del screener para estabilizar invalidaciones y preservar resultados consistentes entre ejecuciones consecutivas.

### Added
- Gestión de favoritos en el portafolio que habilita marcar emisores clave y priorizarlos en los listados derivados.

### Documentation
- Plan de documentación para describir el refactor del portafolio, los escenarios de cacheo y el uso de favoritos en la próxima iteración.

## [0.3.23] - 2025-10-07
### Added
- Cliente dedicado para FRED con autenticación, gestión de rate limiting y normalización de observaciones para enriquecer el screener de oportunidades con contexto macro/sectorial. ([`infrastructure/macro/fred_client.py`](infrastructure/macro/fred_client.py))
- Métrica de salud que expone el estado de la nueva dependencia externa (`macro_api`), ampliando la observabilidad del sistema. ([`services/health.py`](services/health.py))
### Changed
- El controlador de oportunidades combina la información sectorial proveniente de FRED (o del fallback configurado) con los resultados del screening, agregando la columna `macro_outlook` y notas contextuales. ([`controllers/opportunities.py`](controllers/opportunities.py))
### Documentation
- README actualizado con los pasos para habilitar la integración macro, variables de entorno requeridas y consideraciones de failover. ([`README.md`](README.md#datos-macro-y-sectoriales-fred--fallback))
### Tests
- Cobertura específica para los flujos de fallback del controlador frente a la dependencia macro, asegurando la continuidad del screener. ([`controllers/test/test_opportunities_macro.py`](controllers/test/test_opportunities_macro.py))

## [0.3.22] - 2025-10-06
### Changed
- Sincronización del número de versión `0.3.22` entre `pyproject.toml`, el helper `shared.version`
  y las superficies visibles para mantener el encabezado de pestañas y el sidebar actualizados.
### Documentation
- Quick-start y menús documentados mencionando explícitamente la release 0.3.22 y reforzando el
  recordatorio de versión visible en la UI.

## [0.3.21] - 2025-10-05
### Changed
- Refinamiento UX del mini-dashboard del healthcheck para resaltar los tiempos cacheados vs. recientes con etiquetas de estado
  claras y tooltips que explican la metodología de medición. ([`ui/health_sidebar.py`](ui/health_sidebar.py))
### Added
- Telemetría histórica del screener que persiste los tiempos de ejecución previos y permite graficar tendencias directamente
  desde el panel de salud. ([`services/health.py`](services/health.py), [`controllers/opportunities.py`](controllers/opportunities.py))
### Documentation
- Se incorporó documentación multimedia (capturas y clips) que guía la interpretación del mini-dashboard y la navegación por la
  nueva telemetría histórica. ([`README.md`](README.md#caché-del-screener-de-oportunidades))

## [0.3.20] - 2025-10-04
### Added
- Mini-dashboard en el healthcheck que expone la duración previa y cacheada de los screenings de oportunidades, permitiendo
  comparar tiempos desde la UI. ([`controllers/opportunities.py`](controllers/opportunities.py), [`services/health.py`](services/health.py),
  [`ui/health_sidebar.py`](ui/health_sidebar.py))
### Changed
- Telemetría extendida para registrar aciertos de caché y variaciones de filtros del screener, dejando trazabilidad directa en el
  panel de salud. ([`services/health.py`](services/health.py), [`ui/health_sidebar.py`](ui/health_sidebar.py))
### Tests
- Casos que validan *cache hits* e invalidaciones al cambiar filtros del screener de oportunidades. ([`tests/controllers/test_opportunities_controller.py`](tests/controllers/test_opportunities_controller.py))
### Documentation
- Limpieza de referencias legacy y actualización de la estrategia de cacheo documentada para reflejar el nuevo dashboard y la
  telemetría extendida. ([`README.md`](README.md#caché-del-screener-de-oportunidades))

## [0.3.19] - 2025-10-03
### Added
- Presets personalizados en la UI del screener que permiten guardar y reutilizar combinaciones propias de filtros sin depender de configuraciones globales.
### Changed
- Flujo de comparación enriquecido para revisar lado a lado los resultados de presets activos, destacando las diferencias en filtros y métricas clave antes de confirmar los cambios.
### Fixed
- Cacheo de respuestas de Yahoo Finance homogeneizado entre backend y stub, evitando expiraciones adelantadas y asegurando consistencia en los resultados servidos a la UI.
### Documentation
- Limpieza de referencias legacy en las guías internas, documentando el nuevo flujo de presets personalizados y eliminando instrucciones obsoletas.

## [0.3.18] - 2025-10-02
### Added
- Los listados de oportunidades ahora incluyen enlaces clickeables hacia Yahoo Finance, permitiendo abrir la ficha del ticker directamente desde la UI o los reportes exportados.
### Changed
- Se unificó la tabla visible y el CSV descargable para compartir columnas, orden y formato de los enlaces, preservando la paridad entre ambas superficies.
### Fixed
- Se eliminaron las advertencias duplicadas que aparecían al regenerar el listado cuando coexistían datos de Yahoo y del stub.
### Documentation
- Se actualizaron las guías internas para describir los enlaces hacia Yahoo Finance y los criterios de sincronización entre la UI y el CSV exportable.

## [0.3.17] - 2025-10-01
### Added
- La estrategia Andy fue promovida a release estable tras validar los filtros financieros activos, el score normalizado y la telemetría espejo entre Yahoo y el stub, dejando documentada la cobertura manual que respalda el corte.
### Changed
- El stub de oportunidades ahora genera notas de telemetría con severidades `ℹ️/⚠️` según el tiempo de ejecución y deja trazabilidad de los descartes aplicados para facilitar la observabilidad durante los failovers. ([`application/screener/opportunities.py`](application/screener/opportunities.py))
- La UI y el backend leen la versión desde `pyproject.toml` mediante `shared.version.__version__`, evitando desfasajes entre las superficies y simplificando la sincronización de releases. ([`shared/version.py`](shared/version.py), [`ui/footer.py`](ui/footer.py))
### Documentation
- Se incorporó una guía de interpretación para la telemetría del barrido, con ejemplos de severidades y métricas monitoreadas tanto en el stub como en Yahoo. ([`README.md`](README.md#telemetría-del-barrido))
- README documenta la estrategia Andy lista para producción, enumerando:
  - los filtros financieros activos que se aplican en la tabla de oportunidades;
  - la normalización del `score_compuesto` en escala 0-100;
  - la telemetría compartida entre Yahoo Finance y el stub determinista, junto con los casos de failover;
  - la columna `Yahoo Finance Link`, ejemplificando cómo se pobla con universos live y con el stub.
- La guía de QA aclara que los 37 tickers del stub y los universos dinámicos comparten el mismo formato de enlace hacia Yahoo Finance para mantener paridad en las verificaciones.

## [0.3.16] - 2025-09-30
### Added
- Se amplió el stub de fundamentals para cubrir emisores adicionales y acompañar las nuevas validaciones del flujo beta.
- Prueba de integración que combina la selección de presets con el fallback al stub para validar el pipeline UI → controlador → screener bajo filtros reforzados.
### Changed
- Se endurecieron los filtros de fundamentals en la UI para reflejar los criterios reforzados del backend y mantener consistencia entre fuentes.
### Documentation
- README actualizado con la tabla completa del universo determinista de 19 emisores, explicando cómo el fallback replica la estrategia Andy durante los failovers.

## [0.3.15] - 2025-09-30
### Fixed
- El healthcheck del sidebar reutiliza `shared.ui.notes.format_note` para unificar la iconografía y el énfasis de los mensajes con el resto de la UI, evitando divergencias en la presentación de severidades. ([ui/health_sidebar.py](ui/health_sidebar.py))
### Tests
- Documentado el procedimiento para habilitar `pytest -m live_yahoo` mediante la variable `RUN_LIVE_YF` y advertir sobre su naturaleza no determinista. ([README.md](README.md#pruebas))
### Documentation
- Documentadas las severidades soportadas por `shared.ui.notes.format_note`, sus prefijos (⚠️/ℹ️/✅/❌) y el helper compartido para mantener mensajes consistentes en la UI. ([README.md](README.md#notas-del-listado-y-severidades), [tests/shared/test_notes.py](tests/shared/test_notes.py))

## [3.0.1]
### Changed
- El `score_compuesto` ahora se normaliza en escala 0-100 y se filtra automáticamente usando el umbral configurable `MIN_SCORE_THRESHOLD` (80 por defecto) para reducir ruido en los resultados de la pestaña beta.
- El listado final de oportunidades respeta el límite configurable `MAX_RESULTS` (20 por defecto), manteniendo la tabla acotada incluso cuando Yahoo Finance devuelve universos extensos.

### UI
- La cabecera de "Empresas con oportunidad" indica cuándo se aplican el umbral mínimo y el recorte del top N, explicando al usuario por qué ciertos tickers quedan fuera del informe.

## [0.3.14]
### Added
- Universo automático de oportunidades generado con `list_symbols_by_markets` y la configuración `OPPORTUNITIES_TARGET_MARKETS` para alinear los emisores con los mercados habilitados en cada sesión.
- Nuevos filtros en el screener: `min_eps_growth`, `min_buyback`, selector de sectores y un toggle para indicadores técnicos, que permiten ajustar dinámicamente la priorización de emisores.
- Caption de fuente visible en la UI de oportunidades para dejar claro el origen de los datos mostrados.

## [0.3.13] - 2025-09-30
### Changed
- La leyenda en la pestaña beta ahora destaca dinámicamente si los datos provienen de Yahoo Finance o del stub local, evitando confusiones durante los failovers.
- Se diferencian explícitamente las captions de Yahoo y del stub para que cada flujo muestre su fuente en el encabezado correspondiente.

### Tests
- Se actualizaron las pruebas de UI para validar la nueva diferenciación de captions entre Yahoo y el stub.

## [0.3.12] - 2025-09-29
### Fixed
- Se repararon las pruebas de `shared.settings` para que consuman los TTL y alias directamente desde la configuración compartida.
### Tests
- La suite de CI recuperó su estabilidad al eliminar los falsos negativos que provocaba la discrepancia en los tests de configuración.

## [0.3.11] - 2025-10-01
### Fixed
- Se repararon los tests de la pestaña beta para alinear las expectativas con el flujo visible en la UI.
### Changed
- Toda visualización de versión ahora se alimenta dinámicamente desde `pyproject.toml`, evitando desfasajes entre backend y UI.
- Se maneja explícitamente el feature flag de la pestaña beta para controlar su activación sin efectos secundarios.

## [0.3.10] - 2025-09-30
### Fixed
- Se corrigió el `ImportError` que se disparaba al inicializar los módulos de Yahoo Finance en entornos sin dependencias opcionales.
### Changed
- Los TTL por defecto de Yahoo Finance ahora se aplican automáticamente cuando no hay configuración explícita, permitiendo reutilizar cachés sin sobrecostos manuales.

## [0.3.9] - 2025-09-29
### Changed
- Los filtros de payout ratio, racha de dividendos y CAGR mínima ahora se aplican
  también en el screener de Yahoo para mantener una experiencia consistente con
  el stub local.
- Refactorización de `_apply_filters_and_finalize` para compartir la lógica de
  filtrado entre la integración de Yahoo Finance y el stub de respaldo.
### Tests
- Refuerzo de pruebas que cubren el filtrado compartido y la alineación de
  resultados entre ambas fuentes de datos.

## [0.3.8] - 2025-09-29
### Added
- Integración con Yahoo Finance para descargar históricos, indicadores técnicos y
  métricas fundamentales visibles en la pestaña de portafolio.
- Nuevos paneles con métricas fundamentales y ranking ESG del portafolio basados
  en los datos enriquecidos de Yahoo Finance.
### Changed
- Caché configurable para las consultas de Yahoo Finance mediante los TTL
  `CACHE_TTL_YF_*`, documentados en la configuración.
### Fixed
- Fallback automático al stub `infrastructure/cache/ta_fallback.csv` cuando la
  API de Yahoo Finance devuelve errores, con trazabilidad en el healthcheck.

## [0.3.7] - 2025-09-28
### Added
- Se agregó la pestaña beta de "Empresas con oportunidad" junto con su stub inicial para explorar la integración futura.

## [0.3.6] - 2025-09-17
### Removed
- Se eliminó la referencia obsoleta a `TimeProvider.now().moment` para evitar invocaciones inexistentes.
### Fixed
- Se corrigió el uso de `bearer_time` asegurando que utilice la clave actualizada.

## [0.3.5] - 2025-09-17
### Fixed
- Se configuró `fileWatcherType = "poll"` en Streamlit para evitar bloqueos del recargador
  en entornos con sistemas de archivos basados en red.

## [0.3.4] - 2025-09-17
### Fixed
- Se corrigió la incompatibilidad aware/naive al comparar las marcas de tiempo.
- Se actualizó `bearer_time` a naive en el cliente legacy para alinear el formato de fechas.

## [0.3.3] - 2025-09-21
### Fixed
- Se corrigió `shared.time_provider.TimeProvider` para garantizar que los timestamps y objetos `datetime`
  generados compartan la misma zona horaria y formato.
### Changed
- Se unificó la API de `TimeProvider` documentando explícitamente `now()` y `now_datetime()` para
  elegir entre cadena formateada u objeto `datetime` según la necesidad.

## [0.3.2] - 2025-09-20
### Changed
- Se unificó el manejo de hora mediante `shared.time_provider.TimeProvider` para mantener
  timestamps consistentes en formato `YYYY-MM-DD HH:MM:SS` (UTC-3).

## [0.3.1] - 2025-09-19
### Changed
- El healthcheck del sidebar ahora expone la versión actual de la aplicación y se movió al final para concentrar en un único bloque el estado de los servicios monitoreados.

## [0.3.0] - 2025-09-18
### Added
- El bloque de seguridad del login ahora muestra dinámicamente la versión actual de la aplicación.

## [0.2.1] - 2025-09-17
### Added
- Se incorporó un timestamp argentino en el footer para reflejar la hora local
  de manera consistente.

## [0.2.0] - 2025-09-16
### Added
- Centralized cache TTL configuration in `shared/settings` and documented the
  new environment keys for quote and FX caches.
- Added a sidebar health-check indicator so operators can quickly confirm
  downstream service availability from the navigation.
- Jerarquía de errores compartida (PR1) para estandarizar cómo controllers y
  services reportan incidencias recuperables.
### Changed
- Refactored the Streamlit session and routing logic to reuse authentication and
  page-loading flows between the main application and auxiliary entry points.
### Fixed
- Successful login now marks the session as authenticated to access the main page.
- Fixed: los paneles ahora se recargan automáticamente después de logout/login sin requerir refresco manual.
- Se corrigieron los tests de logout para reflejar la nueva firma y el comportamiento de la función.
- Se corrigieron pruebas fallidas en ta_service, portfolio_controller y
  portfolio_service_utils para alinear expectativas de tests con la
  implementación real.
- Deployment stable on Streamlit Cloud.

### Security
- Removed passwords from `session_state`; authentication now relies solely on local variables and tokens.

### Removed
- Removed deprecated `use_container_width` parameter (Streamlit ≥ 1.30).

### Tests
- Nuevas pruebas de TTL, health sidebar y propagación de errores (PR2).

## [2025-09-13]
### Tests
- Se agregaron pruebas de cobertura para UI, controllers, servicios, application, infrastructure y shared.

