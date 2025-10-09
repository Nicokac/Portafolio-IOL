# Roadmap de Evolución — Módulo de Recomendaciones e Insight Automático

## Resumen ejecutivo
El módulo de Recomendaciones cuenta actualmente con tres modos de generación de cartera, visualizaciones interactivas, panel de insight automático y simulador de impacto. Para evolucionarlo hacia un motor de inversión inteligente, se proponen iteraciones enfocadas en trazabilidad de decisiones, personalización persistente y capacidades de análisis histórico, manteniendo la cohesión del dashboard y la validación offline.

## v0.4.3 — Exportabilidad y transparencia inmediata
**Objetivo principal:** Facilitar el consumo externo de las recomendaciones y enriquecer la interpretación de cada sugerencia.

| Frente | Funcionalidades nuevas o extendidas | Archivos/módulos afectados | Cambios UI | Testing/documentación |
| --- | --- | --- | --- | --- |
| Exportación | Descarga de CSV/XLSX con tabla de recomendaciones, métricas agregadas e insight. | `ui/tabs/recommendations.py`, utilidades nuevas en `services` o `application`. | Botones de descarga integrados bajo la tabla, siguiendo estilo del dashboard. | Tests unitarios de generación de archivos (paths temporales), prueba manual `--mock-data`. |
| Explicabilidad | Mostrar racionales ampliados por activo y métricas de aporte a retorno/beta. | `application/recommendation_service.py`, plantillas de racionales. | Sección expandible en la tabla con tooltips o acordeones. | Ajustar fixtures en `tests/application/test_recommendation_service.py`; captura de pantalla de Streamlit. |
| Insight | Incorporar promedio de sector/industria dominante dentro del insight. | `ui/tabs/recommendations.py`. | Texto enriquecido con etiquetas de sector. | Test de render por script (`_render_for_test`). |
| Calidad | Actualizar versión a v0.4.3, changelog y README con exportación y explicabilidad. | `shared/version.py`, `pyproject.toml`, `CHANGELOG.md`, `README.md`. | N/A | Validación de linters/formatters si aplica; documentación revisada. |

## v0.4.4 — Personalización persistente y benchmarks dinámicos
**Objetivo principal:** Ajustar recomendaciones al perfil del inversor y compararlas con referencias de mercado.

| Frente | Funcionalidades nuevas o extendidas | Archivos/módulos afectados | Cambios UI | Testing/documentación |
| --- | --- | --- | --- | --- |
| Perfil inversor | Persistir modo preferido, tolerancia a riesgo y horizonte en `session_state` y almacenamiento local cifrado (p. ej. `config.json`). | `controllers/portfolio/portfolio.py`, `ui/tabs/recommendations.py`, módulo de configuración. | Controles de preferencias en un subpanel, con badges que indiquen perfil aplicado. | Tests unitarios para serialización/deserialización; prueba de rehidratación por script. |
| Benchmarking | Calcular comparativos frente a índices (Merval, S&P 500, bonos). Incorporar diferencias de retorno y beta. | Nuevos servicios en `application/benchmark_service.py`; actualizaciones en `recommendation_service`. | Bloque visual con mini cards o gráfico de líneas comparando allocation simulada vs benchmark. | Tests de integración para cálculos benchmark; validación visual en `--mock-data`. |
| Telemetría básica | Registrar eventos de uso (modo seleccionado, exportaciones) en logs estructurados. | `controllers/portfolio/portfolio.py`, utilidades de logging. | Sin cambios visuales, pero incluir consentimiento en README. | Tests de logging (captura de logs); revisión manual de archivos generados. |
| Calidad | Versión v0.4.4 y documentación ampliada con perfil/benchmark. | `shared/version.py`, `pyproject.toml`, `CHANGELOG.md`, `README.md`, posibles FAQs en `docs/`. | N/A | Actualizar changelog, guías de usuario y capturas. |

## v0.5.0 — Motor de inversión inteligente con análisis histórico
**Objetivo principal:** Consolidar un flujo integral que justifique las decisiones de inversión, evalúe desempeño histórico y soporte validación offline robusta.

| Frente | Funcionalidades nuevas o extendidas | Archivos/módulos afectados | Cambios UI | Testing/documentación |
| --- | --- | --- | --- | --- |
| Análisis histórico | Modo que simule la evolución de las recomendaciones en periodos pasados (backtesting ligero). | Nuevos módulos `application/backtesting_service.py`, integración en `portfolio_service`. | Pestaña secundaria o sección dentro de recomendaciones con gráficos de series temporales. | Tests unitarios con datos sintéticos; pruebas de regresión offline. |
| Motor de decisión explicable | Añadir trazas de cálculo (feature importance, pesos sectoriales) y generar reporte descargable en PDF/Markdown. | `application/recommendation_service.py`, nuevas utilidades en `docs/`. | Panel de “Explicación detallada” con tabs (Riesgo, Retorno, Contribución). | Tests de consistencia de trazas; validación manual de reporte exportado. |
| Caching avanzado | Cachear datos de mercado y simulaciones con invalidación basada en timestamp; mejorar `_render_for_test` para cargar fixtures. | `services/cache.py` (nuevo), ajustes en `controllers` y `ui`. | Indicador visual de tiempo de última actualización. | Tests de expiración de cache, ejecución de scripts offline. |
| Validación offline mejorada | Paquete de datos de ejemplo (fixtures) para ejecutar dashboard sin conexión: portfolios, benchmarks, históricos. | `docs/fixtures/`, scripts en `scripts/generate_mock_data.py`. | Selector de dataset mock en UI inicial. | Tests automatizados que verifiquen carga de fixtures; guía en README. |
| Calidad | Versión v0.5.0 con documentación completa y checklist de liberación. | `shared/version.py`, `pyproject.toml`, `CHANGELOG.md`, `README.md`, documentación técnica en `docs/`. | N/A | Cobertura recomendada >85 % en servicios de recomendación/simulación; validación visual Streamlit con screenshots para QA. |

## Recomendaciones generales de calidad
- **Cobertura de pruebas:** Mantener y ampliar las pruebas unitarias en `tests/application` y añadir pruebas de integración en `tests/ui` para scripts de render. Para v0.5.0, apuntar a >85 % en módulos de recomendación, simulación y backtesting.
- **Validación visual:** Capturar screenshots por versión usando `streamlit run app.py -- --mock-data` con fixtures offline, especialmente tras cambios de layout o nuevos paneles.
- **Documentación:** Actualizar `CHANGELOG.md`, `README.md` y crear fichas en `docs/` para nuevas funcionalidades (exportaciones, perfiles, backtesting). Añadir notas de requisitos (conectividad/API) y FAQs.
- **Versionado:** Sincronizar `shared/version.py` y `pyproject.toml` en cada release; incluir notas de riesgos conocidos y pendientes en el changelog.
- **Telemetría y monitoreo:** A partir de v0.4.4, definir esquema de logs estructurados y revisar periódicamente su almacenamiento.

Este roadmap proporciona hitos claros para evolucionar el módulo hacia un motor de inversión inteligente, priorizando usabilidad, precisión analítica y trazabilidad de las recomendaciones.
