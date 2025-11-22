# Estrategia para reducir el tiempo de arranque manteniendo la precarga científica

## 1. Objetivo
Reducir el tiempo de arranque visible por debajo de 1 segundo sin sacrificar la disponibilidad temprana de `pandas`, `plotly` y `statsmodels` para el dashboard posterior al login.

## 2. Principios
- El hilo `preload_worker` seguirá siendo el único responsable de cargar librerías pesadas.
- El login no debe bloquearse por importaciones opcionales; sólo se cargan dependencias estrictamente necesarias para la autenticación.
- La precarga científica debe completarse antes de que se abran las pantallas de análisis en el dashboard, pero puede ocurrir tras la autenticación del usuario.

## 3. Plan propuesto
1. **Dividir el arranque en dos fases**:
   - **Fase A (pre-login)**: arranque mínimo que inicializa configuración, logging, y dependencias ligeras para la pantalla de login. Lanzar el hilo `preload_worker` en estado "pausado" sin ejecutar importaciones.
   - **Fase B (post-login)**: cuando se valida el primer usuario, reanudar el worker para que ejecute las importaciones de `pandas`, `plotly` y `statsmodels`, y extenderlo con métricas para reportar la duración por módulo.
2. **Sincronización con el dashboard**: las vistas que dependen de estas librerías consultan un flag o `Future` que indica si la precarga terminó. Si el flag está activo, continúan sin bloqueos; en caso contrario, muestran un spinner breve y se suscriben al evento de finalización.
3. **Lazy imports para módulos adicionales**:
   - `application.predictive_service` y `controllers.portfolio.charts` permanecen fuera de la precarga por defecto y ahora se consumen vía `shared.lazy_import.lazy_import` para evitar importarlos antes de que se abra la pestaña correspondiente.
   - Mantener también fuera de la precarga módulos secundarios que sólo se usan en vistas concretas (por ejemplo, `controllers.recommendations_controller`, `services.system_diagnostics`, `ui.panels.diagnostics`). Estos módulos dependen del caché predictivo y de Plotly, por lo que conviene importarlos únicamente cuando el usuario abre la pestaña de recomendaciones o el panel de diagnóstico.
   - Cuando alguna pantalla requiera estos módulos, usar `shared.lazy_import.lazy_import(module_name)` dentro de la función específica y cachear el resultado.
4. **Snapshot opcional de bytecode**: habilitar `PYTHONDONTWRITEBYTECODE=0` y ejecutar un script de warm-up durante el despliegue para que se generen archivos `.pyc` en el contenedor. Esto reduce el coste de importación inicial sin modificar el flujo de ejecución.
5. **Medición continua**: añadir métricas de arranque (por ejemplo, logging estructurado con timestamps) para validar que la fase A permanece < 500 ms y que la fase B concluye < 1 s tras el login.

## 4. Riesgos y mitigaciones
- **Primer login más lento si el worker se despierta demasiado tarde**: se puede disparar la reanudación del worker al renderizar la pantalla de login (de manera asíncrona) para que la precarga esté lista cuando el usuario termine de autenticarse.
- **Condiciones de carrera**: usar locks o eventos del módulo `threading` para coordinar el estado del worker.
- **Módulos nuevos en el dashboard**: mantener una lista configurable para añadir futuras librerías a la precarga sin tocar el código.

## 5. Próximos pasos
1. Refactorizar el `preload_worker` para aceptar un estado "paused" y una señal de reanudación.
2. Implementar el flag/Future compartido para que las vistas sepan si la precarga terminó.
3. Añadir script de warm-up en el pipeline de despliegue para generar `.pyc`.
4. Medir nuevamente el arranque y documentar los resultados.
