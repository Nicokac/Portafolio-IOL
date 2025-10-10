# Adaptive Learning Overview (v0.5.2)

La release **0.5.2** introduce un motor de aprendizaje adaptativo que ajusta las predicciones sectoriales según el error histórico.
Este documento resume la arquitectura, los datos involucrados y la persistencia del estado.

## Componentes principales

- **`application.adaptive_predictive_service`**
  - Normaliza predicciones y retornos reales por sector, calculando un error relativo `(predicted - actual) / |actual|`.
  - Aplica un suavizado exponencial (EMA) sobre la matriz de errores para derivar ajustes β (*β-shift*) y recalcular las correlaciones adaptativas.
  - Persiste el estado (`AdaptiveModelState`) y la última matriz de correlaciones en caché con TTL de **12 horas** (`CacheService` namespace `adaptive_predictive`).
  - Expone `simulate_adaptive_forecast` para iterar históricos y medir **MAE**, **RMSE** y **bias**, comparando la predicción original vs. la ajustada.
  - Ofrece `prepare_adaptive_history` (históricos reales desde backtests) y `generate_synthetic_history` (cuando no hay datos), garantizando una inicialización determinística.

- **`ui/charts/correlation_matrix.py`**
  - Construye matrices histórica, rolling y adaptativa con anotaciones βΔ sobre la diagonal.
  - Utiliza la paleta activa para mantener consistencia cromática y resalta el ajuste adaptativo.

- **`ui/tabs/recommendations.py`**
  - Integra un tab "Correlaciones sectoriales" con resumen de β promedio, correlación media y dispersión σ.
  - El insight automático incorpora los valores adaptativos (β-shift promedio y correlación dinámica).
  - `_render_for_test()` genera datos sintéticos para validar el flujo completo sin depender de APIs externas.

- **`tests/application/test_adaptive_predictive_service.py`**
  - Verifica la actualización incremental del estado y el TTL de 12h.
  - Asegura que el motor adaptativo reduzca MAE/RMSE frente a la predicción base.
  - Comprueba la persistencia del estado entre simulaciones y nuevas actualizaciones.

## Flujo de datos y persistencia

1. El servicio recibe predicciones (`predicted_return`) y retornos observados (`actual_return`) por sector.
2. Se normaliza el error relativo para evitar magnitudes extremas (limitado a ±5).
3. El historial se guarda en `AdaptiveModelState.history` (máx. 720 registros), permitiendo derivar correlaciones con EMA.
4. Las matrices y el estado se almacenan en caché con TTL de 12h, reutilizadas por la pestaña de recomendaciones y las simulaciones offline.
5. En ausencia de históricos reales, `generate_synthetic_history` produce una serie determinística por sector para mantener consistencia visual.

## Métricas reportadas

- **MAE adaptativo** y delta vs. MAE original.
- **RMSE adaptativo** y delta vs. RMSE original.
- **Bias** (promedio del error ajustado) con delta frente al bias base.
- **β-shift promedio**, correlación media y dispersión sectorial σ.

Estas métricas se muestran en la pestaña "Correlaciones sectoriales" y alimentan el insight automático para reflejar la dinámica del modelo adaptativo.
