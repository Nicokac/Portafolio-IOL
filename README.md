# Portafolio IOL

Aplicación Streamlit para consultar y analizar carteras de inversión en IOL.

> Nota: todos los timestamps visibles provienen de `shared.time_provider.TimeProvider` y se muestran
> en formato `YYYY-MM-DD HH:MM:SS` (UTC-3). El footer de la aplicación se actualiza en cada
> renderizado con la hora de Argentina.

## Quick-start (release 0.3.27.1)

La versión **0.3.27.1** fortifica la capa de APIs y los mecanismos de resiliencia: centraliza la configuración de timeouts y backoff, monitorea la disponibilidad de cada proveedor y publica la salud consolidada en todas las superficies visibles.
## Quick-start (release 0.3.27.1 — 2025-11-07)

La versión **0.3.27.1** destaca cuatro ejes principales:
- El **header del login y del dashboard** reutiliza el helper centralizado de versión para mostrar `Versión 0.3.27.1` junto con la hora de Argentina y una insignia del estado actual de las APIs configuradas.
- El **health sidebar** incorpora un monitor de resiliencia con badges de cache hit/miss, ratios de fallback y buckets de latencia sincronizados con los contadores globales, resaltando la última respuesta exitosa por proveedor.
- El menú **⚙️ Acciones** del dashboard publica notificaciones internas (`st.toast`) cuando la aplicación se recupera de un fallback y cuando los proveedores externos vuelven a estar disponibles, dejando trazabilidad para los analistas.
- Las notas del screening y el resumen de riesgo muestran la secuencia de degradación aplicada (proveedor primario, secundario, fallback estático) junto con los identificadores de la nueva telemetría consolidada.

Sigue estos pasos para reproducir el flujo completo y validar las novedades clave:

### Ejemplo completo

1. **Instala dependencias.**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Para entornos de desarrollo agrega `requirements-dev.txt` si necesitas las herramientas de QA.
2. **Levanta la aplicación y valida el dashboard renovado.** Con el entorno activado ejecuta:
   ```bash
   streamlit run app.py
   ```
   La cabecera del sidebar y el banner del login mostrarán el número de versión `0.3.27.1` junto con
   el timestamp generado por `TimeProvider`, confirmando que la actualización quedó aplicada. El menú
   **⚙️ Acciones** mostrará un toast "Proveedor primario restablecido" cuando ejecutes **⟳ Refrescar**
   tras un fallback, dejando registro inmediato de la recuperación documentada en esta release.
   Abre la pestaña **Empresas con oportunidad**, activa la casilla **Mostrar resumen del
   screening** y ejecuta una búsqueda con los datos stub incluidos para ver las tarjetas de KPIs:
   universo analizado, candidatos finales y sectores activos (con deltas de descartes y tiempos de
   cómputo).
3. **Lanza un screening con presets personalizados y revisa la telemetría ampliada.** Al mismo
   tiempo, el mini-dashboard superior renderizará tarjetas con el valor total de la cartera, la
   variación diaria y el cash disponible usando los datos stub incluidos.
   - Abre la pestaña **Empresas con oportunidad** y selecciona `Perfil recomendado → Crear preset`.
   - Completa los filtros (score mínimo, payout, racha, sectores, indicadores técnicos) y presiona
     **Guardar preset**. La UI confirmará con un toast "Preset guardado" y el nuevo preset quedará
     disponible en el selector.
   - El botón **Comparar presets** despliega una vista dividida que muestra, a la izquierda, el
     preset base y, a la derecha, el preset recién guardado. Cada columna lista los filtros con
     resaltados verdes/rojos según subidas o bajadas respecto del original, facilitando la revisión
     antes de lanzar el barrido definitivo.
   - Pulsa **Ejecutar screening** para correr con el preset actual. Si repites exactamente los mismos
     filtros durante la sesión, la telemetría enriquecida del health sidebar mostrará el último modo
     (hit/miss), el ahorro promedio frente a la caché, la cronología completa de screenings, los
     contadores sincronizados de aciertos/fallback y la procedencia más resiliente disponible.
4. **Valida el fallback multinivel de datos macro.** Arranca con `MACRO_API_PROVIDER="fred,worldbank"` y deja sin definir `FRED_API_KEY` para forzar el salto al segundo proveedor. Declara una serie World Bank (`WORLD_BANK_SECTOR_SERIES='{"Energy": "EG.USE.PCAP.KG.OE"}'`) y ejecuta un screening: las notas mostrarán "Datos macro (World Bank)" y el health sidebar actualizará los contadores de éxito, fallbacks y buckets de latencia para ese proveedor. Si luego quitas también la clave de World Bank o las series configuradas, la secuencia finalizará en el fallback estático y registrará el motivo en la misma telemetría.

### Validar el fallback jerárquico desde el health sidebar

1. Abre el panel lateral **Salud del sistema** y localiza el bloque **Resiliencia de proveedores**. La release 0.3.27.1 preserva la
   última secuencia de degradación en `st.session_state`, por lo que verás los mismos datos incluso después de un `Rerun`.
2. Ejecuta nuevamente **⟳ Refrescar** desde el menú **⚙️ Acciones** y observa el timeline: debe listar `primario → secundario →
   fallback` con la marca temporal de cada intento y la insignia que indica si la recuperación fue completa o parcial.
3. En la sección **Último proveedor exitoso** verifica que el identificador coincida con las notas del screening y que la latencia
   agregada conserve el valor reportado durante la degradación. Si fuerzas un error manual (por ejemplo, quitando todas las claves),
   el bloque mostrará `Fallback estático` junto con el mensaje explicativo persistente.
4. Consulta la guía de soporte para escenarios extendidos y flujos de depuración en
   [docs/troubleshooting.md#fallback-jerarquico-desde-health-sidebar](docs/troubleshooting.md#fallback-jerarquico-desde-health-sidebar).

**Notas clave del flujo**

- El mini-dashboard inicial resume valor de la cartera, variación diaria y cash disponible con formato de tarjetas, y se actualiza automáticamente después de cada screening.
- El toast "Preset guardado" deja visible el preset recién creado dentro del selector para reutilizarlo en corridas posteriores.
- Las notificaciones internas del menú **⚙️ Acciones** confirman tanto los refrescos como los cierres de sesión exitosos, y ahora también anuncian la recuperación de los proveedores externos principales.
- La comparación de presets presenta dos columnas paralelas con indicadores verdes/rojos que señalan qué filtros fueron ajustados antes de confirmar la ejecución definitiva.
- El bloque de telemetría enriquecida marca explícitamente los *cache hits*, diferencia el tiempo invertido en descarga remota vs. normalización y calcula el ahorro neto de la caché cooperativa durante la sesión.

**Resiliencia de APIs (0.3.27.1).** Cuando guardas un preset, la aplicación persiste la combinación de filtros y el resultado del último screening asociado. Al relanzarlo, la telemetría agrega la procedencia del dato (primario, secundario, fallback estático) y clasifica la recuperación según la estrategia aplicada:

- Si los filtros no cambiaron y el proveedor primario respondió, se muestra una insignia "⚡ Resultado servido desde caché" en la tabla y la telemetría reduce el runtime (<1 s en stub, ≈2 s en Yahoo) al evitar descargas redundantes, resaltando en verde el ahorro neto respecto de la corrida anterior.
- Si el proveedor primario falla pero existe un secundario configurado, la UI muestra "🛡️ Fallback activado" y el health sidebar registra el tiempo adicional invertido en la degradación controlada.
- Cuando todos los proveedores remotos fallan, la secuencia finaliza en el fallback estático con la leyenda "📦 Snapshot de contingencia" y el contador de resiliencia incrementa el total de recuperaciones exitosas sin datos frescos.

Estas novedades convierten a la release 0.3.27.1 en la referencia para validar onboarding, telemetría y resiliencia multi-API: toda la UI recuerda la versión activa, expone KPIs agregados de disponibilidad en el health sidebar (incluyendo el resumen macro con World Bank) y los presets continúan recortando los tiempos de iteración al dejar a la vista el impacto de cada cambio.

## Configuración de claves API

La release 0.3.27.1 consolida la carga de credenciales desde `config.json`, variables de entorno o `streamlit secrets`. Antes de
ejecutar la aplicación en modo live, define las claves según el proveedor habilitado. Si una clave falta, el health sidebar registrará
el evento como `disabled` y la degradación continuará con el siguiente proveedor disponible.

### Variables mínimas por proveedor

- **Alpha Vantage** (`ALPHA_VANTAGE_API_KEY`): requerida para los históricos OHLC en `services.ohlc_adapter`. Puedes opcionalmente
  ajustar `ALPHA_VANTAGE_BASE_URL` para entornos de prueba.
- **Polygon** (`POLYGON_API_KEY`): habilita los precios intradía y los agregados en vivo; respeta el orden de `MACRO_API_PROVIDER`
  cuando figura como proveedor secundario de mercado. Usa `POLYGON_BASE_URL` para entornos aislados.
- **Financial Modeling Prep (FMP)** (`FMP_API_KEY`): alimenta ratios fundamentales y `application.ta_service`. Opcionalmente regula
  `FMP_TIMEOUT` y `FMP_BASE_URL` para ensayos offline.
- **FRED** (`FRED_API_KEY` y `FRED_SECTOR_SERIES`): primer escalón de datos macro; sin esta clave el intento queda marcado como
  `disabled` y se recurre al siguiente proveedor configurado.
- **World Bank** (`WORLD_BANK_API_KEY` y `WORLD_BANK_SECTOR_SERIES`): respaldo macro tras FRED; cuando falta se documenta como
  `unavailable` y se activa el fallback estático configurado en `MACRO_SECTOR_FALLBACK`.

Guarda las claves sensibles en `.env` o en `~/.streamlit/secrets.toml` e integra su carga en tus pipelines de CI/CD (ver
[docs/troubleshooting.md#claves-api](docs/troubleshooting.md#claves-api) para detalles y validaciones automatizadas).

## Persistencia de favoritos

La lista de símbolos marcados como favoritos se comparte entre pestañas y ahora también se
sincroniza con disco para mantenerla entre sesiones. Por defecto se serializa como un archivo JSON
en ``~/.portafolio_iol/favorites.json`` con la siguiente estructura:

```json
[
  "GGAL",
  "PAMP",
  "TXAR"
]
```

- El archivo se crea automáticamente la primera vez que marcás un símbolo como favorito. Cada
  entrada es una cadena en mayúsculas.
- Si el archivo está corrupto o no se puede leer, la aplicación continúa funcionando con una lista
  vacía y muestra el error en la sesión actual para que puedas depurarlo.
- Podés borrar el archivo para reiniciar la lista; se volverá a generar cuando agregues un nuevo
  favorito.

## Documentación

- [Guía de troubleshooting](docs/troubleshooting.md)
- [Guía de pruebas](docs/testing.md)
- [Integración en CI/CD](docs/testing.md#integración-en-cicd): ejemplos de pipelines para instalar dependencias,
  forzar los mocks (`RUN_LIVE_YF=0`) y ejecutar `pytest --maxfail=1 --disable-warnings -q`. Los jobs adjuntan
  el directorio `htmlcov`; descárgalo desde los artefactos del pipeline y abre `index.html` para revisar la
  cobertura en detalle.

## Uso del proveedor de tiempo

Para generar fechas consistentes en toda la aplicación, importa la clase `TimeProvider`:

```python
from shared.time_provider import TimeProvider

timestamp = TimeProvider.now()          # "2025-09-21 10:15:42"
moment = TimeProvider.now_datetime()    # datetime consciente de zona (UTC-3)
```

- `TimeProvider.now()` devuelve la representación en texto lista para mostrar en la interfaz.
- `TimeProvider.now_datetime()` expone el mismo instante como un objeto `datetime` con zona horaria de Buenos Aires.

El método `TimeProvider.now_datetime()` retorna un `datetime` consciente de zona en UTC-3, lo que garantiza que el `tzinfo` se mantenga alineado con Buenos Aires para todo cálculo dentro de la aplicación. Algunos integradores —como el cliente legacy de IOL— deben convertirlo explícitamente a naive (`tzinfo=None`) para continuar siendo compatibles con librerías que no gestionan zonas horarias.

Ambos métodos apuntan al mismo reloj centralizado, por lo que los valores son intercambiables según si necesitas una cadena o un
`datetime` para cálculos adicionales.

Desde Streamlit 1.30 se reemplazó el parámetro `use_container_width` y se realizaron ajustes mínimos de diseño.

### Empresas con oportunidad (disponible de forma estable)

La pestaña ya se encuentra disponible de forma estable y en cada sesión combina:

- Tickers provistos manualmente por el usuario en la interfaz cuando existen; si no hay input manual, se utiliza `YahooFinanceClient.list_symbols_by_markets` parametrizada mediante la variable de entorno `OPPORTUNITIES_TARGET_MARKETS`.
- Un conjunto determinista de respaldo basado en el stub local (`run_screener_stub`) para garantizar resultados cuando no hay configuración externa ni datos remotos, o cuando Yahoo Finance no está disponible.

El stub local expone un universo determinista de 37 emisores que cubre múltiples sectores (Technology, Healthcare, Industrials, Financial Services, Consumer Defensive, Consumer Cyclical, Consumer, Financials, Utilities, Energy, Real Estate, Communication Services y Materials) con métricas fundamentales completas. Cada sector crítico —Technology, Energy, Industrials, Consumer, Healthcare, Financials, Utilities y Materials— cuenta con al menos tres emisores para ejercitar filtros exigentes sin perder diversidad. Las cifras se calibraron para que los filtros de payout, racha, CAGR, EPS, buybacks y fundamentals críticos dispongan siempre de datos consistentes y se puedan ejercitar escenarios complejos de QA aun cuando Yahoo Finance no esté disponible, incluso en esta fase estable.

La columna `Yahoo Finance Link` documenta el origen de cada símbolo con la URL `https://finance.yahoo.com/quote/<ticker>`. En universos dinámicos descargados de Yahoo la columna reutiliza directamente el *slug* oficial (por ejemplo, `AAPL`), mientras que el stub determinista sintetiza enlaces equivalentes para sus 37 emisores (`UTLX`, `FNCL1`, etc.) manteniendo el mismo formato. Esto permite a QA y a los integradores validar rápidamente la procedencia sin importar si el listado proviene de datos live o del fallback. A partir de la release actual, el listado añade la columna `Score` para dejar a la vista el puntaje compuesto que define el orden del ranking y, cuando corresponde, explicita el preset o filtro destacado que disparó la selección.

| Ticker | Sector | Payout % | Racha (años) | CAGR % | EPS trailing | EPS forward | Buyback % | Market cap (M USD) | P/E | Revenue % | Score | Filtro destacado | Yahoo Finance Link |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AAPL | Technology | 18.5 | 12 | 14.2 | 6.1 | 6.6 | 1.8 | 2,800,000 | 30.2 | 7.4 | 88 | Crecimiento balanceado ≥85 | [Ver ficha](https://finance.yahoo.com/quote/AAPL) |
| MSFT | Technology | 28.3 | 20 | 11.7 | 9.2 | 9.8 | 1.1 | 2,450,000 | 33.5 | 14.8 | 90 | Growth + buybacks ≥80 | [Ver ficha](https://finance.yahoo.com/quote/MSFT) |
| KO | Consumer Defensive | 73.0 | 61 | 7.5 | 2.3 | 2.4 | 0.3 | 260,000 | 24.7 | 4.3 | 84 | Dividendos defensivos ≥80 | [Ver ficha](https://finance.yahoo.com/quote/KO) |
| NEE | Utilities | 56.2 | 28 | 10.8 | 3.1 | 3.5 | 0.0 | 160,000 | 25.7 | 7.1 | 82 | Dividendos defensivos ≥80 | [Ver ficha](https://finance.yahoo.com/quote/NEE) |
| UTLX | Utilities | 61.5 | 19 | 6.7 | 3.1 | 3.3 | 0.0 | 58,600 | 19.2 | 4.6 | 86 | Stub estable ≥80 | [Ver ficha](https://finance.yahoo.com/quote/UTLX) |
| ENRGX | Energy | 38.7 | 18 | 5.6 | 5.5 | 5.8 | 1.9 | 95,000 | 13.6 | 8.9 | 83 | Recompras agresivas ≥80 | [Ver ficha](https://finance.yahoo.com/quote/ENRGX) |

El muestreo superior refleja la combinación live + fallback que hoy ve la UI: los símbolos clásicos (`AAPL`, `MSFT`, `KO`, `NEE`) provienen de Yahoo, mientras que `UTLX` y `ENRGX` pertenecen al stub determinista y conservan las mismas métricas que en la versión estable anterior para garantizar reproducibilidad en QA.

El botón **"Descargar resultados (.csv)"** replica esta grilla y genera un archivo con las mismas columnas visibles en la UI (incluidos `score_compuesto`, el filtro aplicado y el enlace a Yahoo). Así se asegura paridad total entre lo que se analiza en pantalla y lo que se comparte para backtesting o QA, sin importar si la sesión proviene del origen `yahoo` o `stub`.

## Exportación de análisis enriquecido

La aplicación permite llevarte un paquete completo de métricas, rankings y visualizaciones del portafolio sin salir del dashboard o desde la línea de comandos si trabajás con snapshots persistidos.

### Desde el dashboard

1. Abrí la pestaña **📂 Portafolio** y desplegá el acordeón **📦 Exportar análisis enriquecido**.
2. Seleccioná las métricas que querés incluir en el reporte (valor total, P/L, cantidad de posiciones, etc.). Cada opción muestra una breve descripción para que identifiques rápidamente qué KPI estás incorporando.
3. Elegí los gráficos que se van a embeber en el Excel (por defecto se incluyen P/L Top N, composición por tipo, distribución valorizada, la evolución histórica y el mapa de calor por símbolo/tipo). Si Kaleido no está disponible la UI te lo indicará para que habilites la dependencia.
4. Activá o desactivá la exportación de rankings e historial y definí el límite de filas para cada ranking.
5. Descargá el ZIP con los CSV (`kpis.csv`, `positions.csv`, `history.csv`, `contribution_by_symbol.csv`, etc.) o el Excel enriquecido (`*_analisis.xlsx`) que incluye todas las tablas en hojas dedicadas y los gráficos renderizados como imágenes.

### Desde la línea de comandos

El script `scripts/export_analysis.py` procesa snapshots serializados en JSON (por ejemplo los generados por jobs batch o instrumentación de QA) y genera los mismos artefactos enriquecidos que la UI.

```bash
python scripts/export_analysis.py \
  --input .cache/portfolio_snapshots \
  --output ./exports/nocturno \
  --metrics total_value total_pl total_pl_pct positions symbols \
  --charts pl_top composition timeline heatmap \
  --limit 15
```

- El argumento `--metrics help` lista todos los KPIs disponibles; `--charts help` hace lo propio con los gráficos.
- Con `--formats csv` o `--formats excel` podés limitar la salida a un solo formato.
- Cada snapshot genera un subdirectorio dentro de `--output` con todos los CSV y, si corresponde, el Excel `analysis.xlsx`.
- Se adjunta además `summary.csv` en la raíz con los KPIs crudos (`raw_value`) de cada snapshot para facilitar comparaciones rápidas o integraciones en pipelines.

> Dependencias: asegurate de instalar `kaleido` y `XlsxWriter` (ambos incluidos en `requirements.txt`) para que el script pueda renderizar los gráficos y escribir el Excel correctamente.

Cada registro respeta los principios de la estrategia Andy: payout y P/E saludables, rachas y CAGR positivos, EPS forward por encima del trailing, buybacks y crecimiento de ingresos cuando corresponde. En la release actual, ese set determinista permite verificar que `score_compuesto` se mantenga estable tanto en modo `yahoo` como `stub`, sosteniendo la comparabilidad del ranking.

Durante los failovers la UI etiqueta el origen como `stub` y conserva las notas contextuales del caption principal. Los tests automatizados siguen apoyándose en este dataset extendido para validar diversidad sectorial, completitud de fundamentals y la presencia de la nueva columna `Score`.

#### Datos macro y sectoriales (FRED + fallback)

- La tabla incorpora la columna `macro_outlook`, alimentada por la API de [FRED](https://fred.stlouisfed.org/) cuando existe configuración válida. Cada celda combina el último valor publicado para la serie sectorial asociada y la fecha del dato (`1.75 (2023-09-01)`), facilitando la lectura del contexto macro sin abandonar la grilla del screening.
- Para habilitar la integración se deben definir las siguientes variables (vía `.env`, `streamlit secrets` o `config.json`):

  ```bash
  export MACRO_API_PROVIDER="fred,worldbank"
  export FRED_API_KEY="<tu-clave>"
  export FRED_SECTOR_SERIES='{"Technology": "IPN31152N", "Finance": "IPN52300N"}'
  # Opcional: tuning avanzado
  export FRED_API_BASE_URL="https://api.stlouisfed.org/fred"
  export FRED_API_RATE_LIMIT_PER_MINUTE=120
  export WORLD_BANK_API_KEY="<tu-clave-wb>"
  export WORLD_BANK_SECTOR_SERIES='{"Energy": "EG.USE.PCAP.KG.OE"}'
  export WORLD_BANK_API_BASE_URL="https://api.worldbank.org/v2"
  export WORLD_BANK_API_RATE_LIMIT_PER_MINUTE=60
  export MACRO_SECTOR_FALLBACK='{"Technology": {"value": 2.1, "as_of": "2023-06-30"}}'
  ```

  - `FRED_SECTOR_SERIES` mapea el nombre del sector que aparece en el screener con el identificador de serie en FRED. Es sensible a los sectores retornados por Yahoo/stub, por lo que conviene mantener la misma capitalización mostrada en la tabla.
- `MACRO_SECTOR_FALLBACK` permite declarar valores estáticos (por sector) que se aplican automáticamente cuando la API externa no está disponible, cuando el proveedor configurado no es soportado o cuando falta alguna serie en la configuración.
- Flujo de failover: si la API devuelve errores, alcanza el límite de rate limiting o falta la clave, el controlador intenta poblar `macro_outlook` con los valores declarados en `MACRO_SECTOR_FALLBACK`. Cuando no hay fallback, la columna queda en blanco y se agrega una nota explicando la causa (`Datos macro no disponibles: FRED sin credenciales configuradas`). Todos los escenarios se registran en `services.health.record_macro_api_usage`, exponiendo en el healthcheck si el último intento fue exitoso, error o fallback.
- El rate limiting se maneja desde `infrastructure/macro/fred_client.py`, que serializa las llamadas según el umbral configurado (`FRED_API_RATE_LIMIT_PER_MINUTE`) y reutiliza el `User-Agent` global para respetar los términos de uso de FRED.

##### Escenarios de fallback macro (0.3.27.1)

1. **Secuencia `fred → worldbank → fallback`.** Con `MACRO_API_PROVIDER="fred,worldbank"` y sin `FRED_API_KEY`, el intento inicial queda marcado como `disabled`, el World Bank responde con `success` y la nota "Datos macro (World Bank)" deja registro de la latencia. El monitor de resiliencia del health sidebar incrementa los contadores de éxito, actualiza los buckets de latencia del proveedor secundario y agrega la insignia "Fallback cubierto".
2. **World Bank sin credenciales o series.** Si el segundo proveedor no puede inicializarse (sin `WORLD_BANK_API_KEY` o sin `WORLD_BANK_SECTOR_SERIES`), el intento se registra como `error` o `unavailable` y el fallback estático cierra la secuencia con el detalle correspondiente, incluyendo el identificador `contingency_snapshot` en la telemetría.
3. **Proveedor no soportado.** Cuando `MACRO_API_PROVIDER` apunta a valores fuera del set `fred/worldbank`, el controlador descarta la integración live y aplica el fallback estático si existe. El health sidebar deja el estado `disabled` con el detalle "proveedor no soportado" y dispara un toast de advertencia.
4. **Errores de API o rate limiting.** Ante un `MacroAPIError` (incluye timeouts y límites de FRED o del World Bank), la telemetría conserva la latencia que disparó el problema, agrega el código de error y los contadores globales incrementan tanto el `error` como el `fallback` resultante para visibilizar la resiliencia aplicada.
5. **Series faltantes u observaciones inválidas.** Cuando un proveedor responde sin datos válidos o no hay series configuradas para un sector activo, la nota lista los sectores faltantes (`macro_missing_series`), el fallback estático aporta el valor definitivo y el monitor etiqueta el evento como `partial_recovery`.

#### Telemetría del barrido

El panel muestra una nota de telemetría por cada barrido, tanto si la corrida proviene de Yahoo Finance como del stub local. El helper `shared.ui.notes.format_note` arma el texto en base a los campos reportados por cada origen y selecciona la severidad adecuada (`ℹ️` o `⚠️`) según los umbrales vigentes.

#### Caché del screening de oportunidades

- `controllers.opportunities.generate_opportunities_report` guarda en memoria el último resultado para cada combinación de filtros, tickers manuales y toggles críticos. Cuando el usuario repite una búsqueda con la misma configuración, la respuesta se obtiene desde caché y evita recalcular el screener completo.
- Un *cache hit* queda registrado en el nuevo bloque "🔎 Screening de oportunidades" del healthcheck lateral, que muestra tanto la duración de la lectura cacheada como la corrida completa previa para comparar la reducción de tiempos. En escenarios típicos de QA, la ejecución inicial ronda las decenas de milisegundos mientras que la respuesta cacheada se resuelve en el orden de 1 ms, dejando visible la mejora.
- Cualquier cambio en los filtros —por ejemplo, alternar el toggle de indicadores técnicos, ajustar umbrales numéricos o modificar el universo manual— invalida automáticamente la entrada, garantizando que las corridas posteriores utilicen los parámetros más recientes.

**Campos reportados**

- **Runtime (`elapsed` / `elapsed time`)**: segundos invertidos en la corrida completa, medidos desde la descarga hasta el post-procesamiento. Es el primer indicador para detectar degradaciones.
- **Universo inicial (`universe initial`)**: cantidad de símbolos recibidos antes de aplicar filtros; Yahoo lo informa con el universo crudo según los mercados solicitados, mientras que el stub siempre expone 37 emisores determinísticos.
- **Universo final (`universe` / `universe size`)**: tickers que sobreviven al filtrado; permite visualizar el recorte efectivo.
- **Ratios de descarte (`discarded`)**: descomposición porcentual entre descartes por fundamentals y por técnicos, útil para saber qué bloque necesita ajustes.
- **Fuente (`origin`)**: etiqueta visible (`yahoo` / `stub`) que coincide con el caption del listado para asegurar trazabilidad.
- **Score medio (`score_avg`)**: promedio del `score_compuesto` tras aplicar filtros; ayuda a detectar si el preset activo está elevando o relajando el umbral configurado.

**Ejemplos actualizados**

```
ℹ️ Yahoo • runtime: 5.8 s • universe initial: 142 • universe final: 128 • discarded: 8% fundamentals / 2% técnicos • score_avg: 86
ℹ️ Stub • runtime: 2.4 s • universe initial: 37 • universe final: 37 • discarded: 18% fundamentals / 10% técnicos • score_avg: 84
⚠️ Yahoo • runtime: 11.6 s • universe initial: 142 • universe final: 9 • discarded: 54% fundamentals / 34% técnicos • score_avg: 79
⚠️ Stub • runtime: 6.1 s • universe initial: 37 • universe final: 12 • discarded: 51% fundamentals / 17% técnicos • score_avg: 76
```

En condiciones saludables la nota se mantiene en severidad `ℹ️`. Cuando el runtime supera los límites esperados (≈3 s para el stub, 8–9 s para Yahoo), el universo final cae por debajo del umbral mínimo configurado o los ratios de descarte exceden el 35 % de manera sostenida, la severidad escala automáticamente a `⚠️` y se resalta en la UI.

**Guía rápida para QA y usuarios**

| Señal | Qué revisar | Acción sugerida |
| --- | --- | --- |
| `runtime > 3 s` (stub) o `> 9 s` (Yahoo) | Posibles problemas de IO, throttling o jobs en segundo plano. | Revisar logs y latencias externas antes de reintentar. |
| `universe final < 10` | Filtros demasiado agresivos o caída de datos en la fuente. | Relajar filtros temporalmente y validar la disponibilidad de Yahoo/stub. |
| `discarded fundamentals > 35 %` | Fundamentales incompletos para gran parte del universo. | Revisar los símbolos afectados; puede requerir recalibrar la caché o invalidar datos corruptos. |
| `discarded técnicos > 35 %` | Indicadores técnicos no disponibles. | Confirmar que el toggle de indicadores esté activo y que las series históricas se descarguen correctamente. |
| `score_avg < 80` (en presets exigentes) | Preset demasiado permisivo para la estrategia elegida. | Ajustar el slider de score o cambiar el preset recomendado. |

Las notas siempre incluyen los porcentajes de descarte fundamental y técnico. Cuando alguno de los dos no aplica, el stub reporta explícitamente `0%` para preservar la consistencia del formato y evitar falsos positivos en los tests automatizados. Los equipos de QA pueden apoyarse en estos indicadores para automatizar aserciones: por ejemplo, validar que en modo stub el universo final se mantenga en 37 con severidad `ℹ️` o que en pruebas de resiliencia la degradación quede marcada con `⚠️`.

Adicionalmente, las guías de QA asumen que tanto los 37 tickers deterministas del stub como los universos dinámicos de Yahoo exponen la columna `Yahoo Finance Link` con el patrón `https://finance.yahoo.com/quote/<ticker>`. Cualquier verificación de UI o fixtures debe asegurar que la URL se construya con el mismo formato sin importar el origen para conservar paridad funcional entre ambientes.

El ranking final pondera dividendos, valuación, crecimiento y cobertura geográfica para sostener la consistencia del score compuesto.

Cada oportunidad obtiene un **score normalizado en escala 0-100** que promedia aportes de payout, racha de dividendos, CAGR, recompras, RSI y MACD. Esta normalización permite comparar emisores de distintas fuentes con un criterio homogéneo. Los resultados que queden por debajo del umbral configurado se descartan automáticamente para reducir ruido.

Los controles disponibles en la UI permiten ajustar esos filtros sin modificar código, y la interfaz incluye un glosario interactivo [¿Qué significa cada métrica?](#qué-significa-cada-métrica) con ejemplos numéricos para alinear la interpretación de payout, EPS, CAGR, buybacks y score entre la documentación y la aplicación:

- Multiselect de sectores para recortar el universo devuelto por la búsqueda.
- Checkbox **Incluir indicadores técnicos** para agregar RSI y medias móviles al resultado.
- Inputs dedicados a crecimiento mínimo de EPS y porcentaje mínimo de recompras (`buybacks`).
- Sliders y number inputs para capitalización, payout, P/E, crecimiento de ingresos, racha/CAGR de dividendos e inclusión de Latinoamérica.
- Slider de score mínimo para ajustar `score_compuesto` sin salir de la UI.
- Selector **Perfil recomendado** para aplicar presets preconfigurados según el tipo de oportunidad que se quiera priorizar:
  - **Dividendos defensivos**: favorece emisores consolidados con payout moderado, más de 10 años de dividendos, crecimiento estable y foco en sectores defensivos (``Consumer Defensive`` y ``Utilities``).
  - **Crecimiento balanceado**: combina expansión de ingresos/EPS de dos dígitos con payout controlado y sesgo hacia ``Technology`` y ``Healthcare`` para captar historias de crecimiento rentable.
  - **Recompras agresivas**: apunta a compañías con recompras netas relevantes, valuaciones razonables e inclusión de indicadores técnicos para reforzar el timing, con foco en ``Financial Services``, ``Technology`` e ``Industrials``.

El umbral mínimo de score y el recorte del **top N** de oportunidades son parametrizables mediante las variables `MIN_SCORE_THRESHOLD` (valor por defecto: `80`) y `MAX_RESULTS` (valor por defecto: `20`). La interfaz utiliza ese valor centralizado como punto de partida en el selector "Máximo de resultados" para reflejar cualquier override definido en la configuración. Puedes redefinirlos desde `.env`, `secrets.toml` o `config.json` para adaptar la severidad del filtro o ampliar/restringir el listado mostrado en la UI. La cabecera del listado muestra notas contextuales cuando se aplican estos recortes y sigue diferenciando la procedencia de los datos con un caption que alterna entre `yahoo` y `stub`, manteniendo la trazabilidad de la fuente durante los failovers.

Los ejemplos documentados (tabla, presets y telemetría) reflejan la release vigente, donde la UI muestra `score_compuesto` en la grilla principal y conserva el caption `yahoo`/`stub` para todas las variantes de origen.


#### ¿Qué significa cada métrica?

- **Payout:** porcentaje de las ganancias que se reparte como dividendo. Ejemplo: con un payout del 60 %, una empresa distribuye US$0,60 por cada dólar de utilidad.
- **EPS (Earnings Per Share):** ganancias por acción. Si una firma genera US$5 millones y tiene 1 millón de acciones, su EPS es US$5.
- **Crecimiento de ingresos:** variación interanual de ventas. Un aumento de US$100 a US$112 implica un crecimiento del 12 %.
- **Racha de dividendos:** cantidad de años consecutivos pagando dividendos. Una racha de 7 significa pagos sin interrupciones durante siete ejercicios.
- **CAGR de dividendos:** crecimiento anual compuesto del dividendo. Pasar de US$1 a US$1,50 en cinco años implica un CAGR cercano al 8 %.
- **Buybacks:** recompras netas que reducen el flotante. Un buyback del 2 % indica que la empresa retiró 2 de cada 100 acciones en circulación.
- **Score compuesto:** puntaje de 0 a 100 que combina valuación, crecimiento, dividendos y técnicos; por ejemplo, un score de 85 señala atributos superiores al umbral típico de 80.


### Notas del listado y severidades

Las notas del listado utilizan una clasificación estandarizada para transmitir la severidad del mensaje. Cada nivel comparte prefijos visibles en el texto bruto (útiles en pruebas o fixtures) y un icono renderizado al pasar por `shared.ui.notes.format_note`:

| Severidad | Prefijos esperados | Icono renderizado | Uso típico |
| --- | --- | --- | --- |
| `warning` | `⚠️` | `:warning:` | Avisar que los datos provienen de un stub, que el universo está recortado o que hubo fallbacks. |
| `info` | `ℹ️` | `:information_source:` | Recordatorios operativos o mensajes neutrales relacionados con disponibilidad de datos. |
| `success` | `✅` | `:white_check_mark:` | Confirmar procesos completados o resultados satisfactorios. |
| `error` | `❌` | `:x:` | Indicar fallas irrecuperables que el usuario debe revisar. |

Siempre que sea posible prefija el contenido con el emoji correspondiente para que el helper lo clasifique correctamente. El siguiente ejemplo mínimo muestra cómo centralizar el formato en la UI:

```python
from shared.ui.notes import format_note

format_note("⚠️ Solo se encontraron 3 tickers con datos recientes.")
# ":warning: **Solo se encontraron 3 tickers con datos recientes.**"
```

## Integración con Yahoo Finance

La aplicación consulta [Yahoo Finance](https://finance.yahoo.com/) mediante la librería `yfinance` para enriquecer la vista de portafolio con series históricas, indicadores técnicos y métricas fundamentales/ESG. La barra lateral de healthcheck refleja si la última descarga provino de Yahoo o si fue necesario recurrir a un respaldo local, facilitando la observabilidad de esta dependencia externa.

### Smoke-test nocturno y guardas de frecuencia

El workflow [`CI`](.github/workflows/ci.yml) ejecuta un smoke-test live contra Yahoo Finance todas las noches a las **02:30 UTC** a través del job `live-yahoo-smoke`. El disparador manual (`workflow_dispatch`) permanece disponible con el input `run-live-yahoo` para validar la dependencia bajo demanda sin esperar a la corrida programada, y se complementa con el toggle `skip-live-yahoo` para omitir la corrida si necesitas preservar cuota en una ejecución manual.

Para evitar saturar el rate limit de Yahoo se exponen variables de repositorio que controlan la frecuencia:

- `LIVE_YAHOO_SMOKE_SCHEDULE_MODE` (default: `nightly`) acepta `manual` para deshabilitar por completo los disparos programados, `weekdays` para limitarse a lunes-viernes y `custom` para utilizar una lista explícita de días.
- `LIVE_YAHOO_SMOKE_ALLOWED_DAYS` define la lista de días permitidos (por ejemplo `mon,thu`) cuando el modo es `custom`. Los valores usan abreviaturas en inglés y se evalúan en UTC.
- `LIVE_YAHOO_SMOKE_FORCE_SKIP` (default: `false`) pausa cualquier ejecución del job —manual o programada— hasta que lo vuelvas a colocar en `false`. Útil cuando se detecta throttling y conviene guardar cuota sin tocar la configuración de horarios.

Si sólo necesitas suspender una corrida puntual, lanza el workflow manualmente con `run-live-yahoo=true` para habilitar el resto de jobs y marca `skip-live-yahoo=true`. El job quedará documentado como omitido en el historial sin consumir requests adicionales.

Cada ejecución deja trazabilidad en los logs del job y, en modo programado, documenta si se omitió el smoke-test debido a la guarda de frecuencia o a un skip explícito. Ante un fallo:

1. Revisa el paso **Run live Yahoo Finance smoke-test** para capturar el traceback y confirmar si se trata de un error transitorio (timeouts, throttling) o funcional.
2. Si el fallo proviene de rate limiting, habilita temporalmente `LIVE_YAHOO_SMOKE_FORCE_SKIP=true` o relanza manualmente el workflow con `skip-live-yahoo=true` hasta que la cuota se recupere.
3. Para validar la corrección, vuelve a lanzar el job desde la UI de Actions o ejecuta `pytest -m live_yahoo` localmente con `RUN_LIVE_YF=1`.

### Indicadores técnicos y fallback local

La función `fetch_with_indicators` descarga OHLCV y calcula indicadores (SMA, EMA, MACD, RSI, Bollinger, ATR, Estocástico e Ichimoku). Los resultados se almacenan en caché durante el intervalo definido por `CACHE_TTL_YF_INDICATORS` (predeterminado: 900 segundos) para evitar llamadas redundantes. Cuando `yfinance` produce un `HTTPError` o `Timeout`, la aplicación recurre automáticamente a `infrastructure/cache/ta_fallback.csv` como stub hasta que el servicio se restablezca.

### Métricas fundamentales y ranking del portafolio

`get_fundamental_data` obtiene valuaciones básicas (PE, P/B, márgenes, ROE, deuda, dividend yield depurado) y respeta el TTL de `CACHE_TTL_YF_FUNDAMENTALS` (6 horas por defecto). Para el ranking consolidado se utiliza `portfolio_fundamentals`, que agrega métricas y puntajes ESG por símbolo y persiste los resultados según `CACHE_TTL_YF_PORTFOLIO_FUNDAMENTALS` (4 horas por defecto). Ambos bloques se muestran en la pestaña principal del portafolio, con mensajes claros cuando los datos no están disponibles.

### Históricos y monitoreo

`get_portfolio_history` construye series ajustadas para todos los símbolos y las conserva durante `CACHE_TTL_YF_HISTORY` (valor inicial: 3600 segundos). El healthcheck `📈 Yahoo Finance` indica si la última consulta provino de la API, de la caché o del stub, junto con detalles del símbolo involucrado.

## Seguridad de credenciales

### 🔒 Seguridad de tus credenciales

- Cifrado de tokens con [Fernet](https://cryptography.io/en/latest/fernet/)
- Almacenamiento de secretos con [Streamlit Secrets](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/secrets-management)
- Tokens guardados en archivos cifrados locales (no en la nube)
- Limpieza inmediata de contraseñas en `session_state`

Tus credenciales nunca se almacenan en servidores externos. El acceso a IOL se realiza de forma segura mediante tokens cifrados, protegidos con clave Fernet y gestionados localmente por la aplicación.

El bloque de login muestra la versión actual de la aplicación con un mensaje como "Estas medidas de seguridad aplican a la versión 0.3.27.1".

El menú **⚙️ Acciones** refuerza la seguridad operativa al anunciar con toasts cada vez que se refrescan los datos o se completa el cierre de sesión, dejando constancia en la propia UI sin depender de logs externos.

El sidebar finaliza con un bloque de **Healthcheck (versión 0.3.27.1)** que lista el estado de los servicios monitoreados, resalta si la respuesta proviene de la caché o de un fallback y ahora agrega estadísticas agregadas de latencia, resiliencia y reutilización, incluyendo el resumen macro con World Bank.

### Interpretación del health sidebar (KPIs agregados)

- **Conexión IOL (`🔐`)**: informa el último refresco exitoso o fallido con timestamp y detalle para incidentes de autenticación.
- **Yahoo Finance (`📈`)**: muestra si las cotizaciones provienen de Yahoo, del fallback local o si hubo errores; cada entrada incluye el timestamp y un detalle del símbolo involucrado.
- **FX (`💱`)**: divide en dos líneas el estado de la API y de la caché, exponiendo latencia en milisegundos, edad del dato y mensajes de error en caso de fallar.
- **Screening de oportunidades (`🔎`)**: indica si el último barrido reutilizó la caché o corrió completo, con tiempos actuales, baseline cacheado, universo inicial/final, ratio de descartes y sectores destacados. Cuando hay historial suficiente, la nueva línea de "tendencia" agrega promedios, desvíos, ratio de *hits* (incluidos los totales) y métricas de mejora frente a la caché.
- **Historial de screenings (`🗂️`)**: renderiza una tabla con los barridos recientes, marcando cada modo (`hit`/`miss`), el delta frente al promedio y el tiempo cacheado de referencia.
- **Latencias (`⏱️`)**: resume en líneas separadas la latencia de la carga del portafolio y de las cotizaciones, incluyendo fuente, cantidad de ítems y timestamp para correlacionar con incidentes puntuales.

## Requisitos de sistema

- Python 3.10 o superior
- `pip` y recomendablemente `venv` o `virtualenv`

## Instalación

1. Clonar el repositorio y crear un entorno virtual (opcional).
2. Instalar dependencias de producción:

```bash
pip install -r requirements.txt
```

Para un entorno de desarrollo con herramientas de linting y pruebas:

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Configuración del entorno

Crear un archivo `.env` en la raíz del proyecto con los ajustes necesarios (las credenciales se ingresan desde la interfaz de la aplicación):

```env
USER_AGENT="Portafolio-IOL/1.0"
# Ruta opcional del archivo de tokens
IOL_TOKENS_FILE="tokens_iol.json"
# Clave para cifrar el archivo de tokens (Fernet). Debe definirse en producción
IOL_TOKENS_KEY="..."
# Permite guardar tokens sin cifrar (NO recomendado)
IOL_ALLOW_PLAIN_TOKENS=0
# Otros ajustes opcionales
CACHE_TTL_PORTFOLIO=20
CACHE_TTL_LAST_PRICE=10
CACHE_TTL_QUOTES=8
CACHE_TTL_FX=60
CACHE_TTL_YF_INDICATORS=900
CACHE_TTL_YF_HISTORY=3600
CACHE_TTL_YF_FUNDAMENTALS=21600
CACHE_TTL_YF_PORTFOLIO_FUNDAMENTALS=14400
YAHOO_FUNDAMENTALS_TTL=3600
YAHOO_QUOTES_TTL=300
MIN_SCORE_THRESHOLD=80
MAX_RESULTS=20
ASSET_CATALOG_PATH="/ruta/a/assets_catalog.json"
# Nivel de los logs ("DEBUG", "INFO", etc.; predeterminado: INFO)
LOG_LEVEL="INFO"
# Formato de los logs: "plain" o "json" (predeterminado: plain)
LOG_FORMAT="plain"
# Usuario opcional incluido en los logs
LOG_USER="usuario"
```
Los parámetros `CACHE_TTL_YF_*` ajustan cuánto tiempo se reutiliza cada respuesta de Yahoo Finance antes de volver a consultar la API (indicadores técnicos, históricos, fundamentales individuales y ranking del portafolio, respectivamente). Las variables `YAHOO_FUNDAMENTALS_TTL` (3600 segundos por defecto) y `YAHOO_QUOTES_TTL` (300 segundos por defecto) controlan el TTL de la caché específica para fundamentales y cotizaciones de Yahoo; puedes redefinir estos valores en el `.env` o en `secrets.toml` según tus necesidades. Ambos parámetros también se exponen con alias en minúsculas (`yahoo_fundamentals_ttl` y `yahoo_quotes_ttl`) para facilitar su lectura desde `st.secrets`, y cualquier alias o nombre en mayúsculas puede sobrescribirse indistintamente mediante variables de entorno, archivos `.env` o secretos.

`MIN_SCORE_THRESHOLD` (80 por defecto) define el puntaje mínimo aceptado para que una empresa aparezca en el listado estable de oportunidades, mientras que `MAX_RESULTS` (20 por defecto) determina cuántas filas finales mostrará la UI tras aplicar filtros y ordenar el score normalizado. Ambos valores pueden sobreescribirse desde el mismo `.env`, `secrets.toml` o `config.json` si necesitás afinar la agresividad del recorte.
También puedes definir estos valores sensibles en `secrets.toml`,
el cual `streamlit` expone a través de `st.secrets`. Los valores en
`secrets.toml` tienen prioridad sobre las variables de entorno.

Ejemplo de `.streamlit/secrets.toml`:

```toml
IOL_USERNAME = "tu_usuario"
IOL_PASSWORD = "tu_contraseña"
IOL_TOKENS_KEY = "clave"
IOL_TOKENS_FILE = "tokens_iol.json"
```

`LOG_LEVEL` controla la verbosidad de los mensajes (`DEBUG`, `INFO`, etc.). Evita usar `DEBUG` u otros niveles muy verbosos en producción, ya que pueden revelar información sensible y generar un volumen excesivo de datos. `LOG_FORMAT` puede ser `plain` para un formato legible o `json` para registros estructurados, útil cuando se integran sistemas de logging centralizado o se requiere auditoría. Si `LOG_LEVEL` o `LOG_FORMAT` no están definidos, la aplicación utiliza `INFO` y `plain` como valores por defecto. El valor de `LOG_USER` se incluye en los registros si está definido.

Las credenciales de IOL se utilizan para generar un token de acceso que se guarda en `tokens_iol.json` (o en la ruta indicada por `IOL_TOKENS_FILE`). Si `IOL_TOKENS_KEY` no está configurada y `IOL_ALLOW_PLAIN_TOKENS` no está habilitado, la aplicación registrará un error y se cerrará con código 1 para evitar guardar el archivo sin cifrar. Se puede forzar este comportamiento (solo para entornos de prueba) estableciendo `IOL_ALLOW_PLAIN_TOKENS=1`. Puedes generar una clave con:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Este archivo es sensible: **manténlo fuera del control de versiones** (ya está incluido en `.gitignore`) y con permisos restringidos, por ejemplo `chmod 600`. Si el token expira o se desea forzar una nueva autenticación, borra dicho archivo.

## Ejecución local

```bash
streamlit run app.py
```

## Despliegue

En entornos de producción es obligatorio definir la variable `IOL_TOKENS_KEY` para que el archivo de tokens se almacene cifrado. Si falta y `IOL_ALLOW_PLAIN_TOKENS` no está habilitado, la aplicación registrará el problema y se cerrará.

### Docker

1. Construir la imagen:

```bash
docker build -t portafolio-iol .
```

2. Ejecutar el contenedor (requiere un archivo `.env` con las variables descritas en la sección anterior):

```bash
docker run --env-file .env -p 8501:8501 portafolio-iol
```

La imagen define un `HEALTHCHECK` que consulta `http://localhost:8501/_stcore/health` para comprobar la disponibilidad del servicio durante el despliegue.

Para conservar los tokens generados por la aplicación, se puede montar un volumen:

```bash
mkdir -p tokens
docker run --env-file .env -p 8501:8501 -v $(pwd)/tokens:/app/tokens portafolio-iol
```

Al usar un volumen, define en `.env` la ruta del archivo:

```env
IOL_TOKENS_FILE=/app/tokens/tokens_iol.json
```

### Streamlit Cloud

1. Subir el repositorio a GitHub.
2. En [Streamlit Cloud](https://streamlit.io/cloud), crear una nueva aplicación apuntando a `app.py`.
   La carpeta `.streamlit` ya incluye un `config.toml` con `fileWatcherType = "poll"` para evitar el límite de inotify en la plataforma.
3. En el panel de la aplicación, abre el menú **⋮** y selecciona **Edit secrets** para mostrar la pestaña **Secrets**.
4. Completa el editor con un `secrets.toml` mínimo:
   ```toml
   USER_AGENT = "Portafolio-IOL/1.0"
   IOL_TOKENS_FILE = "tokens_iol.json"
   IOL_TOKENS_KEY = "..."
   IOL_ALLOW_PLAIN_TOKENS = 0
   CACHE_TTL_PORTFOLIO = 20
   CACHE_TTL_LAST_PRICE = 10
   CACHE_TTL_QUOTES = 8
   CACHE_TTL_FX = 60
   CACHE_TTL_YF_INDICATORS = 900
   CACHE_TTL_YF_HISTORY = 3600
   CACHE_TTL_YF_FUNDAMENTALS = 21600
   CACHE_TTL_YF_PORTFOLIO_FUNDAMENTALS = 14400
   ASSET_CATALOG_PATH = "/ruta/a/assets_catalog.json"
   LOG_LEVEL = "INFO"
   LOG_FORMAT = "plain"
   LOG_USER = "usuario"
   ```
   Estos valores coinciden con las variables del ejemplo `.env`.
5. Guarda los cambios y despliega la aplicación.

Para más detalles, consulta la [documentación oficial de Streamlit sobre gestión de secrets](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/secrets-management).

## Post-deploy notes

- **Revisar métricas de performance**:
  - En despliegues locales o Docker, usa `docker stats <container>` para monitorear CPU/memoria y `docker logs <container>` para tiempos de respuesta.
  - En Streamlit Cloud, abre el menú **⋮** y selecciona **View app logs** para ver métricas de la instancia.
- **Prueba de login multiusuario**:
  1. Abre dos navegadores distintos o ventanas en modo incógnito.
  2. Inicia sesión en cada uno con credenciales válidas.
  3. Verifica que cada sesión opere de forma independiente; cerrar sesión en una no debe afectar a la otra.

## Pruebas

Consulta la guía extendida en [docs/testing.md](docs/testing.md) para instrucciones detalladas,
marcadores y flags recomendados.

Con las dependencias de desarrollo instaladas, ejecutar la suite completa de pruebas:

```bash
pytest
```

Para ejecutar solo un subconjunto por carpeta, indica la ruta deseada:

```bash
pytest application/test
```

Para habilitar las pruebas que consultan Yahoo Finance en vivo, exporta la
variable `RUN_LIVE_YF=1` y ejecuta la etiqueta dedicada. Estas verificaciones
descargan datos reales y, por tratarse de información del mercado en tiempo
real, pueden arrojar resultados no deterministas entre corridas.

```bash
RUN_LIVE_YF=1 pytest -m live_yahoo
```

Para verificar el estilo del código:

```bash
flake8
```

### Job opcional de Yahoo Finance en CI

El workflow `CI` incluye un job opcional que ejecuta `pytest -m live_yahoo`
con `RUN_LIVE_YF=1` para validar la integración real con Yahoo Finance.
Este smoke-test **no se ejecuta automáticamente** porque consume datos en
tiempo real y los resultados pueden variar entre corridas. Para activarlo:

1. Ingresa a **Actions → CI → Run workflow**.
2. Habilita el toggle **Run live Yahoo Finance smoke-test**.
3. Ejecuta el workflow manualmente.

Al hacerlo, GitHub Actions exportará `RUN_LIVE_YF=1` antes de invocar el
marcador `live_yahoo`. Usa este job sólo cuando necesites verificar la
integración en vivo o validar incidentes relacionados con Yahoo Finance.

### Barrido prolongado del stub y presets en CI

Adicionalmente, el workflow programa un job nocturno `stub-fallback-sweep`
que se ejecuta todos los días a las **03:00 UTC** para ejercitar el stub y
los presets recomendados. Este barrido ejecuta la batería prolongada de
pruebas de fallback sobre el stub (incluye validaciones de notas, presets y
consistencia entre corridas) y registra métricas de duración junto con los
totales de `passed/failed/errors/skipped`.

Para detonarlo manualmente:

1. Ingresa a **Actions → CI → Run workflow**.
2. Habilita el toggle **Run prolonged stub fallback & preset sweep**.
3. Ejecuta el workflow.

Al finalizar, revisa el resumen del job en GitHub Actions o descarga el
artefacto `stub-sweep-logs`, que incluye `stub_sweep.log` y
`stub_sweep_metrics.json` con las métricas necesarias para seguimiento de QA. Allí se registran el `elapsed_time`, el `universe_initial`, el universo final y los porcentajes de descartes de fundamentals/técnicos que muestra la nota de telemetría (ver [guía de interpretación](#telemetría-del-barrido) para detalles). En los monitoreos nocturnos consideramos saludable que el stub termine en menos de 3 segundos, que el universo se mantenga estable (37 símbolos) y que las tasas de descarte se mantengan por debajo del 25 %; desvíos persistentes disparan revisiones manuales o ajustes en los presets.

## Tiempos de referencia

Los siguientes tiempos se observan en condiciones normales (aprox. 20 posiciones):

| Paso                | Tiempo objetivo | Detalles |
|---------------------|-----------------|----------|
| `login`             | < 1 s           | `auth_service.login` |
| `fetch_portfolio`   | < 600 ms        | ~20 posiciones |
| `fetch_quotes_bulk` | < 1 s           | 20 símbolos |

Si algún paso supera estos valores, considera reducir llamadas redundantes, ajustar los TTL de cache en `shared/settings.py` o incrementar `MAX_QUOTE_WORKERS` cuando existan muchas posiciones.

## Fallback de análisis técnico

Si ocurre un `HTTPError` o un `Timeout` al descargar datos con `yfinance`,
la función `fetch_with_indicators` recurre al archivo local
`infrastructure/cache/ta_fallback.csv`. Este archivo contiene datos
de respaldo con formato OHLCV utilizados para generar los indicadores.

Para actualizarlo con información reciente, ejecuta el servicio cuando
tengas conexión y guarda el resultado en la misma ruta:

```bash
python - <<'PY'
from application.ta_service import fetch_with_indicators
import pandas as pd
df = fetch_with_indicators('AAPL')  # o el símbolo deseado
df.to_csv('infrastructure/cache/ta_fallback.csv')
PY
```


## Actualización de dependencias

Las versiones de las dependencias están fijadas en `requirements.txt`. Para actualizarlas de forma segura:

```bash
bash scripts/update_dependencies.sh
```

El script actualiza los paquetes a sus últimas versiones, ejecuta las pruebas y, si todo pasa, escribe las nuevas versiones en `requirements.txt`. Este proceso también se ejecuta mensualmente mediante [GitHub Actions](.github/workflows/dependency-update.yml).

La guía interna que detalla cómo recrear los assets del dashboard se apoya en el script generador correspondiente; a partir de ahora `matplotlib` queda instalada automáticamente al ejecutar `pip install -r requirements.txt`, por lo que no hace falta agregarla manualmente antes de correr ese flujo.

## Políticas de sesión y manejo de tokens

Cada sesión de usuario genera un identificador aleatorio almacenado en `st.session_state["session_id"]`, que debe mantenerse constante para aislar los recursos cacheados.

Los tokens de autenticación se guardan en `tokens_iol.json` (o en la ruta indicada por `IOL_TOKENS_FILE`) y deben almacenarse cifrados mediante `IOL_TOKENS_KEY`. Este archivo no debe versionarse y conviene mantenerlo con permisos restringidos (por ejemplo `chmod 600`). Para renovar los tokens:

1. Eliminar el archivo de tokens.
2. Volver a ejecutar la aplicación para que se generen nuevamente.

## Licencia

Este proyecto se distribuye bajo la licencia MIT.
