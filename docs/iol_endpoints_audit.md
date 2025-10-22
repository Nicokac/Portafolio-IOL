# Auditoría de endpoints IOL

Esta nota resume todas las llamadas HTTP efectivas que realiza Portafolio-IOL contra la API de InvertirOnline (IOL), las capas que las consumen y los campos relevantes que se utilizan en la aplicación. También detalla el flujo de clasificación de activos y las oportunidades de depuración detectadas.

## Endpoints en uso

| Método | Endpoint | Ubicación (archivo → función) | Propósito | Campos utilizados | Estado |
| --- | --- | --- | --- | --- | --- |
| POST | `/token` | `infrastructure/iol/auth.py` → `IOLAuth.login` / `IOLAuth.refresh` | Obtener `access_token` y `refresh_token` mediante los grants `password` y `refresh_token`. | `access_token`, `refresh_token`, `expires_in`, `user_id` (derivado para auditoría). | Activo (flujo de login y refresh). |
| GET | `/api/v2/portafolio/{pais}` | `infrastructure/iol/client.py` → `IOLClient._fetch_portfolio_live` / `get_portfolio` | Recuperar posiciones y metadatos de cada activo del portafolio. | `activos[].titulo.{tipo, descripcion, simbolo, mercado}`, `cantidad`, `cantidadDisponible`, `cantidadNominal`, `costoUnitario`, `costoTotal`, `inversion`. | Activo (carga inicial del portafolio). |
| GET | `/api/v2/estadocuenta` | `infrastructure/iol/account_client.py` → `IOLAccountClient.fetch_balances` (invocado desde `IOLClient.get_portfolio`) | Complementar el portafolio con saldos de efectivo. | `disponibleEnPesos`, `disponibleEnDolares`, `cotizacionDolar`, `cuentas[].{moneda, descripcion, cotizacion, disponibleParaOperar…}` | Activo (se ejecuta después de cada `/portafolio`). |
| GET | `/api/v2/{mercado}/Titulos/{simbolo}/Cotizacion` (`panel` opcional) | `infrastructure/iol/client.py` → `IOLClient.get_quote` | Obtener la cotización en vivo y variación porcentual. | `ultimoPrecio` (y variantes), `variacion`, `fecha`, `moneda`, `cierreAnterior`, `provider`. | Activo (cotizaciones en tiempo real vía caché). |
| GET | `/api/v2/portafolio/{pais}` | `infrastructure/iol/legacy/iol_client.py` → `LegacyIOLClient.get_portfolio` | Versión heredada del fetch de portafolio. | Igual que la versión actual (`activos[].*`). | Definido pero no referenciado en los flujos vigentes (solo queda para compatibilidad/legacy). |

### Cómo se consumen en la aplicación

- `services/cache/portfolio_cache.fetch_portfolio()` invoca a `IOLClient.get_portfolio()`, por lo que dispara primero `/api/v2/portafolio/{pais}` y luego `/api/v2/estadocuenta` para adjuntar `_cash_balances` al payload cacheado.【F:services/cache/portfolio_cache.py†L24-L69】【F:infrastructure/iol/client.py†L229-L284】【F:infrastructure/iol/account_client.py†L73-L163】
- `services/cache/quotes._get_quote_cached()` usa `IOLClient.get_quote()` para cada símbolo, aplicando reintentos ante HTTP 429 y propagando la información normalizada hacia la UI y los cálculos del portafolio.【F:services/cache/quotes.py†L680-L739】【F:infrastructure/iol/client.py†L607-L758】
- `application/auth_service.IOLAuthenticationProvider.login()` crea una instancia de `IOLAuth` y ejecuta `login()`, que es quien emite el `POST /token`; la renovación se realiza automáticamente a través de `ensure_token()` y `refresh()` cuando expira el `access_token`.【F:application/auth_service.py†L25-L88】【F:infrastructure/iol/auth.py†L268-L417】

### Wrappers y capas auxiliares

- `IOLClient.get_last_price()` y los fallbacks de `IOLClient.get_quote()` utilizan `iolConn` (SDK propietario de IOL) y un adaptador OHLC como respaldo; estas rutas no hacen nuevas llamadas REST distintas a las listadas arriba, sino que reutilizan el resultado de `/Titulos/Cotizacion` o fuentes alternativas locales.【F:infrastructure/iol/client.py†L296-L758】
- El cliente legado (`infrastructure/iol/legacy/iol_client.py`) persiste para compatibilidad y como fallback de cotizaciones mediante la sesión compartida `LegacySession`, pero sus métodos HTTP (`get_portfolio`) ya no están conectados al flujo principal.【F:infrastructure/iol/legacy/iol_client.py†L41-L191】

## Clasificación de activos

1. **Información declarada por IOL.** `classify_asset()` prioriza los campos `titulo.tipo` y `titulo.descripcion` entregados por `/api/v2/portafolio`, normalizándolos con `normalize_asset_type()` y reglas específicas (`_match_declared_type`).【F:application/portfolio_service.py†L224-L279】【F:shared/asset_type_aliases.py†L1-L36】
2. **Catálogo local.** Si IOL no provee un tipo utilizable, `classify_symbol()` consulta primero el catálogo consolidado (`infrastructure/asset_catalog.get_asset_catalog()`), que ya viene con `tipo_estandar` cuando existe.【F:application/portfolio_service.py†L148-L198】【F:infrastructure/asset_catalog.py†L1-L81】
3. **Alias y heurísticas de configuración.** Persisten reglas basadas en listas de configuración (`cedear_to_us`, `etfs`, `fci_symbols`, `acciones_ar`) y patrones regex (`classification_patterns`). Si todo falla, aplica heurísticas simples (prefijos de bonos/letras o longitud del ticker) y devuelve `"Otro"`.【F:application/portfolio_service.py†L148-L198】
4. **Escalas y totales.** La información de `/estadocuenta` se inyecta como `_cash_balances` para que los totales combinen efectivo en ARS y USD, mientras que `scale_for()` ajusta bonos/letras (VN 100 → factor 0.01).【F:infrastructure/iol/client.py†L273-L283】【F:application/portfolio_service.py†L200-L221】【F:application/portfolio_service.py†L285-L381】

En resumen, la jerarquía es: **etiquetas de IOL → alias normalizados → catálogo local → listas de configuración/patrones → heurísticas finales**.

## Observaciones y oportunidades de mejora

- **Depurar código legado:** `LegacyIOLClient.get_portfolio()` no se utiliza desde la migración al cliente nuevo; puede eliminarse o aislarse detrás de pruebas para reducir mantenimiento del endpoint duplicado.【F:infrastructure/iol/legacy/iol_client.py†L41-L191】
- **Centralizar telemetría HTTP:** hoy la instrumentación de latencia (telemetría/health) vive en `IOLClient` y `services/cache/quotes`. Consolidar el registro en un middleware común simplificaría la observabilidad de `/portafolio`, `/estadocuenta` y `/Cotizacion`.
- **Cliente único para tokens:** `IOLAuth` instancia su propio `requests.Session`, mientras que `IOLClient` y `IOLAccountClient` reutilizan otra sesión. Extraer un “IOLHttpClient” compartido permitiría configurar retries, headers y logging en un solo lugar.

> _Última actualización: automatización Codex Cloud, commit inicial de auditoría._
