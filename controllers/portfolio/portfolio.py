import logging
import re
import time
import unicodedata
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence, cast

import streamlit as st

from domain.models import Controls
from ui.sidebar_controls import render_sidebar
from ui.fundamentals import render_fundamental_data
from ui.export import PLOTLY_CONFIG
from ui.charts import plot_technical_analysis_chart
from ui.favorites import render_favorite_badges, render_favorite_toggle
from application.portfolio_service import PortfolioService, map_to_us_ticker
from application.ta_service import TAService
from application.portfolio_viewmodel import build_portfolio_viewmodel
from shared.errors import AppError
from shared.favorite_symbols import FavoriteSymbols, get_persistent_favorites
from services.notifications import NotificationFlags, NotificationsService
from services.health import record_tab_latency
from services.portfolio_view import PortfolioViewModelService
from services import snapshots as snapshot_service
from ui.notifications import render_technical_badge, tab_badge_label, tab_badge_suffix
from shared.utils import _as_float_or_none, format_money
from services.performance_metrics import measure_execution

from .load_data import load_portfolio_data
from .charts import render_basic_section, render_advanced_analysis
from .risk import render_risk_analysis
from .fundamentals import render_fundamental_analysis
logger = logging.getLogger(__name__)

_SERVICE_REGISTRY_KEY = "__portfolio_services__"
_VIEW_MODEL_SERVICE_KEY = "view_model_service"
_VIEW_MODEL_FACTORY_KEY = "view_model_service_factory"
_NOTIFICATIONS_SERVICE_KEY = "notifications_service"
_NOTIFICATIONS_FACTORY_KEY = "notifications_service_factory"
_SNAPSHOT_BACKEND_KEY = "snapshot_backend_override"
_PORTFOLIO_SERVICE_KEY = "portfolio_service"
_TA_SERVICE_KEY = "ta_service"

def _get_service_registry() -> dict[str, Any]:
    """Return the per-session registry that stores portfolio services."""

    state = getattr(st, "session_state", None)
    if state is not None:
        try:
            registry = state.get(_SERVICE_REGISTRY_KEY)  # type: ignore[attr-defined]
        except AttributeError:
            try:
                registry = state[_SERVICE_REGISTRY_KEY]  # type: ignore[index]
            except Exception:  # pragma: no cover - defensive branch
                registry = None
        if not isinstance(registry, dict):
            registry = {}
            try:
                state[_SERVICE_REGISTRY_KEY] = registry  # type: ignore[index]
            except Exception:  # pragma: no cover - defensive branch
                pass
        if isinstance(registry, dict):
            return registry

    return {}


def default_view_model_service_factory() -> PortfolioViewModelService:
    """Create a portfolio view model service bound to the configured backend."""

    registry = _get_service_registry()
    snapshot_backend = registry.get(_SNAPSHOT_BACKEND_KEY)
    backend = snapshot_backend if snapshot_backend is not None else snapshot_service
    return PortfolioViewModelService(snapshot_backend=backend)


def default_notifications_service_factory() -> NotificationsService:
    """Return a fresh notifications service instance."""

    return NotificationsService()


def default_portfolio_service_factory() -> PortfolioService:
    """Return a fresh portfolio service instance."""

    return PortfolioService()


def default_ta_service_factory() -> TAService:
    """Return a fresh technical-analysis service instance."""

    return TAService()


def _get_or_create_service(
    key: str,
    *,
    default_factory: Callable[[], Any],
    override_factory: Callable[[], Any] | None = None,
) -> Any:
    """Return a cached service for the current session, creating it if needed."""

    registry = _get_service_registry()
    factory_key = f"{key}_factory"
    if override_factory is not None:
        registry[factory_key] = override_factory
        registry.pop(key, None)

    factory = registry.get(factory_key)
    if not callable(factory):
        factory = default_factory
        registry[factory_key] = factory

    service = registry.get(key)
    if service is None:
        service = factory()
        registry[key] = service
    return service


def get_portfolio_view_service(
    factory: Callable[[], PortfolioViewModelService] | None = None,
) -> PortfolioViewModelService:
    """Return the cached portfolio view service, creating it if necessary."""

    service = _get_or_create_service(
        _VIEW_MODEL_SERVICE_KEY,
        default_factory=default_view_model_service_factory,
        override_factory=factory,
    )
    return cast(PortfolioViewModelService, service)


def get_notifications_service(
    factory: Callable[[], NotificationsService] | None = None,
) -> NotificationsService:
    """Return the cached notifications service, creating it if necessary."""

    service = _get_or_create_service(
        _NOTIFICATIONS_SERVICE_KEY,
        default_factory=default_notifications_service_factory,
        override_factory=factory,
    )
    return cast(NotificationsService, service)


def get_portfolio_service(
    factory: Callable[[], PortfolioService] | None = None,
) -> PortfolioService:
    """Return the cached portfolio service instance."""

    service = _get_or_create_service(
        _PORTFOLIO_SERVICE_KEY,
        default_factory=default_portfolio_service_factory,
        override_factory=factory,
    )
    return cast(PortfolioService, service)


def get_ta_service(factory: Callable[[], TAService] | None = None) -> TAService:
    """Return the cached technical-analysis service instance."""

    service = _get_or_create_service(
        _TA_SERVICE_KEY,
        default_factory=default_ta_service_factory,
        override_factory=factory,
    )
    return cast(TAService, service)


def reset_portfolio_services() -> None:
    """Clear cached portfolio services for the current session."""

    if getattr(st, "session_state", None) is None:
        return

    registry = _get_service_registry()
    for key in (
        _VIEW_MODEL_SERVICE_KEY,
        _VIEW_MODEL_FACTORY_KEY,
        _NOTIFICATIONS_SERVICE_KEY,
        _NOTIFICATIONS_FACTORY_KEY,
        _PORTFOLIO_SERVICE_KEY,
        _TA_SERVICE_KEY,
    ):
        registry.pop(key, None)


def configure_snapshot_backend(snapshot_backend: Any | None) -> None:
    """Override the snapshot backend used by the cached portfolio service."""

    if getattr(st, "session_state", None) is None:
        return

    registry = _get_service_registry()
    registry[_SNAPSHOT_BACKEND_KEY] = snapshot_backend

    service = registry.get(_VIEW_MODEL_SERVICE_KEY)
    configure = getattr(service, "configure_snapshot_backend", None)
    if callable(configure):
        configure(snapshot_backend)


def _apply_tab_badges(tab_labels: list[str], flags: NotificationFlags) -> list[str]:
    """Return updated tab labels including descriptive badge suffixes for active flags."""

    updated = list(tab_labels)

    def _append(label: str, variant: str) -> str:
        suffix = tab_badge_suffix(variant)
        descriptor = tab_badge_label(variant)
        icon = suffix.strip()
        if descriptor in label or (icon and icon in label):
            return label
        annotation = f"{suffix} {descriptor}" if icon else f" ({descriptor})"
        return f"{label}{annotation}"

    if flags.risk_alert and len(updated) > 2:
        updated[2] = _append(updated[2], "risk")
    if flags.upcoming_earnings and len(updated) > 3:
        updated[3] = _append(updated[3], "earnings")
    if flags.technical_signal and len(updated) > 4:
        updated[4] = _append(updated[4], "technical")
    return updated


def _render_snapshot_comparison_controls(viewmodel) -> None:
    snapshot_id = getattr(viewmodel, "snapshot_id", None)
    catalog = getattr(viewmodel, "snapshot_catalog", {}) or {}
    if not snapshot_id or not isinstance(catalog, Mapping):
        return

    options = list(catalog.get("portfolio", ()))
    options = [opt for opt in options if getattr(opt, "id", None) and opt.id != snapshot_id]
    if not options:
        return

    default_index = 0
    baseline_id = getattr(getattr(viewmodel, "comparison", None), "reference_id", None)
    if baseline_id:
        for idx, opt in enumerate(options):
            if opt.id == baseline_id:
                default_index = idx
                break

    labels = [getattr(opt, "label", str(idx)) for idx, opt in enumerate(options)]
    selected_idx = st.selectbox(
        "Comparar portafolio con",
        options=range(len(options)),
        format_func=lambda i: labels[i],
        index=min(default_index, len(options) - 1),
        key="portfolio_snapshot_compare",
    )

    try:
        selected = options[selected_idx]
    except (IndexError, TypeError):
        return

    comparison = snapshot_service.compare_snapshots(snapshot_id, getattr(selected, "id", None))
    if not comparison:
        st.info("No hay datos comparativos disponibles para esta selección.")
        return

    _render_snapshot_metrics(comparison, getattr(selected, "label", "Snapshot"))


def _render_snapshot_metrics(comparison: Mapping[str, Any], label: str) -> None:
    totals_a = comparison.get("totals_a") if isinstance(comparison, Mapping) else {}
    totals_b = comparison.get("totals_b") if isinstance(comparison, Mapping) else {}
    delta = comparison.get("delta") if isinstance(comparison, Mapping) else {}

    st.caption(f"Evolución frente a {label}")
    metrics = (
        ("Valorizado actual", "total_value"),
        ("Costo actual", "total_cost"),
        ("P/L actual", "total_pl"),
    )
    cols = st.columns(len(metrics))
    for col, (title, key) in zip(cols, metrics):
        current_val = _as_float_or_none((totals_a or {}).get(key))
        baseline_val = _as_float_or_none((totals_b or {}).get(key))
        delta_val = _as_float_or_none((delta or {}).get(key))
        if current_val is None:
            continue
        delta_text = None if delta_val is None else format_money(delta_val)
        col.metric(title, format_money(current_val), delta=delta_text)
        if baseline_val is not None:
            col.caption(f"Referencia: {format_money(baseline_val)}")


def render_basic_tab(viewmodel, favorites, snapshot) -> None:
    """Render the summary view for the basic portfolio tab."""

    _render_snapshot_comparison_controls(viewmodel)
    render_basic_section(
        viewmodel.positions,
        viewmodel.controls,
        viewmodel.metrics.ccl_rate,
        favorites=favorites,
        totals=viewmodel.totals,
        historical_total=viewmodel.historical_total,
        contribution_metrics=viewmodel.contributions,
        snapshot=snapshot,
    )


def render_risk_tab(
    df_view,
    tasvc: TAService,
    favorites,
    notifications: NotificationFlags,
    *,
    available_types: Sequence[str] | None = None,
) -> None:
    """Render risk analysis information for the given snapshot."""

    render_risk_analysis(
        df_view,
        tasvc,
        favorites=favorites,
        notifications=notifications,
        available_types=available_types,
    )


def render_fundamentals_tab(
    df_view,
    tasvc: TAService,
    favorites,
    notifications: NotificationFlags,
) -> None:
    """Render fundamentals tab using the given data sources."""

    render_fundamental_analysis(
        df_view,
        tasvc,
        favorites=favorites,
        notifications=notifications,
    )


def render_notifications_panel(
    favorites,
    notifications: NotificationFlags,
    *,
    ui: Any = st,
) -> None:
    """Render badges and indicators for the notifications panel."""

    if notifications.technical_signal:
        render_technical_badge(
            help_text="Tenés señales técnicas recientes para revisar en tus activos favoritos.",
        )
    render_favorite_badges(
        favorites,
        empty_message="⭐ Aún no marcaste favoritos para seguimiento rápido.",
    )


def _select_first(options: Iterable[str]) -> str | None:
    for item in options:
        return item
    return None


def render_technical_tab(
    tasvc: TAService,
    favorites,
    notifications: NotificationFlags,
    all_symbols: list[str],
    viewmodel,
    *,
    map_symbol: Callable[[str], str] | None = None,
    ui: Any | None = None,
    timer: Callable[[], float] = time.perf_counter,
    record_latency: Callable[[str, float | None, str], None] = record_tab_latency,
    plot_chart: Callable[..., Any] | None = None,
    render_fundamentals: Callable[[Mapping[str, Any]], None] | None = None,
) -> None:
    """Render the technical indicators tab for a specific symbol selection."""

    if map_symbol is None:
        map_symbol = map_to_us_ticker
    if ui is None:
        ui = st
    if plot_chart is None:
        plot_chart = plot_technical_analysis_chart
    if render_fundamentals is None:
        render_fundamentals = render_fundamental_data

    ui.subheader("Indicadores técnicos por activo")
    render_notifications_panel(favorites, notifications, ui=ui)
    if not all_symbols:
        ui.info("No hay símbolos en el portafolio para analizar.")
        return
    all_symbols_vm = list(viewmodel.metrics.all_symbols)
    if not all_symbols_vm:
        ui.info("No hay símbolos en el portafolio para analizar.")
        return

    options = favorites.sort_options(all_symbols_vm)
    if not options:
        options = all_symbols_vm
    sym = ui.selectbox(
        "Seleccioná un símbolo (CEDEAR / ETF)",
        options=options,
        index=favorites.default_index(options),
        key="ta_symbol",
        format_func=favorites.format_symbol,
    )
    if not sym:
        sym = _select_first(options)
    if not sym:
        ui.info("No hay símbolos en el portafolio para analizar.")
        return

    render_favorite_toggle(
        sym,
        favorites,
        key_prefix="ta",
        help_text="Los favoritos quedan disponibles en todas las secciones.",
    )

    try:
        us_ticker = map_symbol(sym)
    except ValueError:
        ui.info("No se encontró ticker US para este activo.")
        return

    try:
        fundamental_data = tasvc.fundamentals(us_ticker) or {}
    except AppError as err:
        ui.error(str(err))
    except Exception:
        logger.exception("Error al obtener datos fundamentales para %s", sym)
        ui.error("No se pudieron obtener datos fundamentales, intente más tarde")
    else:
        render_fundamentals(fundamental_data)

    cols = ui.columns([1, 1, 1, 1])
    with cols[0]:
        period = ui.selectbox("Período", ["3mo", "6mo", "1y", "2y"], index=1)
    with cols[1]:
        interval = ui.selectbox("Intervalo", ["1d", "1h", "30m"], index=0)
    with cols[2]:
        sma_fast = ui.number_input(
            "SMA corta",
            min_value=5,
            max_value=100,
            value=20,
            step=1,
        )
    with cols[3]:
        sma_slow = ui.number_input(
            "SMA larga",
            min_value=10,
            max_value=250,
            value=50,
            step=5,
        )

    with ui.expander("Parámetros adicionales"):
        c1, c2, c3 = ui.columns(3)
        macd_fast = c1.number_input(
            "MACD rápida", min_value=5, max_value=50, value=12, step=1
        )
        macd_slow = c2.number_input(
            "MACD lenta", min_value=10, max_value=200, value=26, step=1
        )
        macd_signal = c3.number_input(
            "MACD señal", min_value=5, max_value=50, value=9, step=1
        )
        c4, c5, c6 = ui.columns(3)
        atr_win = c4.number_input(
            "ATR ventana", min_value=5, max_value=200, value=14, step=1
        )
        stoch_win = c5.number_input(
            "Estocástico ventana", min_value=5, max_value=200, value=14, step=1
        )
        stoch_smooth = c6.number_input(
            "Estocástico suavizado", min_value=1, max_value=50, value=3, step=1
        )
        c7, c8, c9 = ui.columns(3)
        ichi_conv = c7.number_input(
            "Ichimoku conv.", min_value=1, max_value=50, value=9, step=1
        )
        ichi_base = c8.number_input(
            "Ichimoku base", min_value=2, max_value=100, value=26, step=1
        )
        ichi_span = c9.number_input(
            "Ichimoku span B", min_value=2, max_value=200, value=52, step=1
        )

    indicator_latency: float | None = None
    try:
        start_time = timer()
        df_ind = tasvc.indicators_for(
            sym,
            period=period,
            interval=interval,
            sma_fast=sma_fast,
            sma_slow=sma_slow,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            atr_win=atr_win,
            stoch_win=stoch_win,
            stoch_smooth=stoch_smooth,
            ichi_conv=ichi_conv,
            ichi_base=ichi_base,
            ichi_span=ichi_span,
        )
        indicator_latency = (timer() - start_time) * 1000.0
    except AppError as err:
        if indicator_latency is None:
            indicator_latency = (timer() - start_time) * 1000.0
        record_latency("tecnico", indicator_latency, status="error")
        ui.error(str(err))
        return
    except Exception:
        logger.exception("Error al obtener indicadores técnicos para %s", sym)
        if indicator_latency is None:
            indicator_latency = (timer() - start_time) * 1000.0
        record_latency("tecnico", indicator_latency, status="error")
        ui.error("No se pudieron obtener indicadores técnicos, intente más tarde")
        return
    record_latency("tecnico", indicator_latency, status="success")
    if df_ind.empty:
        ui.info("No se pudo descargar histórico para ese símbolo/periodo/intervalo.")
    else:
        fig = plot_chart(df_ind, sma_fast, sma_slow)
        ui.plotly_chart(
            fig,
            width="stretch",
            key="ta_chart",
            config=PLOTLY_CONFIG,
        )
        ui.caption(
            "Gráfico de precio con indicadores técnicos como "
            "medias móviles, RSI o MACD para detectar tendencias "
            "y señales."
        )
        alerts = tasvc.alerts_for(df_ind)
        if alerts:
            for a in alerts:
                al = a.lower()
                if "bajista" in al or "sobrecompra" in al:
                    ui.warning(a)
                elif "alcista" in al or "sobreventa" in al:
                    ui.success(a)
                else:
                    ui.info(a)
        else:
            ui.caption("Sin alertas técnicas en la última vela.")

        ui.subheader("Backtesting")
        strat = ui.selectbox(
            "Estrategia", ["SMA", "MACD", "Estocástico", "Ichimoku"], index=0
        )
        backtest_latency: float | None = None
        try:
            start_time = timer()
            bt = tasvc.backtest(df_ind, strategy=strat)
            backtest_latency = (timer() - start_time) * 1000.0
        except AppError as err:
            if backtest_latency is None:
                backtest_latency = (timer() - start_time) * 1000.0
            record_latency("tecnico", backtest_latency, status="error")
            ui.error(str(err))
            return
        except Exception:
            logger.exception("Error al ejecutar backtesting para %s", sym)
            if backtest_latency is None:
                backtest_latency = (timer() - start_time) * 1000.0
            record_latency("tecnico", backtest_latency, status="error")
            ui.error("No se pudo ejecutar el backtesting, intente más tarde")
            return
        record_latency("tecnico", backtest_latency, status="success")
        if bt.empty:
            ui.info("Sin datos suficientes para el backtesting.")
        else:
            ui.line_chart(bt["equity"])
            ui.caption(
                "La línea muestra cómo habría crecido la inversión usando la estrategia seleccionada."
            )
            ui.metric("Retorno acumulado", f"{bt['equity'].iloc[-1] - 1:.2%}")


def render_portfolio_section(
    container,
    cli,
    fx_rates,
    *,
    view_model_service_factory: Callable[[], PortfolioViewModelService] | None = None,
    notifications_service_factory: Callable[[], NotificationsService] | None = None,
    timings: dict[str, float] | None = None,
) -> Any:
    """Render the main portfolio section and return refresh interval."""
    with container:
        psvc = get_portfolio_service()
        tasvc = get_ta_service()

        view_model_service = get_portfolio_view_service(view_model_service_factory)
        notifications_service = get_notifications_service(notifications_service_factory)

        if snapshot_service.is_null_backend():
            backend_name = snapshot_service.current_backend_name()
            st.warning(
                "El almacenamiento de snapshots está deshabilitado "
                f"(backend: {backend_name}). Configurá `SNAPSHOT_BACKEND` a "
                "`json` o `sqlite` en `config.json` (o llamando a "
                "`services.snapshots.configure_storage`) y verificá los permisos "
                "definidos en `SNAPSHOT_STORAGE_PATH` para volver a habilitarlo."
            )

        with _record_stage("load_data", timings):
            df_pos, all_symbols, available_types = load_portfolio_data(cli, psvc)

        favorites = _get_cached_favorites()

        with _record_stage("apply_filters", timings):
            controls: Controls = render_sidebar(
                all_symbols,
                available_types,
            )

        refresh_secs = controls.refresh_secs
        with _record_stage("build_viewmodel", timings):
            snapshot = view_model_service.get_portfolio_view(
                df_pos=df_pos,
                controls=controls,
                cli=cli,
                psvc=psvc,
            )

            viewmodel = build_portfolio_viewmodel(
                snapshot=snapshot,
                controls=controls,
                fx_rates=fx_rates,
                all_symbols=all_symbols,
            )

        try:
            st.session_state["portfolio_last_viewmodel"] = viewmodel
            st.session_state["portfolio_last_positions"] = viewmodel.positions
            st.session_state["portfolio_last_totals"] = viewmodel.totals
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug("No se pudo almacenar el viewmodel en session_state", exc_info=True)

        with _record_stage("notifications", timings):
            notifications = notifications_service.get_flags()
        tab_labels = _apply_tab_badges(list(viewmodel.tab_options), notifications)

        tab_idx = st.radio(
            "Secciones",
            options=range(len(tab_labels)),
            format_func=lambda i: tab_labels[i],
            horizontal=True,
            key="portfolio_tab",
        )
        df_view = viewmodel.positions

        try:
            base_label = viewmodel.tab_options[tab_idx]
        except (IndexError, TypeError):
            base_label = str(tab_idx)
        tab_slug = _slugify_metric_label(base_label)

        _set_active_tab(tab_slug)
        render_cache = _ensure_render_cache()
        cache_entry = _ensure_tab_cache(render_cache, tab_slug)
        tab_signature = _tab_signature(viewmodel, df_view, tab_slug)

        with _record_stage(f"render_tab.{tab_slug}", timings):
            should_render = cache_entry.get("signature") != tab_signature or not cache_entry.get(
                "rendered"
            )
            source = "fresh" if not cache_entry.get("rendered") else "cache"
            latency_ms: float | None = cache_entry.get("latency_ms")
            if should_render:
                if cache_entry.get("rendered"):
                    source = "hot"
                body_placeholder = cache_entry["body_placeholder"]
                body_placeholder.empty()
                spinner_message = _loading_message(base_label)
                start = time.perf_counter()
                with body_placeholder.container():
                    with st.spinner(spinner_message):
                        _render_selected_tab(
                            tab_idx,
                            df_view,
                            tasvc,
                            favorites,
                            notifications,
                            available_types,
                            all_symbols,
                            viewmodel,
                            snapshot,
                        )
                latency_ms = (time.perf_counter() - start) * 1000.0
                cache_entry["signature"] = tab_signature
                cache_entry["rendered"] = True
                cache_entry["latency_ms"] = latency_ms
                record_tab_latency(tab_slug, latency_ms, status=source)
            else:
                record_tab_latency(tab_slug, 0.0, status="cache")

            cache_entry["last_source"] = source
            _update_status_message(
                cache_entry["info_placeholder"],
                base_label,
                latency_ms,
                source,
            )

        return refresh_secs
@contextmanager
def _record_stage(name: str, timings: dict[str, float] | None = None) -> Iterator[None]:
    """Measure a render stage while recording diagnostics and timings."""

    start = time.perf_counter()
    metric_name = f"portfolio_ui.{name}"
    with measure_execution(metric_name):
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            if timings is not None:
                timings[name] = round(elapsed, 2)


def _slugify_metric_label(label: str) -> str:
    """Return a slug suitable for metric identifiers based on ``label``."""

    normalized = unicodedata.normalize("NFKD", str(label))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")
    return normalized or "tab"


@st.cache_resource(show_spinner=False)
def _get_cached_favorites() -> FavoriteSymbols:
    """Return the persistent favorites manager using Streamlit's resource cache."""

    return get_persistent_favorites()


def _set_active_tab(tab_slug: str) -> None:
    """Persist the currently active tab slug in the session state."""

    try:
        st.session_state["active_tab"] = tab_slug
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudo actualizar active_tab en session_state", exc_info=True)


def _ensure_render_cache() -> dict[str, dict[str, Any]]:
    """Return the mutable render cache stored in the session state."""

    cache = st.session_state.get("render_cache")
    if not isinstance(cache, dict):
        cache = {}
        st.session_state["render_cache"] = cache
    return cache


def _ensure_tab_cache(cache: dict[str, dict[str, Any]], tab_slug: str) -> dict[str, Any]:
    """Ensure cache scaffolding for ``tab_slug`` and return the entry."""

    entry: dict[str, Any]
    raw_entry = cache.get(tab_slug)
    if isinstance(raw_entry, dict):
        entry = raw_entry
    else:
        entry = {}
        cache[tab_slug] = entry

    info_placeholder = entry.get("info_placeholder")
    if not hasattr(info_placeholder, "markdown"):
        info_placeholder = st.empty()
        entry["info_placeholder"] = info_placeholder

    body_placeholder = entry.get("body_placeholder")
    if not hasattr(body_placeholder, "container"):
        body_placeholder = st.empty()
        entry["body_placeholder"] = body_placeholder

    return entry


def _tab_signature(viewmodel: Any, df_view: Any, tab_slug: str) -> tuple[Any, ...]:
    """Build a lightweight signature to detect content changes per tab."""

    snapshot_id = getattr(viewmodel, "snapshot_id", None)
    rows = 0
    symbols: tuple[str, ...] = ()
    total_value = None
    try:
        if hasattr(df_view, "shape"):
            rows = int(getattr(df_view, "shape", (0,))[0])
        values = getattr(df_view, "get", lambda _: None)("valor_actual")
        if values is not None:
            total_value = float(values.sum())  # type: ignore[assignment]
        raw_symbols = getattr(df_view, "get", lambda _: [])("simbolo")
        if raw_symbols is not None:
            symbols = tuple(str(sym) for sym in raw_symbols)
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudo calcular la firma de la pestaña %s", tab_slug, exc_info=True)
    return (tab_slug, snapshot_id, rows, symbols, total_value)


def _loading_message(base_label: str) -> str:
    """Return a user-friendly loading message for ``base_label``."""

    label = re.sub(r"^[^\w]+", "", base_label).strip()
    if not label:
        label = base_label.strip()
    label = label or "sección"
    return f"Cargando {label.lower()}…"


def _render_selected_tab(
    tab_idx: int,
    df_view,
    tasvc: TAService,
    favorites,
    notifications: NotificationFlags,
    available_types: Sequence[str] | None,
    all_symbols: list[str],
    viewmodel,
    snapshot,
) -> None:
    """Dispatch rendering to the appropriate tab implementation."""

    if tab_idx == 0:
        render_basic_tab(viewmodel, favorites, snapshot)
    elif tab_idx == 1:
        render_advanced_analysis(df_view, tasvc)
    elif tab_idx == 2:
        render_risk_tab(
            df_view,
            tasvc,
            favorites,
            notifications,
            available_types=available_types,
        )
    elif tab_idx == 3:
        render_fundamentals_tab(df_view, tasvc, favorites, notifications)
    else:
        render_technical_tab(
            tasvc,
            favorites,
            notifications,
            all_symbols,
            viewmodel,
        )


_SOURCE_LABELS = {
    "fresh": "cálculo inicial",
    "hot": "recalculado",
    "cache": "caché en memoria",
}


def _update_status_message(placeholder, base_label: str, latency_ms: float | None, source: str) -> None:
    """Render a status line with latency metadata for the active tab."""

    label = base_label.strip() or "Sección"
    latency_text = "–" if latency_ms is None else f"{latency_ms:.0f} ms"
    source_label = _SOURCE_LABELS.get(source, source)
    message = f"**{label}** · {latency_text} · Fuente: {source_label}"
    try:
        placeholder.markdown(message)
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudo actualizar el estado de la pestaña %s", base_label, exc_info=True)
