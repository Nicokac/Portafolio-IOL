"""Service adapter orchestrating macroeconomic providers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
)

from infrastructure.macro import (
    FredClient,
    MacroAPIError,
    MacroSeriesObservation,
    WorldBankClient,
)
from services.base_adapter import BaseProviderAdapter
from shared import settings as shared_settings

_DEFAULT_PROVIDER_SEQUENCE = ("fred",)
_PROVIDER_LABELS = {
    "fred": "FRED",
    "worldbank": "World Bank",
}


@dataclass
class MacroFetchResult:
    """Normalized outcome returned by :class:`MacroAdapter`."""

    provider: Optional[str]
    provider_label: Optional[str]
    entries: Dict[str, Dict[str, Any]]
    attempts: List[Dict[str, Any]]
    notes: List[str]
    missing_series: List[str]
    fallback_entries: Dict[str, Dict[str, Any]]
    last_reason: Optional[str]

    @property
    def latest(self) -> Optional[Dict[str, Any]]:
        if not self.attempts:
            return None
        return dict(self.attempts[-1])


class MacroAdapter(BaseProviderAdapter):
    """Combine macro providers (FRED, World Bank, fallback)."""

    def __init__(
        self,
        *,
        providers: Optional[Sequence[str]] = None,
        settings: Any = shared_settings,
        client_factories: Optional[Mapping[str, Callable[[], Any]]] = None,
        timer: Optional[Callable[[], float]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._settings = settings
        sequence = providers or self._parse_provider_sequence(
            getattr(settings, "macro_api_provider", ""),
        )
        super().__init__(
            sequence or _DEFAULT_PROVIDER_SEQUENCE,
            labels=_PROVIDER_LABELS,
            timer=timer,
            clock=clock,
        )
        self._client_factories = dict(client_factories or {})

    # Public API ---------------------------------------------------------
    def fetch(self, sectors: Sequence[str]) -> MacroFetchResult:
        """Fetch macro indicators for the given ``sectors``."""

        normalized_sectors = self._normalize_sectors(sectors)
        context = {
            "sectors": normalized_sectors,
            "client_cache": {},
        }
        adapter_result = super().run(**context)
        fallback_entries = self._build_fallback_entries(normalized_sectors)

        attempts = adapter_result.attempts
        last_reason = adapter_result.detail

        if adapter_result.provider is None:
            fallback_attempt = self._build_fallback_attempt(
                success=bool(fallback_entries),
                detail=last_reason,
            )
            attempts.append(fallback_attempt)

        return MacroFetchResult(
            provider=adapter_result.provider,
            provider_label=adapter_result.label,
            entries=dict(adapter_result.payload or {}),
            attempts=attempts,
            notes=list(adapter_result.notes),
            missing_series=list(adapter_result.missing_series or []),
            fallback_entries=fallback_entries,
            last_reason=last_reason,
        )

    # BaseProviderAdapter hooks -----------------------------------------
    def check_availability(self, provider: str, **context: Any) -> tuple[bool, Optional[str]]:
        client, reason = self._ensure_client(provider, **context)
        if client is None:
            return False, reason or "no disponible"
        return True, None

    def build_request(
        self,
        provider: str,
        **context: Any,
    ) -> tuple[Mapping[str, str], MutableMapping[str, Any]]:
        sectors = context.get("sectors") or []
        mapping: Dict[str, str] = {}
        missing: List[str] = []
        lookup = self._series_lookup(provider)
        for sector in sectors:
            series_id = lookup.get(sector.casefold())
            if series_id:
                mapping[sector] = series_id
            else:
                missing.append(sector)
        meta: Dict[str, Any] = {}
        if missing:
            meta["missing_series"] = missing
        if not mapping:
            meta["detail"] = "no hay series configuradas para los sectores seleccionados"
        return mapping, meta

    def call_provider(
        self,
        provider: str,
        request: Mapping[str, str],
        **context: Any,
    ) -> Dict[str, MacroSeriesObservation]:
        client_cache = context.get("client_cache") or {}
        client = client_cache.get(provider)
        if client is None:
            client, _ = self._ensure_client(provider, **context)
            if client is None:
                raise RuntimeError(f"{self.provider_label(provider)} no disponible")
            client_cache[provider] = client
        return client.get_latest_observations(request)

    def normalize_response(
        self,
        provider: str,
        response: Mapping[str, Any],
        request: Mapping[str, str],
        **context: Any,
    ) -> Dict[str, Dict[str, Any]]:
        entries: Dict[str, Dict[str, Any]] = {}
        if not isinstance(response, Mapping):
            return entries
        for sector, observation in response.items():
            if not isinstance(observation, MacroSeriesObservation):
                continue
            entries[sector] = {
                "value": observation.value,
                "as_of": observation.as_of,
            }
        return entries

    def is_successful(
        self,
        provider: str,
        payload: Mapping[str, Dict[str, Any]],
        request: Mapping[str, str],
        **context: Any,
    ) -> bool:
        return bool(payload)

    def describe_error(
        self,
        provider: str,
        exc: BaseException,
        request: Mapping[str, Any],
        **context: Any,
    ) -> str:
        if isinstance(exc, MacroAPIError):
            return str(exc)
        return str(exc) or exc.__class__.__name__

    def describe_empty(
        self,
        provider: str,
        payload: Mapping[str, Dict[str, Any]],
        request: Mapping[str, str],
        **context: Any,
    ) -> str:
        return f"{self.provider_label(provider)} no devolvió observaciones válidas"

    # Internal helpers ---------------------------------------------------
    def _ensure_client(self, provider: str, **context: Any) -> tuple[Any, Optional[str]]:
        cache = context.get("client_cache")
        if isinstance(cache, dict) and provider in cache:
            return cache[provider], None

        factory = self._client_factories.get(provider)
        if callable(factory):
            try:
                client = factory()
            except Exception as exc:  # pragma: no cover - delegated to tests
                return None, str(exc) or "error al inicializar"
            if client is None:
                return None, "cliente no disponible"
            if isinstance(cache, dict):
                cache[provider] = client
            return client, None

        builder = self._client_builders().get(provider)
        if builder is None:
            return None, "proveedor desconocido"
        try:
            client = builder()
        except Exception as exc:  # pragma: no cover - defensive guard
            return None, str(exc) or "error al inicializar"
        if isinstance(cache, dict):
            cache[provider] = client
        return client, None

    def _build_fallback_entries(self, sectors: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        lookup = self._fallback_lookup()
        entries: Dict[str, Dict[str, Any]] = {}
        for sector in sectors:
            entry = lookup.get(sector.casefold())
            if entry:
                entries[sector] = dict(entry)
        return entries

    def _build_fallback_attempt(self, *, success: bool, detail: Optional[str]) -> Dict[str, Any]:
        status = "success" if success else "unavailable"
        attempt: Dict[str, Any] = {
            "provider": "fallback",
            "provider_key": "fallback",
            "label": "Fallback",
            "provider_label": "Fallback",
            "status": status,
            "fallback": True,
            "ts": float(self._clock()),
        }
        if detail:
            attempt["detail"] = detail
        return attempt

    def _normalize_sectors(self, sectors: Sequence[str]) -> List[str]:
        normalized: List[str] = []
        seen: set[str] = set()
        for raw in sectors:
            text = str(raw or "").strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
        return normalized

    def _series_lookup(self, provider: str) -> Mapping[str, str]:
        normalized = provider.casefold()
        if normalized == "worldbank":
            source = getattr(self._settings, "world_bank_sector_series", {})
        else:
            source = getattr(self._settings, "fred_sector_series", {})
        mapping: Dict[str, str] = {}
        if isinstance(source, Mapping):
            for raw_label, raw_series in source.items():
                label = str(raw_label or "").strip()
                series_id = str(raw_series or "").strip()
                if not label or not series_id:
                    continue
                mapping[label.casefold()] = series_id
        return mapping

    @lru_cache(maxsize=1)
    def _fallback_lookup(self) -> Mapping[str, Dict[str, Any]]:
        source = getattr(self._settings, "macro_sector_fallback", {})
        normalized: Dict[str, Dict[str, Any]] = {}
        if isinstance(source, Mapping):
            for raw_label, entry in source.items():
                if not isinstance(entry, Mapping):
                    continue
                label = str(raw_label or "").strip()
                if not label:
                    continue
                value = entry.get("value")
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                as_of_raw = entry.get("as_of")
                as_of = None
                if as_of_raw is not None:
                    text = str(as_of_raw).strip()
                    if text:
                        as_of = text
                normalized[label.casefold()] = {
                    "value": numeric_value,
                    "as_of": as_of,
                }
        return normalized

    @staticmethod
    def _parse_provider_sequence(raw: Any) -> List[str]:
        if not raw:
            return []
        if isinstance(raw, str):
            return [part.strip() for part in raw.split(",") if part.strip()]
        if isinstance(raw, Iterable):
            return [str(item or "").strip() for item in raw if str(item or "").strip()]
        return []

    @lru_cache(maxsize=8)
    def _client_builders(self) -> Mapping[str, Any]:
        user_agent = getattr(self._settings, "USER_AGENT", "Portafolio-IOL/1.0 (+app)")

        def _build_fred() -> FredClient:
            api_key = getattr(self._settings, "fred_api_key", None)
            if not api_key:
                raise RuntimeError("FRED sin credenciales configuradas")
            base_url = getattr(self._settings, "fred_api_base_url", "https://api.stlouisfed.org/fred")
            try:
                rate_limit = int(getattr(self._settings, "fred_api_rate_limit_per_minute", 120) or 0)
            except (TypeError, ValueError):
                rate_limit = 0
            return FredClient(
                api_key,
                base_url=base_url,
                calls_per_minute=max(rate_limit, 0),
                user_agent=user_agent,
            )

        def _build_worldbank() -> WorldBankClient:
            api_key = getattr(self._settings, "world_bank_api_key", None)
            base_url = getattr(
                self._settings,
                "world_bank_api_base_url",
                "https://api.worldbank.org/v2",
            )
            try:
                rate_limit = int(getattr(self._settings, "world_bank_api_rate_limit_per_minute", 60) or 0)
            except (TypeError, ValueError):
                rate_limit = 0
            return WorldBankClient(
                api_key=api_key,
                base_url=base_url,
                calls_per_minute=max(rate_limit, 0),
                user_agent=user_agent,
            )

        return {
            "fred": _build_fred,
            "worldbank": _build_worldbank,
        }


__all__ = ["MacroAdapter", "MacroFetchResult"]
