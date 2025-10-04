"""Generic helpers to orchestrate sequential provider adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
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
    Tuple,
)

from services.health import record_adapter_fallback


@dataclass
class AdapterResult:
    """Normalized payload returned by :class:`BaseProviderAdapter`."""

    provider: Optional[str]
    label: Optional[str]
    payload: Any
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    missing_series: List[str] = field(default_factory=list)
    detail: Optional[str] = None


class BaseProviderAdapter:
    """Template helper that runs providers sequentially until one succeeds."""

    def __init__(
        self,
        providers: Sequence[str] | None,
        *,
        labels: Optional[Mapping[str, str]] = None,
        timer: Optional[Callable[[], float]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        normalized: List[str] = []
        if providers:
            for provider in providers:
                text = str(provider or "").strip()
                if not text:
                    continue
                normalized.append(text.casefold())
        self._providers: Tuple[str, ...] = tuple(normalized)
        label_mapping: Dict[str, str] = {}
        if labels:
            for key, value in labels.items():
                normalized_key = str(key or "").strip().casefold()
                if not normalized_key:
                    continue
                label_mapping[normalized_key] = str(value or "").strip() or key
        self._labels = label_mapping
        self._timer = timer or time.perf_counter
        self._clock = clock or time.time

    # Public API ---------------------------------------------------------
    def run(self, **context: Any) -> AdapterResult:
        """Execute the provider sequence until one returns a payload."""

        attempts: List[Dict[str, Any]] = []
        failure_notes: List[str] = []
        last_reason: Optional[str] = None

        for provider in self._providers:
            attempt, payload, missing = self._attempt(provider, **context)
            attempts.append(dict(attempt))
            record_adapter_fallback(
                adapter=self.__class__.__name__,
                provider=provider,
                status=attempt.get("status"),
                fallback=len(attempts) > 1,
            )

            status = str(attempt.get("status") or "").casefold()
            if status == "success":
                return AdapterResult(
                    provider=provider,
                    label=attempt.get("provider_label") or attempt.get("label"),
                    payload=payload,
                    attempts=attempts,
                    notes=failure_notes,
                    missing_series=list(missing or []),
                    detail=None,
                )

            detail = attempt.get("detail")
            if detail:
                last_reason = str(detail)
            failure_note = self.describe_failure(provider, attempt, **context)
            if failure_note:
                failure_notes.append(failure_note)

        return AdapterResult(
            provider=None,
            label=None,
            payload=None,
            attempts=attempts,
            notes=failure_notes,
            missing_series=[],
            detail=last_reason,
        )

    # Hooks for subclasses -----------------------------------------------
    def providers(self) -> Tuple[str, ...]:
        """Return the normalized provider sequence."""

        return self._providers

    def provider_label(self, provider: str) -> str:
        """Return a human-readable label for ``provider``."""

        normalized = str(provider or "").casefold()
        return self._labels.get(normalized, provider.upper() if provider else "desconocido")

    def check_availability(self, provider: str, **context: Any) -> Tuple[bool, Optional[str]]:
        """Return whether ``provider`` is enabled for the current run."""

        return True, None

    def build_request(
        self, provider: str, **context: Any
    ) -> Tuple[Mapping[str, Any], MutableMapping[str, Any]]:
        """Return the mapping used to query ``provider`` and metadata."""

        raise NotImplementedError

    def call_provider(
        self, provider: str, request: Mapping[str, Any], **context: Any
    ) -> Any:
        """Invoke ``provider`` with ``request`` and return the raw payload."""

        raise NotImplementedError

    def normalize_response(
        self,
        provider: str,
        response: Any,
        request: Mapping[str, Any],
        **context: Any,
    ) -> Any:
        """Convert the raw payload from ``provider`` into a normalized form."""

        return response

    def is_successful(
        self,
        provider: str,
        payload: Any,
        request: Mapping[str, Any],
        **context: Any,
    ) -> bool:
        """Return whether ``payload`` represents a successful attempt."""

        return bool(payload)

    def describe_error(
        self,
        provider: str,
        exc: BaseException,
        request: Mapping[str, Any],
        **context: Any,
    ) -> str:
        """Return a human readable description for ``exc``."""

        return str(exc)

    def describe_empty(
        self,
        provider: str,
        payload: Any,
        request: Mapping[str, Any],
        **context: Any,
    ) -> str:
        """Return a description when ``provider`` returns no data."""

        return "sin datos disponibles"

    def describe_missing_request(
        self,
        provider: str,
        meta: Mapping[str, Any],
        **context: Any,
    ) -> str:
        """Return a description when ``build_request`` yields no mapping."""

        return "configuración incompleta"

    def describe_failure(
        self, provider: str, attempt: Mapping[str, Any], **context: Any
    ) -> Optional[str]:
        """Return a note summarising ``attempt`` when it fails."""

        detail = attempt.get("detail")
        if not detail:
            return None
        label = attempt.get("provider_label") or attempt.get("label") or provider
        status = str(attempt.get("status") or "").casefold()
        if status == "disabled":
            return f"{label} no disponible: {detail}"
        if status == "error":
            return f"{label} no disponible: {detail}"
        if status == "unavailable":
            return f"{label}: {detail}"
        return f"{label}: {detail}"

    def on_error(
        self,
        provider: str,
        exc: BaseException,
        attempt: MutableMapping[str, Any],
        **context: Any,
    ) -> None:
        """Hook called when ``provider`` raises an exception."""

    def on_empty(
        self,
        provider: str,
        payload: Any,
        attempt: MutableMapping[str, Any],
        **context: Any,
    ) -> None:
        """Hook called when ``provider`` returns an empty payload."""

    # Internal helpers ---------------------------------------------------
    def _normalize_missing(self, missing: Any) -> List[str]:
        if not missing:
            return []
        items: List[str] = []
        if isinstance(missing, str):
            text = missing.strip()
            if text:
                items.append(text)
        elif isinstance(missing, Iterable) and not isinstance(
            missing, (bytes, bytearray, str)
        ):
            for item in missing:
                text = str(item or "").strip()
                if text:
                    items.append(text)
        return sorted(set(items))

    def _attempt(
        self,
        provider: str,
        **context: Any,
    ) -> Tuple[Dict[str, Any], Any, List[str]]:
        label = self.provider_label(provider)
        attempt: Dict[str, Any] = {
            "provider": provider,
            "provider_key": provider,
            "label": label,
            "provider_label": label,
        }

        enabled, reason = self.check_availability(provider, **context)
        if not enabled:
            attempt["status"] = "disabled"
            if reason:
                attempt["detail"] = str(reason)
            attempt["ts"] = float(self._clock())
            return attempt, None, []

        request, meta = self.build_request(provider, **context)
        missing = self._normalize_missing(meta.get("missing_series"))
        if missing:
            attempt["missing_series"] = missing

        if not request:
            detail = meta.get("detail") or self.describe_missing_request(
                provider, meta, **context
            )
            if detail:
                attempt["detail"] = str(detail)
            attempt["status"] = "error"
            attempt["ts"] = float(self._clock())
            return attempt, None, missing

        start = float(self._timer())
        try:
            response = self.call_provider(provider, request, **context)
        except BaseException as exc:  # pragma: no cover - normalized below
            elapsed_ms = (float(self._timer()) - start) * 1000.0
            attempt["status"] = "error"
            attempt["elapsed_ms"] = elapsed_ms
            detail = self.describe_error(provider, exc, request, **context)
            if detail:
                attempt["detail"] = str(detail)
            self.on_error(provider, exc, attempt, **context)
            attempt["ts"] = float(self._clock())
            return attempt, None, missing

        elapsed_ms = (float(self._timer()) - start) * 1000.0
        attempt["elapsed_ms"] = elapsed_ms
        payload = self.normalize_response(provider, response, request, **context)

        if not self.is_successful(provider, payload, request, **context):
            detail = self.describe_empty(provider, payload, request, **context)
            if detail:
                attempt["detail"] = str(detail)
            attempt["status"] = "error"
            self.on_empty(provider, payload, attempt, **context)
            attempt["ts"] = float(self._clock())
            return attempt, None, missing

        attempt["status"] = "success"
        attempt["ts"] = float(self._clock())
        return attempt, payload, missing


__all__ = ["AdapterResult", "BaseProviderAdapter"]

