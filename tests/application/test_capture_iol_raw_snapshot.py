from __future__ import annotations

from typing import Any

import pytest

from application import portfolio_service


class DummyClient:
    def __init__(self, portfolio_payload: dict[str, Any], quote_payload: dict[str, Any]) -> None:
        self._portfolio_payload = portfolio_payload
        self._quote_payload = quote_payload
        self.calls: list[tuple[str, bool]] = []

    def get_raw_portfolio(self, *, country: str) -> dict[str, Any]:
        self.calls.append(("portfolio", False))
        return self._portfolio_payload

    def get_raw_quote(self, *, mercado: str, simbolo: str, detalle: bool = False) -> dict[str, Any]:
        self.calls.append(("quote", detalle))
        assert mercado == "bcba"
        assert simbolo == "BPOC7"
        if detalle:
            return {"detalle": True, **self._quote_payload}
        return dict(self._quote_payload)


@pytest.fixture(autouse=True)
def _freeze_metrics(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any, Any]]:
    calls: list[tuple[str, Any, Any]] = []

    def fake_log_metric(name: str, *, context=None, status=None, duration_ms=None):
        calls.append((name, context, status))

    monkeypatch.setattr(portfolio_service, "log_metric", fake_log_metric)
    return calls


def test_capture_snapshot_combines_payloads(_freeze_metrics: list[tuple[str, Any, Any]]) -> None:
    portfolio_payload = {"activos": [{"simbolo": "BPOC7", "cantidad": 10}]}
    quote_payload = {"ultimoPrecio": 1377.0}
    client = DummyClient(portfolio_payload, quote_payload)

    snapshot = portfolio_service.capture_iol_raw_snapshot(client)

    assert snapshot["portfolio_raw"] == portfolio_payload
    assert snapshot["quote_raw"]["ultimoPrecio"] == 1377.0
    assert snapshot["quote_detail_raw"]["detalle"] is True
    assert snapshot["portfolio_row"] == {"simbolo": "BPOC7", "cantidad": 10}

    metric_names = [name for name, _, _ in _freeze_metrics]
    assert metric_names.count("debug_iol.raw_capture") == 1
    assert any(name == "debug_iol.payload_size_bytes" and ctx.get("block") == "portfolio" for name, ctx, _ in _freeze_metrics)
    assert any(name == "debug_iol.payload_size_bytes" and ctx.get("block") == "quote" for name, ctx, _ in _freeze_metrics)
    assert any(name == "debug_iol.payload_size_bytes" and ctx.get("block") == "quote_detail" for name, ctx, _ in _freeze_metrics)
    assert not any(name == "debug_iol.symbol_not_in_portfolio" for name, _, _ in _freeze_metrics)


def test_capture_snapshot_tracks_missing_symbol(_freeze_metrics: list[tuple[str, Any, Any]]) -> None:
    portfolio_payload = {"activos": [{"simbolo": "GGAL", "cantidad": 5}]}
    quote_payload = {"ultimoPrecio": 200.0}
    client = DummyClient(portfolio_payload, quote_payload)

    snapshot = portfolio_service.capture_iol_raw_snapshot(client)

    assert snapshot["portfolio_row"] is None
    assert any(name == "debug_iol.symbol_not_in_portfolio" for name, _, _ in _freeze_metrics)
