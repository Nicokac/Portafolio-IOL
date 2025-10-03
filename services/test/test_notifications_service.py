from __future__ import annotations

import pytest

from services.notifications import NotificationFlags, NotificationsService


def test_notification_flags_parses_various_payloads() -> None:
    payload = {
        "flags": {
            "risk": "true",
            "technical_signal": 1,
            "earnings": {"active": "yes"},
        }
    }
    flags = NotificationFlags.from_payload(payload)
    assert flags.risk_alert is True
    assert flags.technical_signal is True
    assert flags.upcoming_earnings is True


def test_notifications_service_without_url_returns_empty() -> None:
    service = NotificationsService(base_url=None)
    assert service.get_flags() == NotificationFlags()


def test_notifications_service_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class DummySession:
        def get(self, url: str, timeout: float | None = None):  # noqa: ANN001 - mimic requests API
            captured["called"] = True
            raise RuntimeError("boom")

    service = NotificationsService(base_url="https://example.com/api", session=DummySession())
    flags = service.get_flags()
    assert captured["called"] is True
    assert flags == NotificationFlags()
