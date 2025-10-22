import sys
from pathlib import Path

from controllers.portfolio import portfolio

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _StreamlitStub:
    """Stub streamlit module that mimics a missing session_state attribute."""

    def __init__(self, state_stub: object):
        self._state_stub = state_stub

    @property
    def session_state(self):
        raise AttributeError(f"session_state missing: {self._state_stub}")


def test_get_portfolio_view_service_without_session_state_is_not_shared(monkeypatch):
    """Ensure services are not reused when session_state is unavailable."""

    first_state_stub = object()
    second_state_stub = object()

    first_instance = object()
    second_instance = object()

    monkeypatch.setattr(portfolio, "st", _StreamlitStub(first_state_stub))
    service_one = portfolio.get_portfolio_view_service(lambda: first_instance)

    monkeypatch.setattr(portfolio, "st", _StreamlitStub(second_state_stub))
    service_two = portfolio.get_portfolio_view_service(lambda: second_instance)

    assert service_one is first_instance
    assert service_two is second_instance
    assert service_one is not service_two
