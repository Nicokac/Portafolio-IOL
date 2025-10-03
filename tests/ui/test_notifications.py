import pytest

import ui.notifications as notifications


class _DummyStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}
        self.markdowns: list[dict[str, object]] = []

    def markdown(self, body: str, *, unsafe_allow_html: bool = False) -> None:
        self.markdowns.append({"body": body, "unsafe": unsafe_allow_html})


@pytest.mark.parametrize(
    "render_func,variant",
    [
        (notifications.render_risk_badge, "risk"),
        (notifications.render_technical_badge, "technical"),
        (notifications.render_earnings_badge, "earnings"),
    ],
)
def test_notification_badges_render_expected_markup(monkeypatch: pytest.MonkeyPatch, render_func, variant: str) -> None:
    fake_st = _DummyStreamlit()
    monkeypatch.setattr(notifications, "st", fake_st)

    render_func(help_text="Detalle de prueba")

    assert fake_st.markdowns, "Se espera que el badge agregue contenido via markdown"
    html_calls = [entry for entry in fake_st.markdowns if entry["unsafe"]]
    assert html_calls, "El badge debe renderizar HTML seguro"
    last_call = html_calls[-1]["body"]
    assert f"notification-badge--{variant}" in last_call
    assert "Detalle de prueba" in last_call


def test_badge_styles_injected_once(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = _DummyStreamlit()
    monkeypatch.setattr(notifications, "st", fake_st)

    notifications.render_risk_badge()
    notifications.render_technical_badge()

    css_calls = [entry for entry in fake_st.markdowns if "<style" in entry["body"]]
    assert len(css_calls) == 1, "Los estilos CSS deben inyectarse una sola vez"


def test_tab_badge_suffixes_are_distinct() -> None:
    suffixes = {
        notifications.tab_badge_suffix("risk"),
        notifications.tab_badge_suffix("technical"),
        notifications.tab_badge_suffix("earnings"),
    }
    assert len({s.strip() for s in suffixes}) == 3
    assert all(suffix for suffix in suffixes)
