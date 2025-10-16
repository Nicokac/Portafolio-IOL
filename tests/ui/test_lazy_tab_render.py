"""Tests for lazy tab rendering telemetry and deferral."""

import csv
from pathlib import Path

from tests.ui.test_portfolio_ui import _DummyContainer, FakeStreamlit, _portfolio_setup


def test_lazy_tab_render_records_metrics(monkeypatch, _portfolio_setup) -> None:
    metrics_path = Path("performance_metrics_11.csv")
    if metrics_path.exists():
        metrics_path.unlink()

    fake_st = FakeStreamlit(radio_sequence=[0])
    (
        portfolio_mod,
        basic,
        advanced,
        risk,
        fundamental,
        _technical_badge,
        view_model_factory,
        notifications_factory,
    ) = _portfolio_setup(fake_st)

    labels: list[str] = []

    class _ProfileStub:
        def __enter__(self) -> "_ProfileStub":  # pragma: no cover - trivial
            return self

        def __exit__(self, *_exc) -> bool:  # pragma: no cover - trivial
            return False

    def _fake_profile_block(label: str, **_kwargs):
        labels.append(label)
        return _ProfileStub()

    monkeypatch.setattr(portfolio_mod, "profile_block", _fake_profile_block)

    portfolio_mod.render_portfolio_section(
        _DummyContainer(),
        cli=object(),
        fx_rates={"ccl": 0.0},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    assert labels == ["render_tab.portafolio"]
    tab_loaded = fake_st.session_state.get("tab_loaded", {})
    assert tab_loaded == {"portafolio": True}

    assert basic.call_count == 3
    assert advanced.call_count == 0
    assert risk.call_count == 0
    assert fundamental.call_count == 0

    assert metrics_path.exists()
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows and rows[0]["tab_name"] == "portafolio"
    assert float(rows[0]["render_duration_s"]) >= 0.0

    try:
        metrics_path.unlink()
    except FileNotFoundError:  # pragma: no cover - defensive cleanup
        pass

