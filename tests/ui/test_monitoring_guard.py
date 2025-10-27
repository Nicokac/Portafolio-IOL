import time
import types

import pandas as pd
import pytest

from services import portfolio_view as portfolio_view_mod
from shared.ui import monitoring_guard
from ui.helpers import preload as preload_mod


def _activate_monitoring(state):
    state["_monitoring_active_panel"] = {"module": "x", "attr": "y"}


def test_preload_skipped_when_monitoring_active(monkeypatch, streamlit_stub):
    streamlit_stub.reset()
    _activate_monitoring(streamlit_stub.session_state)
    monkeypatch.setattr(preload_mod, "st", streamlit_stub, raising=False)
    monkeypatch.setattr(monitoring_guard, "st", streamlit_stub, raising=False)
    captured: list[tuple[str, dict | None]] = []
    monkeypatch.setattr(
        preload_mod,
        "log_metric",
        lambda name, *, context=None, **_: captured.append((name, context)),
    )

    result = preload_mod.ensure_scientific_preload_ready(streamlit_stub)

    assert result is False
    assert ("monitoring.refresh_skipped", {"source": "scientific_preload"}) in captured


def test_monitoring_debounce_skips_consistency_refresh(monkeypatch, streamlit_stub):
    streamlit_stub.reset()
    _activate_monitoring(streamlit_stub.session_state)
    now_ts = time.time()
    streamlit_stub.session_state["_monitoring_last_refresh_ts"] = now_ts

    monkeypatch.setattr(portfolio_view_mod, "st", streamlit_stub, raising=False)
    monkeypatch.setattr(monitoring_guard, "st", streamlit_stub, raising=False)
    monkeypatch.setattr(portfolio_view_mod, "is_monitoring_active", lambda: True)
    monkeypatch.setattr(portfolio_view_mod, "_get_dataset_cache_adapter", lambda: None)

    service = portfolio_view_mod.PortfolioViewModelService()
    service._consistency_guard_active = True

    df_pos = pd.DataFrame({"valor_actual": [100.0]})
    controls = types.SimpleNamespace(selected_syms=[], selected_types=[], symbol_query="")

    metrics: list[tuple[str, dict | None]] = []
    monkeypatch.setattr(
        portfolio_view_mod,
        "log_metric",
        lambda name, context=None, **_: metrics.append((name, context)),
    )

    def _stop_after_monitoring(self, *_args, **_kwargs):
        raise StopIteration

    monkeypatch.setattr(
        portfolio_view_mod.PortfolioViewModelService,
        "_should_invalidate_cache",
        _stop_after_monitoring,
    )

    with pytest.raises(StopIteration):
        service._compute_viewmodel_phase(
            df_pos=df_pos,
            controls=controls,
            cli=object(),
            psvc=object(),
            include_extended=False,
            telemetry_phase="test",
            allow_pending_reuse=True,
            dataset_hash=None,
            skip_invalidation=False,
        )

    assert any(name == "monitoring.refresh_debounced" for name, _ in metrics)
    assert streamlit_stub.session_state["_monitoring_last_refresh_ts"] == pytest.approx(now_ts)
