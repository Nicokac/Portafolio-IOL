import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

pytestmark = pytest.mark.skip(
    reason="Legacy FX controller tests are deprecated in v0.7"
)

# NOTE: Suite legacy mantenida sólo para auditorías puntuales del flujo antiguo
# de FX. La cobertura activa se encuentra en `tests/controllers/` y estos casos
# se eliminarán una vez que confirmemos la paridad de escenarios.
from controllers.fx import render_fx_section


def test_render_fx_section_warns_when_no_rates():
    container = contextlib.nullcontext()
    mock_st = SimpleNamespace(session_state={}, warning=MagicMock())
    with patch('controllers.fx.st', mock_st), \
         patch('controllers.fx.render_spreads') as mock_spreads, \
         patch('controllers.fx.render_fx_history') as mock_hist:
        render_fx_section(container, rates=None)

    mock_st.warning.assert_called_once_with("No se pudieron obtener las cotizaciones del dólar.")
    mock_spreads.assert_called_once_with({})
    mock_hist.assert_not_called()


def test_render_fx_section_updates_history_and_renders():
    container = contextlib.nullcontext()
    mock_st = MagicMock()
    mock_st.session_state = {}
    timestamp = 123456
    rates = {'_ts': timestamp, 'ccl': 1, 'mep': 2, 'blue': 3, 'oficial': 4}
    with patch('controllers.fx.st', mock_st), \
         patch('controllers.fx.render_spreads') as mock_spreads, \
         patch('controllers.fx.render_fx_history') as mock_hist:
        render_fx_section(container, rates)

    mock_spreads.assert_called_once_with(rates)
    assert mock_st.warning.call_count == 0
    assert mock_st.session_state['fx_history'][0]['ts'] == timestamp
    mock_hist.assert_called_once()
