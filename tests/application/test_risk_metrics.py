import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from application.risk_service import drawdown_series


def test_drawdown_series_empty():
    result = drawdown_series(pd.Series(dtype=float))
    assert isinstance(result, pd.Series)
    assert result.empty


def test_drawdown_series_non_empty():
    returns = pd.Series([0.01, -0.02, 0.03])
    result = drawdown_series(returns)

    expected = pd.Series([0.0, -0.02, 0.0])

    assert not result.empty
    assert (result <= 0).all()
    pd.testing.assert_series_equal(
        result.reset_index(drop=True).round(8),
        expected.reset_index(drop=True).round(8),
        check_names=False,
    )
