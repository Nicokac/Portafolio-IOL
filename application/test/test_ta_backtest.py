import pandas as pd

from application.ta_service import run_backtest


def test_run_backtest_sma_positive_return():
    df = pd.DataFrame(
        {
            "Close": [100, 102, 104, 106, 108],
            "SMA_FAST": [5] * 5,
            "SMA_SLOW": [1] * 5,
        }
    )
    bt = run_backtest(df, strategy="sma")
    assert not bt.empty
    assert bt["equity"].iloc[-1] > 1.07