import logging
from pathlib import Path

import pandas as pd
import pytest
from requests.exceptions import HTTPError, Timeout

from application.ta_service import fetch_with_indicators


@pytest.mark.parametrize("exc_cls", [HTTPError, Timeout])
def test_fetch_uses_fallback_on_http_errors(monkeypatch, tmp_path, caplog, exc_cls):
    fetch_with_indicators.clear()

    root = tmp_path
    cache_dir = root / "infrastructure" / "cache"
    cache_dir.mkdir(parents=True)
    csv_path = cache_dir / "ta_fallback.csv"
    sample = pd.DataFrame(
        {
            "Open": range(1, 61),
            "High": range(2, 62),
            "Low": range(0, 60),
            "Close": range(1, 61),
            "Volume": [100] * 60,
        },
        index=pd.date_range("2023-01-01", periods=60),
    )
    sample.to_csv(csv_path)

    module_file = root / "application" / "ta_service.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("")
    monkeypatch.setattr("application.ta_service.__file__", str(module_file))

    def raise_exc(*args, **kwargs):
        raise exc_cls("boom")

    monkeypatch.setattr("application.ta_service.yf.download", raise_exc)
    monkeypatch.setattr("application.ta_service.map_to_us_ticker", lambda s: "AAPL")

    orig_read_csv = pd.read_csv

    def fake_read_csv(path, *args, **kwargs):
        assert Path(path) == csv_path
        return orig_read_csv(path, *args, **kwargs)

    monkeypatch.setattr("application.ta_service.pd.read_csv", fake_read_csv)

    with caplog.at_level(logging.WARNING):
        df = fetch_with_indicators("AAPL")

    assert not df.empty
    expected = orig_read_csv(csv_path, index_col=0, parse_dates=True)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    pd.testing.assert_frame_equal(
        df[cols],
        expected.loc[df.index, cols],
    )
    assert "Usando fallback local" in caplog.text


@pytest.mark.parametrize(
    "mode, message",
    [
        ("missing", "No se encontró el archivo de fallback"),
        ("corrupt", "No se pudo leer el fallback"),
    ],
)
def test_empty_dataframe_when_fallback_missing_or_corrupt(
    monkeypatch, tmp_path, caplog, mode, message
):
    fetch_with_indicators.clear()

    root = tmp_path
    cache_dir = root / "infrastructure" / "cache"
    cache_dir.mkdir(parents=True)
    csv_path = cache_dir / "ta_fallback.csv"

    if mode == "corrupt":
        csv_path.touch()

        def bad_csv(*args, **kwargs):
            raise ValueError("bad csv")

        monkeypatch.setattr("application.ta_service.pd.read_csv", bad_csv)

    module_file = root / "application" / "ta_service.py"
    module_file.parent.mkdir(parents=True)
    module_file.write_text("")
    monkeypatch.setattr("application.ta_service.__file__", str(module_file))

    def raise_http(*args, **kwargs):
        raise HTTPError("boom")

    monkeypatch.setattr("application.ta_service.yf.download", raise_http)
    monkeypatch.setattr("application.ta_service.map_to_us_ticker", lambda s: "AAPL")

    with caplog.at_level(logging.ERROR):
        df = fetch_with_indicators("AAPL")

    assert df.empty
    assert message in caplog.text
