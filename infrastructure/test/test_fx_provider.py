import json
import time
from unittest.mock import MagicMock, patch

from infrastructure.fx.provider import FXProviderAdapter


def test_get_rates_returns_cached_data(tmp_path):
    cache_file = tmp_path / "fx_cache.json"
    data = {"oficial": 1.0, "_ts": time.time()}
    cache_file.write_text(json.dumps(data), encoding="utf-8")

    mock_session = MagicMock()
    with patch("infrastructure.fx.provider.CACHE_FILE", cache_file), \
         patch("infrastructure.fx.provider.build_session", return_value=mock_session):
        with FXProviderAdapter() as adapter:
            rates, err = adapter.get_rates()
        assert err is None
        assert rates["oficial"] == 1.0
        mock_session.get.assert_not_called()


def test_get_rates_fetches_and_saves_cache(tmp_path):
    cache_file = tmp_path / "fx_cache.json"

    def fake_get(url):
        resp = MagicMock()
        resp.ok = True
        if "bluelytics" in url:
            resp.json.return_value = {"blue": {"value_avg": 100}}
        else:
            resp.json.return_value = {"venta": 200}
        return resp

    mock_session = MagicMock()
    mock_session.get.side_effect = fake_get

    with patch("infrastructure.fx.provider.CACHE_FILE", cache_file), \
         patch("infrastructure.fx.provider.build_session", return_value=mock_session):
        with FXProviderAdapter() as adapter:
            rates, err = adapter.get_rates()
        assert rates["blue"] == 100
        assert rates["oficial"] == 200
        assert json.loads(cache_file.read_text(encoding="utf-8"))["blue"] == 100
        assert mock_session.get.call_count == 4
