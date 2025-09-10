from unittest.mock import MagicMock, patch

import infrastructure.cache.quote_cache as qc


def test_get_quote_cached_memory_and_disk(tmp_path):
    mock_cli = MagicMock()
    mock_cli.get_quote.return_value = {"last": 1.0, "chg_pct": 2.0}

    with patch.object(qc, "QUOTE_CACHE_DIR", tmp_path):
        qc._QUOTE_CACHE.clear()
        data1 = qc.get_quote_cached(mock_cli, "bcba", "GGAL", ttl=10)
        assert data1 == {"last": 1.0, "chg_pct": 2.0}
        mock_cli.get_quote.assert_called_once()

        mock_cli.get_quote.reset_mock()
        data2 = qc.get_quote_cached(mock_cli, "bcba", "GGAL", ttl=10)
        assert data2 == data1
        mock_cli.get_quote.assert_not_called()

        qc._QUOTE_CACHE.clear()
        mock_cli2 = MagicMock()
        data3 = qc.get_quote_cached(mock_cli2, "bcba", "GGAL", ttl=10)
        assert data3 == data1
        mock_cli2.get_quote.assert_not_called()
