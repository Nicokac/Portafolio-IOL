import json
from unittest.mock import MagicMock, patch

from infrastructure.iol.auth import IOLAuth
from infrastructure.iol.client import IOLClientAdapter


def test_iol_auth_login_and_clear_tokens(tmp_path):
    tokens_path = tmp_path / "tokens.json"
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"access_token": "abc"}

    with patch("requests.post", return_value=mock_resp):
        auth = IOLAuth("user", "pass", tokens_file=tokens_path)
        tokens = auth.login()
        assert tokens["access_token"] == "abc"
        assert json.loads(tokens_path.read_text()) == {"access_token": "abc"}
        auth.clear_tokens()
        assert not tokens_path.exists()


def test_iol_client_get_portfolio_uses_cache_on_failure(tmp_path):
    cache_file = tmp_path / "portfolio.json"
    mock_legacy = MagicMock()
    mock_legacy.get_portfolio.return_value = {"activos": [1]}

    with patch("infrastructure.iol.client.PORTFOLIO_CACHE", cache_file), \
         patch("infrastructure.iol.client._LegacyIOLClient", return_value=mock_legacy):
        cli = IOLClientAdapter("user", "pass", tokens_file=tmp_path / "tok.json")
        data1 = cli.get_portfolio()
        assert data1 == {"activos": [1]}
        assert json.loads(cache_file.read_text()) == {"activos": [1]}

        mock_legacy.get_portfolio.side_effect = Exception("boom")
        data2 = cli.get_portfolio()
        assert data2 == {"activos": [1]}
