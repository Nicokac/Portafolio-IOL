from unittest.mock import MagicMock, patch

from infrastructure.http.session import build_session


def test_build_session_user_agent_and_timeout():
    mock_request = MagicMock()
    with patch("requests.Session.request", mock_request):
        session = build_session("UA-Test", retries=1, backoff=0, timeout=5)
        assert session.headers["User-Agent"] == "UA-Test"

        session.request("GET", "http://example.com")
        assert mock_request.call_args.kwargs["timeout"] == 5

        mock_request.reset_mock()
        session.request("GET", "http://example.com", timeout=1)
        assert mock_request.call_args.kwargs["timeout"] == 1
