import json

from infrastructure.iol import client as iol_client


class _FailClient:
    def get_portfolio(self):
        raise RuntimeError("boom")


class _OkClient:
    def __init__(self, data):
        self._data = data

    def get_portfolio(self):
        return self._data


def test_portfolio_fallback(monkeypatch, tmp_path):
    cache_file = tmp_path / "cache.json"
    monkeypatch.setattr(iol_client, "PORTFOLIO_CACHE", cache_file)

    # Éxito inicial guarda cache
    adapter = iol_client.IOLClientAdapter.__new__(iol_client.IOLClientAdapter)
    adapter._cli = _OkClient({"activos": [1]})
    assert adapter.get_portfolio() == {"activos": [1]}
    assert json.loads(cache_file.read_text()) == {"activos": [1]}

    # Falla posterior usa cache
    adapter._cli = _FailClient()
    assert adapter.get_portfolio() == {"activos": [1]}

    # Sin cache devuelve estructura vacía
    cache_file.unlink()
    assert adapter.get_portfolio() == {"activos": []}
