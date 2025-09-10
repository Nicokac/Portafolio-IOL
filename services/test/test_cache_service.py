import json

from infrastructure.cache import quote_cache


class _OkClient:
    def __init__(self, data):
        self._data = data
        self.calls = 0

    def get_quote(self, *, mercado: str, simbolo: str):
        self.calls += 1
        return self._data


class _FailClient:
    def get_quote(self, *args, **kwargs):
        raise AssertionError("should not be called")


def test_get_quote_cached_persists_and_uses_disk(monkeypatch, tmp_path):
    monkeypatch.setattr(quote_cache, "QUOTE_CACHE_DIR", tmp_path)

    cli = _OkClient({"last": 10, "chg_pct": 1})
    out = quote_cache.get_quote_cached(cli, "m", "s", ttl=60)
    assert out == {"last": 10, "chg_pct": 1}
    assert cli.calls == 1

    cache_file = tmp_path / "m_S.json"
    saved = json.loads(cache_file.read_text())
    assert saved["data"] == {"last": 10, "chg_pct": 1}

    quote_cache._QUOTE_CACHE.clear()
    out2 = quote_cache.get_quote_cached(_FailClient(), "m", "s", ttl=60)
    assert out2 == {"last": 10, "chg_pct": 1}
