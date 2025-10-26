from __future__ import annotations


def test_iolclient_offline_stub_responds() -> None:
    from infrastructure.iol import client

    cli = client.IOLClient()
    payload = cli.get_portfolio()
    assert "activos" in payload and isinstance(payload["activos"], list)


def test_cache_stub_attributes() -> None:
    import services.cache as cache

    for attr in ("st", "IOLAuth", "record_fx_api_response", "clear_all"):
        assert hasattr(cache, attr)
