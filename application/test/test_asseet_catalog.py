import json
from application.portfolio_service import classify_symbol
from infrastructure import asset_catalog

def test_catalog_overrides_config(monkeypatch, tmp_path):
    data = {"XYZ": "Bono"}
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data))
    monkeypatch.setenv("ASSET_CATALOG_PATH", str(path))
    asset_catalog.get_asset_catalog.cache_clear()
    assert classify_symbol("xyz") == "Bono"

def test_catalog_fallback_to_patterns(monkeypatch, tmp_path):
    # empty catalog, should use existing regex pattern for S10N5 -> Letra
    path = tmp_path / "empty.json"
    path.write_text("{}")
    monkeypatch.setenv("ASSET_CATALOG_PATH", str(path))
    asset_catalog.get_asset_catalog.cache_clear()
    assert classify_symbol("S10N5") == "Letra"