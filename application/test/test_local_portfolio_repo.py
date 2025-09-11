import json
import pytest

from infrastructure.local_portfolio import LocalPortfolioRepository


def test_local_portfolio_persistence_and_edit(tmp_path):
    repo = LocalPortfolioRepository(tmp_path / "portfolio.json")

    # Start empty (missing file triggers OSError)
    assert repo.load() == {"activos": []}

    # Add a position and reload
    repo.add({"simbolo": "AL30", "cantidad": 100})
    assert repo.load()["activos"] == [{"simbolo": "AL30", "cantidad": 100}]

    # Update existing position
    repo.update("AL30", {"cantidad": 150})
    assert repo.load()["activos"][0]["cantidad"] == 150

    # Remove position
    repo.remove("AL30")
    assert repo.load() == {"activos": []}


def test_load_returns_empty_on_bad_json(tmp_path):
    p = tmp_path / "portfolio.json"
    p.write_text("{bad", encoding="utf-8")
    repo = LocalPortfolioRepository(p)
    assert repo.load() == {"activos": []}


def test_load_propagates_unexpected(monkeypatch, tmp_path):
    p = tmp_path / "portfolio.json"
    p.write_text("{}", encoding="utf-8")
    repo = LocalPortfolioRepository(p)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("infrastructure.local_portfolio.json.loads", boom)
    with pytest.raises(RuntimeError):
        repo.load()
