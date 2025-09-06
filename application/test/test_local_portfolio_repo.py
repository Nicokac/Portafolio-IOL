from infrastructure.local_portfolio import LocalPortfolioRepository


def test_local_portfolio_persistence_and_edit(tmp_path):
    repo = LocalPortfolioRepository(tmp_path / "portfolio.json")

    # Start empty
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