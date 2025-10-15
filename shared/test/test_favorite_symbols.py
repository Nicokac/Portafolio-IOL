import json
import logging

from shared.favorite_symbols import FavoriteSymbols, JSONFavoriteStorage


def test_initializes_state_when_missing():
    state: dict[str, object] = {}
    fav = FavoriteSymbols(state)
    assert state[FavoriteSymbols.STATE_KEY] == set()
    assert fav.list() == []


def test_add_remove_and_toggle_symbols():
    state: dict[str, object] = {}
    fav = FavoriteSymbols(state)
    fav.add("ggal")
    fav.add(" GGAL ")
    fav.add("APBR")
    favorites = fav.list()
    assert set(favorites) == {"GGAL", "APBR"}
    assert len(favorites) == 2

    fav.remove("GGAL")
    assert fav.list() == ["APBR"]

    fav.toggle("apbr")
    assert fav.list() == []
    fav.toggle("PAMP")
    assert fav.list() == ["PAMP"]


def test_replace_and_helpers():
    state: dict[str, object] = {}
    fav = FavoriteSymbols(state)
    replaced = fav.replace(["apbr", "GGAL", "ggal"])
    assert set(replaced) == {"APBR", "GGAL"}
    assert len(replaced) == 2
    assert fav.format_symbol("apbr") == "⭐ APBR"
    assert fav.format_symbol("pamp") == "PAMP"

    options = ["PAMP", "APBR", "GGAL"]
    sorted_opts = fav.sort_options(options)
    assert sorted_opts[:2] == ["APBR", "GGAL"]
    assert fav.default_index(sorted_opts) == 0

    fav.toggle("pamp")
    assert fav.default_index(sorted_opts) == 0


def test_is_favorite_handles_none():
    fav = FavoriteSymbols({})
    assert not fav.is_favorite(None)
    assert fav.toggle(None) is False


def test_is_favorite_initializes_missing_state(caplog):
    state: dict[str, object] = {}
    fav = FavoriteSymbols(state)
    state.pop(FavoriteSymbols.STATE_KEY)

    caplog.set_level(logging.WARNING, logger="shared.favorite_symbols")

    assert fav.is_favorite("GGAL") is False
    assert FavoriteSymbols.STATE_KEY in state
    assert state[FavoriteSymbols.STATE_KEY] == set()
    assert "favorite_symbols no estaba inicializado" in caplog.text


def test_loads_from_storage_when_available(tmp_path):
    storage_path = tmp_path / "favorites.json"
    storage_path.write_text(json.dumps(["ggal", "PAMP", "GGAL"]))
    storage = JSONFavoriteStorage(storage_path)

    state: dict[str, object] = {}
    fav = FavoriteSymbols(state, storage=storage)

    favorites = fav.list()
    assert set(favorites) == {"GGAL", "PAMP"}
    assert len(favorites) == 2
    assert state[FavoriteSymbols.LOADED_FLAG_KEY] is True
    assert fav.last_error is None


def test_persists_changes_to_storage(tmp_path):
    storage = JSONFavoriteStorage(tmp_path / "favorites.json")
    fav = FavoriteSymbols({}, storage=storage)

    fav.add("ggal")
    fav.add("pamp")

    data = json.loads((storage.path).read_text())
    assert set(data) == {"GGAL", "PAMP"}
    assert len(data) == 2

    fav.remove("GGAL")
    data = json.loads(storage.path.read_text())
    assert data == ["PAMP"]


def test_storage_errors_are_exposed(tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("no-dir")
    storage = JSONFavoriteStorage(blocker / "favorites.json")

    fav = FavoriteSymbols({}, storage=storage)
    fav.add("GGAL")

    assert fav.last_error
