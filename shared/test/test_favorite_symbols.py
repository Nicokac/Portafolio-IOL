from shared.favorite_symbols import FavoriteSymbols


def test_initializes_state_when_missing():
    state: dict[str, object] = {}
    fav = FavoriteSymbols(state)
    assert state[FavoriteSymbols.STATE_KEY] == []
    assert fav.list() == []


def test_add_remove_and_toggle_symbols():
    state: dict[str, object] = {}
    fav = FavoriteSymbols(state)
    fav.add("ggal")
    fav.add(" GGAL ")
    fav.add("APBR")
    assert fav.list() == ["GGAL", "APBR"]

    fav.remove("GGAL")
    assert fav.list() == ["APBR"]

    fav.toggle("apbr")
    assert fav.list() == []
    fav.toggle("PAMP")
    assert fav.list() == ["PAMP"]


def test_replace_and_helpers():
    state: dict[str, object] = {}
    fav = FavoriteSymbols(state)
    fav.replace(["apbr", "GGAL", "ggal"])
    assert fav.list() == ["APBR", "GGAL"]
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
