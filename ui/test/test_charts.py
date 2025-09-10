from ui import charts


def test_rgba_converts_hex_to_rgba():
    assert charts._rgba("#ff0000", 0.5) == "rgba(255,0,0,0.5)"


def test_symbol_color_map_cycles(monkeypatch):
    monkeypatch.setattr(charts, "_SYMBOL_PALETTE", ["red", "green"])
    mapping = charts._symbol_color_map(["a", "b", "c"])
    assert mapping == {"a": "red", "b": "green", "c": "red"}


def test_si_formats_number_with_thousand_separator():
    assert charts._si(1234567) == "1.234.567"
