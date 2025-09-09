import pytest
from dataclasses import FrozenInstanceError

from domain.models import Controls


def test_controls_defaults():
    c = Controls()
    assert c.refresh_secs == 30
    assert c.hide_cash is True
    assert c.show_usd is False
    assert c.order_by == "valor_actual"
    assert c.desc is True
    assert c.top_n == 20
    assert c.selected_syms == []
    assert c.selected_types == []
    assert c.symbol_query == ""


def test_controls_immutability():
    c = Controls()
    with pytest.raises(FrozenInstanceError):
        c.refresh_secs = 60  # type: ignore[misc]


def test_controls_default_lists_are_independent():
    a = Controls()
    b = Controls()
    assert a.selected_syms is not b.selected_syms
    assert a.selected_types is not b.selected_types
