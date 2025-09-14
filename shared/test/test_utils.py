import importlib
import sys
import importlib
import sys
import types
import logging
import pytest

class _NumpyStub:
    @staticmethod
    def isfinite(x):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return False
        return not (x != x or x in (float('inf'), float('-inf')))

sys.modules.setdefault('numpy', _NumpyStub())

utils = importlib.import_module('shared.utils')


def test_to_float_handles_european_format():
    assert utils._to_float('1.234,56') == pytest.approx(1234.56)


def test_to_float_invalid_returns_none():
    assert utils._to_float('abc') is None


def test_format_money_negative_usd():
    assert utils.format_money(-1234.5, 'USD') == '-US$ 1.234,50'


def test_format_money_none():
    assert utils.format_money(None) == '—'


def test_format_price_negative():
    assert utils.format_price(-1234.5) == '$ -1.234,500'


def test_format_price_none():
    assert utils.format_price(None) == '—'


def test_format_percent_negative():
    assert utils.format_percent(-12.3456) == '-12.35 %'


def test_format_percent_none():
    assert utils.format_percent(None) == '—'


def test_format_number_negative():
    assert utils.format_number(-1234.5) == '-1.234'


def test_format_number_none():
    assert utils.format_number(None) == '—'


def test_to_float_invalid_no_log(caplog):
    caplog.set_level(logging.WARNING, logger='shared.utils')
    assert utils._to_float('abc', log=False) is None
    assert caplog.records == []


def test_as_float_or_none_invalid_no_log():
    assert utils._as_float_or_none('abc', log=False) is None


def test_as_float_or_none_inf():
    import numpy as np
    if not hasattr(np, 'inf'):
        np.inf = float('inf')
    assert utils._as_float_or_none(np.inf) is None
    assert utils._as_float_or_none(-np.inf) is None