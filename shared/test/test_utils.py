import importlib
import sys
import importlib
import sys
import types
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
