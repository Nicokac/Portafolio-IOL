import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

import ui.header as hdr


@pytest.mark.parametrize("rates", [None, {}])
def test_render_fx_summary_in_header_no_data(monkeypatch, rates):
    columns_mock = MagicMock()
    markdown_mock = MagicMock()
    monkeypatch.setattr(hdr.st, "columns", columns_mock)
    monkeypatch.setattr(hdr.st, "markdown", markdown_mock)

    hdr.render_fx_summary_in_header(rates)

    columns_mock.assert_not_called()
    markdown_mock.assert_not_called()


def test_render_fx_summary_in_header_partial_dict(monkeypatch):
    class DummyCol:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    cols = [DummyCol() for _ in range(6)]
    columns_mock = MagicMock(return_value=cols)
    markdown_mock = MagicMock()
    pal = SimpleNamespace(highlight_bg="bg", highlight_text="text")

    monkeypatch.setattr(hdr.st, "columns", columns_mock)
    monkeypatch.setattr(hdr.st, "markdown", markdown_mock)
    monkeypatch.setattr(hdr, "get_active_palette", lambda: pal)

    rates = {"oficial": 1234.56, "mep": 789}
    hdr.render_fx_summary_in_header(rates)

    columns_mock.assert_called_once_with(6)
    assert markdown_mock.call_count == 6

    html_official = markdown_mock.call_args_list[0][0][0]
    assert "$ 1.234,56" in html_official
    assert pal.highlight_bg in html_official
    assert pal.highlight_text in html_official

    html_mep = markdown_mock.call_args_list[1][0][0]
    assert "$ 789,00" in html_mep

    html_ccl = markdown_mock.call_args_list[2][0][0]
    assert "–" in html_ccl
