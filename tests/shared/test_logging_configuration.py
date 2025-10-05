import logging

import pytest

from shared import config


@pytest.mark.parametrize("json_format", [False, True])
def test_configure_logging_sets_matplotlib_font_manager_level(json_format):
    root_logger = logging.getLogger()
    original_level = root_logger.level
    original_handlers = root_logger.handlers[:]
    original_matplotlib_level = logging.getLogger("matplotlib.font_manager").level

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    try:
        config.configure_logging(level="INFO", json_format=json_format)
        assert logging.getLogger("matplotlib.font_manager").level == logging.WARNING
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        for handler in original_handlers:
            root_logger.addHandler(handler)

        root_logger.setLevel(original_level)
        logging.getLogger("matplotlib.font_manager").setLevel(original_matplotlib_level)
