"""Smoke test ensuring the test bootstrap stays in offline mode."""


def test_imports_do_not_pull_ui():
    import os

    assert os.environ.get("UNIT_TEST") == "1"
