from tests.fixtures.common import DummyCtx


def test_dummy_ctx_usage(dummy_ctx_fixture):
    seen = []
    with dummy_ctx_fixture as ctx:
        seen.append(ctx)
    assert seen and seen[0] is dummy_ctx_fixture


def test_dummy_ctx_custom_enter_result():
    marker = object()
    ctx = DummyCtx(enter_result=marker)
    with ctx as value:
        assert value is marker


def test_dummy_ctx_button_default(dummy_ctx_fixture):
    assert dummy_ctx_fixture.button() is False
