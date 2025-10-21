from tests.fixtures.clock import FakeClock


def test_fake_clock_progression(fake_clock):
    assert fake_clock() == 0.0
    fake_clock.advance(2.5)
    assert fake_clock() == 2.5


def test_fake_clock_start_override():
    clock = FakeClock(start=42)
    assert clock() == 42.0
    clock.advance(1)
    assert clock() == 43.0
