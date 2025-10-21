from tests.fixtures.time import FakeTime


def test_fake_time_progression(fake_time):
    t0 = fake_time.time()
    fake_time.sleep(3.5)
    assert fake_time.time() == t0 + 3.5


def test_fake_time_start_override():
    clock = FakeTime(start=42)
    assert clock.time() == 42.0
    clock.sleep(1)
    assert clock.time() == 43.0
