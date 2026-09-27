from effective_memory.decay import DEFAULT_HALF_LIFE_SECONDS, retention, strength


def test_retention_is_one_at_zero_elapsed():
    assert retention(0, importance=1.0, access_count=0) == 1.0


def test_retention_decreases_with_elapsed_time():
    r1 = retention(1, importance=1.0, access_count=0)
    r2 = retention(DEFAULT_HALF_LIFE_SECONDS, importance=1.0, access_count=0)
    r3 = retention(DEFAULT_HALF_LIFE_SECONDS * 10, importance=1.0, access_count=0)
    assert r1 > r2 > r3


def test_more_accesses_slow_decay():
    elapsed = DEFAULT_HALF_LIFE_SECONDS
    r_unrehearsed = retention(elapsed, importance=1.0, access_count=0)
    r_rehearsed = retention(elapsed, importance=1.0, access_count=5)
    assert r_rehearsed > r_unrehearsed


def test_higher_importance_slows_decay():
    elapsed = DEFAULT_HALF_LIFE_SECONDS
    r_low = retention(elapsed, importance=0.5, access_count=0)
    r_high = retention(elapsed, importance=2.0, access_count=0)
    assert r_high > r_low


def test_strength_grows_with_access_count():
    s0 = strength(1.0, 0)
    s5 = strength(1.0, 5)
    assert s5 > s0
