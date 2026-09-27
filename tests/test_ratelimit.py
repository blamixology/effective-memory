from effective_memory.ratelimit import TokenBucketLimiter


def test_allows_up_to_capacity_then_blocks():
    limiter = TokenBucketLimiter(rate_per_second=0.0, capacity=3)
    assert [limiter.allow("a") for _ in range(4)] == [True, True, True, False]


def test_buckets_are_independent_per_key():
    limiter = TokenBucketLimiter(rate_per_second=0.0, capacity=1)
    assert limiter.allow("a") is True
    assert limiter.allow("b") is True
    assert limiter.allow("a") is False
    assert limiter.allow("b") is False


def test_refills_over_time(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr("time.monotonic", lambda: t[0])

    limiter = TokenBucketLimiter(rate_per_second=1.0, capacity=1)
    assert limiter.allow("a") is True
    assert limiter.allow("a") is False

    t[0] += 1.0  # one full token's worth of time passes
    assert limiter.allow("a") is True
