"""A simple in-process token-bucket rate limiter for the REST API.

Not distributed or persisted -- fine for a single-process `emem serve`
deployment, which is the only supported one. Restarting the process resets
every bucket.
"""

from __future__ import annotations

import threading
import time


class TokenBucketLimiter:
    def __init__(self, rate_per_second: float, capacity: float):
        self.rate = rate_per_second
        self.capacity = capacity
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            return True
