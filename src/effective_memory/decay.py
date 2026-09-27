"""Ebbinghaus-style forgetting curve.

The same mechanism that decides what a human would still remember also
decides what an agent should still keep in context, and what a personal
knowledge base should compact away: retention decays with elapsed time,
and each access ("rehearsal") strengthens the memory and slows the decay,
exactly like spaced repetition.
"""

from __future__ import annotations

import math

DEFAULT_HALF_LIFE_SECONDS = 7 * 24 * 3600  # one week, at importance=1.0, 0 accesses


def strength(importance: float, access_count: int, half_life: float = DEFAULT_HALF_LIFE_SECONDS) -> float:
    """Effective half-life for a memory, in seconds.

    Higher importance and more accesses both extend how long a memory
    stays retrievable before it decays toward the forgetting threshold.
    """
    importance = max(importance, 0.01)
    return half_life * importance * (1.0 + math.log1p(access_count))


def retention(elapsed_seconds: float, importance: float, access_count: int,
              half_life: float = DEFAULT_HALF_LIFE_SECONDS) -> float:
    """Fraction of a memory 'remembered' right now, in [0, 1]."""
    tau = strength(importance, access_count, half_life)
    return math.exp(-max(elapsed_seconds, 0.0) / tau)


def recall_boost(access_count: int) -> float:
    """How much a single access strengthens a memory (diminishing returns)."""
    return math.log1p(access_count + 1) - math.log1p(access_count)
