import math

import pytest

from effective_memory import MemoryStore


class FakeClock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def store(clock):
    with MemoryStore(":memory:", clock=clock) as s:
        yield s


def test_add_and_get(store):
    mid = store.add("the sky is blue", tags=["fact"])
    mem = store.get(mid)
    assert mem.content == "the sky is blue"
    assert mem.tags == ["fact"]
    assert mem.access_count == 0


def test_recall_ranks_relevant_memory_first(store):
    store.add("the user prefers dark mode in the editor")
    store.add("the capital of France is Paris")
    results = store.recall("what UI theme does the user like?", k=2, touch_on_recall=False)
    assert results[0].memory.content.startswith("the user prefers dark mode")


def test_recall_strengthens_memory(store, clock):
    mid = store.add("remember this important fact")
    before = store.get(mid)
    assert before.access_count == 0

    store.recall("important fact", k=1)
    after = store.get(mid)
    assert after.access_count == 1
    assert after.last_accessed == clock.t


def test_retention_decays_over_time(store, clock):
    store.add("a fact that will fade")
    r1 = store.recall("a fact that will fade", k=1, touch_on_recall=False)[0]

    clock.advance(60 * 24 * 3600)  # 60 days, far past the default half-life
    r2 = store.recall("a fact that will fade", k=1, touch_on_recall=False)[0]

    assert r2.retention < r1.retention


def test_review_due_surfaces_decayed_memories(store, clock):
    fresh_id = store.add("fresh memory")
    stale_id = store.add("stale memory")
    clock.advance(90 * 24 * 3600)

    due = store.review_due(threshold=0.5)
    due_ids = {r.memory.id for r in due}
    assert stale_id in due_ids
    # both have decayed since neither has been touched, but this at least
    # confirms the stale one is present and ranked by lowest retention first
    assert due[0].retention <= due[-1].retention


def test_compact_merges_similar_decayed_memories(store, clock):
    store.add("the user likes coffee in the morning")
    store.add("the user prefers coffee at breakfast")
    store.add("unrelated fact about astronomy")
    clock.advance(120 * 24 * 3600)

    new_ids = store.compact(retention_threshold=0.9, cluster_similarity=0.3)
    assert len(new_ids) >= 1

    stats = store.stats()
    assert stats["by_status"].get("compacted", 0) >= 2
    assert stats["by_status"].get("active", 0) >= 1  # the summary is active


def test_link_and_links_for(store):
    a = store.add("Alice works at Acme")
    b = store.add("Acme is a software company")
    store.link(a, b, relation="works_at")

    links = store.links_for(a)
    assert len(links) == 1
    relation, mem = links[0]
    assert relation == "works_at"
    assert mem.id == b


def test_build_context_respects_token_budget(store):
    for i in range(20):
        store.add(f"memory number {i} about the project roadmap")

    context = store.build_context("project roadmap", token_budget=20)
    assert len(context) <= 20 * 4


def test_stats_counts(store):
    store.add("one")
    store.add("two")
    stats = store.stats()
    assert stats["total"] == 2
    assert stats["by_status"]["active"] == 2
    assert stats["links"] == 0
