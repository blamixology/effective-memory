import pytest

pytest.importorskip("sqlite_vec")

from effective_memory import MemoryStore
from effective_memory.embeddings import HashingEmbedder


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
    with MemoryStore(":memory:", embedder=HashingEmbedder(), clock=clock, vector_index=True) as s:
        yield s


def test_recall_via_vector_index_matches_relevant_memory(store):
    store.add("the user prefers dark mode in the editor")
    store.add("the capital of France is Paris")
    results = store.recall("what UI theme does the user like?", k=2, touch_on_recall=False)
    assert results[0].memory.content.startswith("the user prefers dark mode")


def test_delete_removes_from_vector_index(store):
    mid = store.add("a fact about the ocean")
    store.delete(mid)
    results = store.recall("ocean", k=5, touch_on_recall=False)
    assert all(r.memory.id != mid for r in results)


def test_compact_removes_originals_from_vector_index(store, clock):
    store.add("the user likes coffee in the morning")
    store.add("the user prefers coffee at breakfast")
    clock.advance(120 * 24 * 3600)  # far past the default half-life

    new_ids = store.compact(retention_threshold=0.9, cluster_similarity=0.3)
    assert len(new_ids) == 1

    results = store.recall("coffee", k=10, touch_on_recall=False)
    ids = {r.memory.id for r in results}
    assert ids == set(new_ids)


def test_keyword_supplement_finds_match_the_ann_pool_would_miss(clock):
    class BiasedEmbedder:
        """Makes 'noise' documents look identical to the query, and the
        real target look completely dissimilar -- so with a tiny overfetch,
        pure ANN search would never surface the target. Only the keyword
        supplement (matching the literal query term in its content) can.
        """

        dims = 4

        def embed(self, text):
            if text.startswith("random noise"):
                return [1.0, 0.0, 0.0, 0.0]
            return [0.0, 0.0, 1.0, 0.0]

    with MemoryStore(
        ":memory:", embedder=BiasedEmbedder(), clock=clock, vector_index=True, vector_index_overfetch=1
    ) as store:
        for i in range(3):
            store.add(f"random noise memory number {i}")
        target_id = store.add("the secret token XYZ123 was issued")

        results = store.recall("random noise XYZ123", k=5, touch_on_recall=False)
        assert target_id in {r.memory.id for r in results}


def test_existing_store_backfills_index_when_enabled_later(tmp_path):
    db_path = str(tmp_path / "backfill.db")
    with MemoryStore(db_path) as s:
        s.add("memory added before the index existed")

    with MemoryStore(db_path, vector_index=True) as s:
        results = s.recall("memory added before the index existed", k=1, touch_on_recall=False)
        assert len(results) == 1
