import pytest

from effective_memory.embeddings import CachedEmbedder, HashingEmbedder, get_embedder


def test_get_embedder_default_is_hashing_wrapped_in_a_cache():
    e = get_embedder()
    assert isinstance(e, CachedEmbedder)
    assert isinstance(e._embedder, HashingEmbedder)


def test_get_embedder_hashing_explicit():
    e = get_embedder("hashing")
    assert isinstance(e._embedder, HashingEmbedder)


def test_get_embedder_cache_size_zero_disables_wrapping():
    e = get_embedder("hashing", cache_size=0)
    assert isinstance(e, HashingEmbedder)


def test_get_embedder_unknown_raises():
    with pytest.raises(ValueError):
        get_embedder("not-a-real-backend")


def test_cached_embedder_returns_identical_vector_without_recomputing():
    calls = []

    class CountingEmbedder:
        dims = 4

        def embed(self, text):
            calls.append(text)
            return [1.0, 0.0, 0.0, 0.0]

    cache = CachedEmbedder(CountingEmbedder())
    cache.embed("hello")
    cache.embed("hello")
    cache.embed("hello")
    assert calls == ["hello"]  # only computed once


def test_cached_embedder_evicts_least_recently_used():
    calls = []

    class CountingEmbedder:
        dims = 4

        def embed(self, text):
            calls.append(text)
            return [1.0, 0.0, 0.0, 0.0]

    cache = CachedEmbedder(CountingEmbedder(), maxsize=2)
    cache.embed("a")
    cache.embed("b")
    cache.embed("a")  # refresh "a" so "b" becomes least-recently-used
    cache.embed("c")  # evicts "b", not "a"
    calls.clear()

    cache.embed("a")
    cache.embed("b")
    assert calls == ["b"]  # "a" was still cached, "b" had to be recomputed


def test_voyage_embedder_requires_extra_or_reports_missing_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    try:
        embedder = get_embedder("voyage")
    except ImportError:
        pytest.skip("voyage extra not installed")
    with pytest.raises(Exception):
        embedder.embed("hello")
