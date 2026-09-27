import pytest

from effective_memory.embeddings import HashingEmbedder, get_embedder


def test_get_embedder_default_is_hashing():
    e = get_embedder()
    assert isinstance(e, HashingEmbedder)


def test_get_embedder_hashing_explicit():
    e = get_embedder("hashing")
    assert isinstance(e, HashingEmbedder)


def test_get_embedder_unknown_raises():
    with pytest.raises(ValueError):
        get_embedder("not-a-real-backend")


def test_voyage_embedder_requires_extra_or_reports_missing_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    try:
        embedder = get_embedder("voyage")
    except ImportError:
        pytest.skip("voyage extra not installed")
    with pytest.raises(Exception):
        embedder.embed("hello")
