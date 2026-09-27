import importlib

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("EFFECTIVE_MEMORY_DB", str(tmp_path / "api_test.db"))
    monkeypatch.setenv("EFFECTIVE_MEMORY_API_KEY", "")
    monkeypatch.delenv("EFFECTIVE_MEMORY_VECTOR_INDEX", raising=False)
    import effective_memory.api as api

    importlib.reload(api)
    with TestClient(api.app) as c:
        yield c
    api.store.close()


def test_add_get_update_delete_roundtrip(client):
    res = client.post("/api/memories", json={"content": "the user likes tea", "tags": ["preference"]})
    assert res.status_code == 200
    memory_id = res.json()["id"]

    res = client.get(f"/api/memories/{memory_id}")
    assert res.status_code == 200
    assert res.json()["content"] == "the user likes tea"

    res = client.patch(f"/api/memories/{memory_id}", json={"content": "the user likes green tea"})
    assert res.status_code == 200
    assert res.json()["content"] == "the user likes green tea"

    res = client.delete(f"/api/memories/{memory_id}")
    assert res.status_code == 200

    assert client.get(f"/api/memories/{memory_id}").status_code == 404
    assert client.patch(f"/api/memories/{memory_id}", json={"content": "x"}).status_code == 404
    assert client.delete(f"/api/memories/{memory_id}").status_code == 404


def test_recall_and_context(client):
    client.post("/api/memories", json={"content": "the user prefers dark mode in the editor"})
    client.post("/api/memories", json={"content": "the capital of France is Paris"})

    res = client.get("/api/recall", params={"q": "what UI theme does the user like", "k": 2})
    assert res.status_code == 200
    results = res.json()
    assert results[0]["memory"]["content"].startswith("the user prefers dark mode")

    res = client.get("/api/context", params={"q": "user preferences", "budget": 50})
    assert res.status_code == 200
    assert "context" in res.json()


def test_list_memories_pagination_and_filter(client):
    for i in range(5):
        client.post("/api/memories", json={"content": f"memory {i}", "tags": ["batch"] if i % 2 == 0 else []})

    res = client.get("/api/memories", params={"limit": 2, "offset": 0})
    body = res.json()
    assert body["total"] == 5
    assert len(body["items"]) == 2

    res = client.get("/api/memories", params={"tag": "batch"})
    body = res.json()
    assert body["total"] == 3


def test_links(client):
    a = client.post("/api/memories", json={"content": "Alice"}).json()["id"]
    b = client.post("/api/memories", json={"content": "Bob"}).json()["id"]

    res = client.post("/api/links", json={"src_id": a, "dst_id": b, "relation": "knows"})
    assert res.status_code == 200

    res = client.get(f"/api/memories/{a}/links")
    links = res.json()
    assert len(links) == 1
    assert links[0]["relation"] == "knows"
    assert links[0]["memory"]["id"] == b


def test_compact_rejects_unknown_summarizer(client):
    res = client.post("/api/compact", json={"summarizer": "not-a-real-summarizer"})
    assert res.status_code == 400


def test_review_and_stats(client):
    client.post("/api/memories", json={"content": "a fresh memory"})

    res = client.get("/api/review", params={"threshold": 0.99})
    assert res.status_code == 200

    res = client.get("/api/stats")
    body = res.json()
    assert body["total"] == 1


def test_tags_endpoint(client):
    client.post("/api/memories", json={"content": "a", "tags": ["work", "urgent"]})
    client.post("/api/memories", json={"content": "b", "tags": ["work"]})
    res = client.get("/api/tags")
    assert res.status_code == 200
    assert res.json() == [{"tag": "work", "count": 2}, {"tag": "urgent", "count": 1}]


def test_export_import_roundtrip(client):
    client.post("/api/memories", json={"content": "the user likes tea", "tags": ["preference"]})
    dump = client.get("/api/export").json()
    assert len(dump["memories"]) == 1

    res = client.post("/api/import", json=dump)
    assert res.status_code == 200
    assert res.json() == {"memories": 1, "links": 0}

    res = client.get("/api/memories", params={"status": "all"})
    assert res.json()["total"] == 2  # original + imported copy


def test_reindex_endpoint(client):
    client.post("/api/memories", json={"content": "a fact about the ocean"})
    res = client.post("/api/reindex")
    assert res.status_code == 200
    assert res.json() == {"reindexed": 1}


def test_config_reports_current_setup(client):
    res = client.get("/api/config")
    assert res.status_code == 200
    body = res.json()
    assert body["embedder"] == "hashing"
    assert body["auth_enabled"] is False


def test_auth_enforced_when_api_key_set(monkeypatch, tmp_path):
    monkeypatch.setenv("EFFECTIVE_MEMORY_DB", str(tmp_path / "auth_test.db"))
    monkeypatch.setenv("EFFECTIVE_MEMORY_API_KEY", "topsecret")
    import effective_memory.api as api

    importlib.reload(api)
    try:
        with TestClient(api.app) as c:
            assert c.get("/api/stats").status_code == 401
            assert c.get("/api/stats", headers={"X-API-Key": "wrong"}).status_code == 401
            assert c.get("/api/stats", headers={"X-API-Key": "topsecret"}).status_code == 200
            assert c.get("/").status_code == 200  # the page itself stays public
    finally:
        api.store.close()
