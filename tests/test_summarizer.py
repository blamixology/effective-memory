import pytest


def test_claude_summarizer_reports_missing_key_or_extra(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from effective_memory.summarizer import claude_summarizer

    try:
        summarize = claude_summarizer()
    except ImportError:
        pytest.skip("llm extra not installed")

    with pytest.raises(Exception):
        summarize(["memory one", "memory two"])


def test_compact_uses_provided_summarizer():
    from effective_memory import MemoryStore

    clock = {"t": 1_000_000.0}
    with MemoryStore(":memory:", clock=lambda: clock["t"]) as store:
        store.add("the user likes coffee in the morning")
        store.add("the user prefers coffee at breakfast")
        clock["t"] += 120 * 24 * 3600  # far past the default half-life

        calls = []

        def fake_summarizer(contents):
            calls.append(contents)
            return "merged: " + "; ".join(contents)

        new_ids = store.compact(retention_threshold=0.99, cluster_similarity=0.3, summarizer=fake_summarizer)
        assert len(new_ids) == 1
        assert len(calls) == 1
        summary = store.get(new_ids[0])
        assert summary.content.startswith("merged:")
