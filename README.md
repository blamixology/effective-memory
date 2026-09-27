# effective-memory

A local-first memory engine that is, at the same time:

- **agent memory** — semantic `recall()` and `build_context()` retrieve the
  most relevant, least-forgotten memories to inject into an LLM prompt
  under a token budget.
- **a personal knowledge base** — `add()` stores notes with tags and
  metadata, `link()` connects related memories into a small knowledge
  graph, and `review_due()` surfaces things worth revisiting.
- **efficient memory management** — `compact()` finds memories that have
  decayed past a forgetting threshold, clusters the semantically similar
  ones, and replaces each cluster with a single summary, bounding storage
  and context size as the store grows.

The unifying idea: model what to keep, surface, or compress using the same
mechanism a human brain uses — an [Ebbinghaus forgetting
curve](https://en.wikipedia.org/wiki/Forgetting_curve). Retention decays
with elapsed time; every access ("rehearsal") strengthens a memory and
slows its decay, exactly like spaced repetition. That single decay
function decides ranking in `recall()`, what shows up in `review_due()`,
and what's eligible for `compact()`.

No external services required: it ships with a dependency-free hashed
n-gram embedder so it works fully offline, and everything is stored in one
SQLite file. Both the embedder and the summarizer used during compaction
are pluggable, so you can swap in a real embedding model or an LLM-backed
summarizer.

## Install

```bash
pip install -e ".[dev]"        # core + tests
pip install -e ".[dev,web]"    # + REST API and web UI
```

## Python API

```python
from effective_memory import MemoryStore

with MemoryStore("mem.db") as mem:
    mem.add("The user prefers dark mode in the editor", tags=["preference"])
    mem.add("The project deadline is end of Q3", tags=["project"], importance=1.5)

    # agent memory: build a context block to inject into a prompt
    context = mem.build_context("what does the user prefer?", token_budget=200)

    # personal knowledge base: link related notes
    a = mem.add("Alice works at Acme")
    b = mem.add("Acme is a software company")
    mem.link(a, b, relation="works_at")

    # spaced-repetition style review queue
    for r in mem.review_due(threshold=0.3):
        print(r.memory.content, r.retention)

    # efficient memory management: merge decayed, similar memories
    mem.compact(retention_threshold=0.15)
```

## CLI

```bash
emem add "The user prefers dark mode" --tags preference
emem recall "what theme does the user like"
emem context "summarize what I know about the user" --budget 300
emem review --threshold 0.3
emem compact --threshold 0.15
emem stats
```

All commands take `--db path/to/store.db` (defaults to `effective_memory.db`
in the current directory).

## Web UI + REST API

```bash
emem serve --db mem.db --port 8000
```

Open `http://127.0.0.1:8000` for a small browser UI (add notes, recall,
build a context block, review due memories, compact, and see stats), backed
by a JSON API under `/api/*`:

| Method | Path                    | Purpose                              |
|--------|-------------------------|---------------------------------------|
| POST   | `/api/memories`         | add a memory                          |
| GET    | `/api/memories/{id}`    | fetch one memory                      |
| GET    | `/api/memories/{id}/links` | list its links                     |
| POST   | `/api/links`            | link two memories                     |
| GET    | `/api/recall?q=...&k=5` | semantic recall                       |
| GET    | `/api/context?q=...&budget=500` | build an LLM-ready context block |
| GET    | `/api/review?threshold=0.3` | spaced-repetition review queue    |
| POST   | `/api/compact`          | cluster + summarize decayed memories  |
| GET    | `/api/stats`            | store statistics                      |

The API can also be run directly with `uvicorn effective_memory.api:app`,
configuring the database via the `EFFECTIVE_MEMORY_DB` env var.

## Design

- `embeddings.py` — pluggable `Embedder` protocol; ships a dependency-free
  `HashingEmbedder` (bag of hashed character trigrams). Swap in a real
  model by implementing `embed(text) -> list[float]`.
- `decay.py` — the forgetting-curve math: `retention(elapsed, importance,
  access_count)`.
- `store.py` — SQLite-backed `MemoryStore`: CRUD, linking, `recall`,
  `build_context`, `review_due`, `compact`, `stats`.
- `cli.py` — thin argparse wrapper over `MemoryStore`, plus `emem serve`.
- `api.py` / `web/index.html` — FastAPI REST API and a dependency-free
  vanilla-JS single-page UI on top of it (optional `web` extra).

## Tests

```bash
pytest -q
```
