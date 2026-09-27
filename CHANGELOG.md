# Changelog

## 0.2.0

**Store**
- Full CRUD: `update()`/`delete()` alongside `add()`/`get()`.
- `list_memories()`/`count_memories()` (paginated, filterable by status/tag)
  and `tags()` for browsing.
- `export_data()`/`import_data()` for backup/restore, and `reindex()` to
  re-embed a store in place after switching embedders.
- Hybrid recall: semantic similarity plus a small bonus for verbatim
  keyword overlap, so an exact name/ID/term doesn't get outranked by
  something merely "semantically close."
- Optional sqlite-vec ANN index (`vector_index=True`) for `recall()` at
  scale, with a keyword-supplement query to catch exact matches an ANN
  candidate pool alone could miss.
- `claude_summarizer()`: a real LLM-backed summarizer for `compact()`,
  replacing the placeholder extractive join.
- Real embedding backends: `VoyageEmbedder`, `OpenAIEmbedder`,
  `SentenceTransformerEmbedder` alongside the offline `HashingEmbedder`,
  selected via `get_embedder(name, model)`.

**API & web UI**
- FastAPI REST API and a vanilla-JS single-page UI (`emem serve`) covering
  every store operation: Browse (the default tab), Add, Recall, Context,
  Review, Manage (edit/delete/links), Compact, and Stats (config +
  backup/reindex).
- API-key auth on `/api/*`, with `/`, static assets, and `GET /healthz`
  left open so the UI can load/prompt and orchestrators can probe liveness.
- Input limits (content length, `k`/`limit`/`offset`/`budget` ranges,
  `compact()` threshold bounds) return `422` instead of running unbounded.
- Per-key rate limiting on `/api/*` (`EFFECTIVE_MEMORY_RATE_LIMIT`), since
  the API now fronts paid embedding/summarization providers and is a real
  network service, not always a trusted local process.
- Docker packaging (`Dockerfile`, `docker-compose.yml`) with a `HEALTHCHECK`
  backed by `/healthz`; build+run verified against a real Docker daemon,
  which caught and fixed a bug where an explicitly empty
  `EFFECTIVE_MEMORY_API_KEY` was treated the same as unset.

**Project**
- Test suite covering the store, CLI (`cli.main([...])`), and API
  (`TestClient`) rather than manual smoke tests -- store/CLI/API/vector-index/
  embeddings/decay/summarizer, 67 tests.
- CI (`.github/workflows/ci.yml`) running the suite on Python 3.10-3.12,
  `ruff`/`mypy`, and a Docker build + container smoke test, on every push
  and PR.

## 0.1.0

Initial release: `MemoryStore` unifying agent memory (semantic `recall()`/
`build_context()`), a personal knowledge base (`add()`/`link()`/
`review_due()`), and decay-based `compact()`, backed by a dependency-free
hashed-trigram embedder and SQLite. CLI (`emem`).
