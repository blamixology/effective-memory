# effective-memory

A local-first memory engine that is, at the same time:

- **agent memory** — hybrid semantic + keyword `recall()` and
  `build_context()` retrieve the most relevant, least-forgotten memories to
  inject into an LLM prompt under a token budget.
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

See [CHANGELOG.md](CHANGELOG.md) for what changed between versions.

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

    # edit or remove a memory
    mem.update(a, content="Alice works at Acme as a backend engineer")
    mem.delete(b)
```

## CLI

```bash
emem add "The user prefers dark mode" --tags preference
emem update 1 --content "The user prefers dark mode everywhere" --importance 1.5
emem delete 1
emem recall "what theme does the user like"
emem context "summarize what I know about the user" --budget 300
emem review --threshold 0.3
emem compact --threshold 0.15
emem list --status active --tag preference
emem tags
emem export --out backup.json
emem import backup.json
emem reindex
emem stats
```

All commands take `--db path/to/store.db` (defaults to `effective_memory.db`
in the current directory).

## Web UI + REST API

```bash
emem serve --db mem.db --port 8000
```

Open `http://127.0.0.1:8000` for a small browser UI, backed by a JSON API
under `/api/*`:

| Method | Path                    | Purpose                              |
|--------|-------------------------|---------------------------------------|
| POST   | `/api/memories`         | add a memory                          |
| GET    | `/api/memories?status=active&tag=...&limit=50&offset=0` | browse memories, paginated |
| GET    | `/api/memories/{id}`    | fetch one memory                      |
| PATCH  | `/api/memories/{id}`    | edit a memory (re-embeds if `content` changes) |
| DELETE | `/api/memories/{id}`    | delete a memory                       |
| GET    | `/api/memories/{id}/links` | list its links                     |
| POST   | `/api/links`            | link two memories                     |
| GET    | `/api/recall?q=...&k=5` | semantic recall                       |
| GET    | `/api/context?q=...&budget=500` | build an LLM-ready context block |
| GET    | `/api/review?threshold=0.3` | spaced-repetition review queue    |
| POST   | `/api/compact`          | cluster + summarize decayed memories (`summarizer: "none"\|"claude"`) |
| GET    | `/api/stats`            | store statistics                      |
| GET    | `/api/tags`             | distinct tags with counts, most-used first |
| GET    | `/api/export`           | dump all memories + links as JSON     |
| POST   | `/api/import?reset=false` | load a JSON dump, re-embedding every memory |
| POST   | `/api/reindex`          | re-embed every memory with the current embedder |
| GET    | `/api/config`           | active embedder, dims, vector-index and auth state |
| GET    | `/healthz`              | unauthenticated liveness/readiness probe |

The UI has seven tabs:

- **Browse** (the default tab) — paginated list of every memory,
  filterable by status (`active`/`compacted`/`all`) and tag (autocompleted
  from `/api/tags`). A compacted summary shows which original memories it
  was built from.
- **Add** / **Recall** / **Context** / **Review** / **Compact** — as before.
- **Manage** — load a memory by ID to edit or delete it, and see/add its
  outgoing links (the knowledge-graph side of the tool, otherwise only
  reachable via the API).
- **Stats** — counts, a **Configuration** panel (DB path, embedder + model,
  embedding dimensions, vector-index state, auth state), and a
  **Backup & embedder migration** section: export/download a JSON backup,
  import one (optionally wiping existing data first), or reindex everything
  after switching embedders.

Every card everywhere (Browse, Recall, Review) has a delete (✕) button.

The API can also be run directly with `uvicorn effective_memory.api:app`,
configured entirely via env vars (`EFFECTIVE_MEMORY_DB`,
`EFFECTIVE_MEMORY_EMBEDDER`, `EFFECTIVE_MEMORY_EMBEDDING_MODEL`,
`EFFECTIVE_MEMORY_API_KEY` — see below).

### Auth

Every `/api/*` request requires an `X-API-Key` header matching
`EFFECTIVE_MEMORY_API_KEY`. `emem serve` auto-generates and prints a key on
startup only if the env var was never set at all; an *explicitly* empty
value (as `docker-compose.yml` passes by default -- see Docker below) is
treated as "auth intentionally disabled" and left alone, not overridden:

```bash
emem serve --db mem.db                          # nothing set -> prints a generated key
emem serve --db mem.db --api-key mykey          # pin your own key
emem serve --db mem.db --no-auth                # disable auth (local/dev only)
EFFECTIVE_MEMORY_API_KEY= emem serve --db mem.db  # explicit empty -> disabled too
```

The web UI prompts for the key on first use and remembers it in
`localStorage`. `/`, its static assets, and `/healthz` stay unauthenticated
so the page can load and prompt, and orchestrators can probe liveness
without a key; everything under `/api/*` is protected.

### Input limits

`/api/*` bounds the obvious abuse/mistake vectors on a service that's now
network-reachable rather than always local-only: memory `content` is capped
at 20,000 characters, serialized `metadata` at 10,000 bytes,
`k`/`limit`/`offset`/`budget` have sane min/max ranges, and `compact()`'s
thresholds are clamped to their valid `[0, 1]` (or `[-1, 1]` for cosine
similarity) ranges. Out-of-range values return `422` rather than being
silently clamped or allowed to run unbounded queries.

### Rate limiting

`/api/*` also enforces a per-API-key (or per-client-IP when auth is
disabled) token-bucket rate limit -- this now fronts paid embedding and
summarization providers, so an over-trusted or leaked key shouldn't be able
to run up an unbounded bill or hammer the process:

```bash
emem serve --db mem.db --rate-limit 120 --rate-limit-window 60  # the defaults
emem serve --db mem.db --rate-limit 0                            # disabled
```

Exceeding the limit returns `429`. It's in-process only (not distributed or
persisted across restarts), which matches the single-process `emem serve`
deployment model this project supports.

## Real embedding-model backends

By default the store uses a dependency-free hashed-trigram embedder — good
enough to try things out, but a real embedding model gives much better
recall. Swap one in with `--embedder` (CLI) or the matching env vars (API):

```bash
pip install -e ".[dev,voyage]"                 # or [openai] / [sentence-transformers]

emem --embedder voyage add "..."               # needs VOYAGE_API_KEY
emem --embedder openai recall "..."            # needs OPENAI_API_KEY
emem --embedder sentence-transformers stats    # fully local, no API key
```

| `--embedder`             | Backend                          | Needs                          |
|--------------------------|-----------------------------------|---------------------------------|
| `hashing` (default)      | dependency-free hashed trigrams   | nothing                         |
| `voyage`                 | Voyage AI (Anthropic's recommended embeddings provider — Anthropic has no first-party embeddings endpoint) | `voyage` extra, `VOYAGE_API_KEY` |
| `openai`                 | OpenAI embeddings                 | `openai` extra, `OPENAI_API_KEY` |
| `sentence-transformers`  | local model, no network at inference time | `sentence-transformers` extra |

Use `--embedding-model` to pick a specific model name for the chosen
backend. Vectors from different embedders aren't comparable, so switching
`--embedder` on an existing store silently breaks recall for anything added
under the old one -- **run `emem reindex` (or `POST /api/reindex`) right
after switching** to re-embed every memory with the new one.

For `emem serve`, set `EFFECTIVE_MEMORY_EMBEDDER` and
`EFFECTIVE_MEMORY_EMBEDDING_MODEL` instead of the CLI flags.

## Real LLM summarization for compaction

By default `compact()` just concatenates a cluster's memories with `" | "`
— it doesn't actually compress anything. Pass `--summarizer claude` (CLI) or
`"summarizer": "claude"` (API) to have Claude merge each cluster into one
shorter memory that keeps every distinct fact:

```bash
pip install -e ".[dev,llm]"                     # needs ANTHROPIC_API_KEY
emem compact --summarizer claude
emem compact --summarizer claude --summarizer-model claude-opus-5
```

From Python, pass any `list[str] -> str` callable:

```python
from effective_memory.summarizer import claude_summarizer
mem.compact(summarizer=claude_summarizer())
```

## Scaling recall with a vector index

`recall()` normally scores every active memory in Python — fine for small
stores, but it becomes the bottleneck past a few thousand memories. Pass
`--vector-index` (CLI) or `EFFECTIVE_MEMORY_VECTOR_INDEX=1` (API) to back it
with a [sqlite-vec](https://github.com/asg017/sqlite-vec) ANN table instead:

```bash
pip install -e ".[dev,vector-index]"
emem --vector-index recall "..."
```

`recall()` then fetches a candidate pool (`--vector-index-overfetch`,
default 50) by raw cosine similarity from the index, and only re-ranks that
pool by the decay-weighted score in Python — so it stays fast without
rescanning the whole store, at the cost of only ever considering the top
`overfetch` candidates by similarity. Enabling it against a store that
already has data triggers a one-time backfill on open. The embedding
dimension is fixed at index-creation time, so this inherits the existing
"pick one embedder per `--db`" rule.

## Hybrid recall

`recall()` blends two signals: decay-weighted semantic similarity (the
primary driver) plus a small bonus when the memory's content contains the
query's own words verbatim. This matters most with the default hashing
embedder, whose similarity signal is crude enough that an exact name, ID,
or term you typed can otherwise rank below something merely "semantically
close." With `--vector-index` enabled, an exact match that the embedder
scores as dissimilar could fall outside the ANN candidate pool at very
large scale; a small, `LIMIT`-bounded keyword lookup supplements the pool
to catch the common case, though it isn't exhaustive at scale.

## Backup, restore, and embedder migration

`export_data()`/`import_data()` (Python), `emem export`/`emem import`
(CLI), and `/api/export`/`/api/import` (API, plus a button in the web UI's
Stats tab) dump and restore everything as plain JSON -- memories, tags,
metadata, and links, but not embeddings, since those are re-derived from
content on import using whatever embedder is currently configured. That
also makes import the supported way to move a store to a new embedder
wholesale (export, switch `--embedder`, import into a fresh `--db`), and
`reindex()` the way to do it in place on the same store.

```bash
emem export --out backup.json
emem import backup.json              # appended with fresh IDs
emem import backup.json --reset      # replaces existing data entirely
emem reindex                         # re-embed in place after switching --embedder
```

## Docker

```bash
docker build -t effective-memory .
docker run -p 8000:8000 -v emem-data:/data effective-memory
# or:
docker compose up
```

The container persists its SQLite store to the `/data` volume. By default
`docker-compose.yml` passes `EFFECTIVE_MEMORY_API_KEY` through as an
explicitly empty value when unset on the host, which this image treats as
"auth disabled" (see Auth above) -- set a real key in your shell or a
`.env` file before exposing this beyond localhost. Pass `VOYAGE_API_KEY` /
`OPENAI_API_KEY` and `EFFECTIVE_MEMORY_EMBEDDER` as env vars to use a real
embedding backend in the container. `GET /healthz` backs both the
Dockerfile's `HEALTHCHECK` and the compose file's `healthcheck:` block.

Actually build-and-run verified (not just reviewed): `docker build`,
`docker run`, `docker inspect --format '{{json .State.Health}}'` reporting
`"healthy"`, and a full add/recall/stats/healthz smoke test against the
running container, all passing. CI now runs this same build-run-smoke-test
sequence on every push and PR (see Tests below), so this doesn't stay true
only as of when it was last checked by hand.

## Design

- `embeddings.py` — pluggable `Embedder` protocol; ships a dependency-free
  `HashingEmbedder` (bag of hashed character trigrams) plus `VoyageEmbedder`,
  `OpenAIEmbedder`, and `SentenceTransformerEmbedder`, selected via
  `get_embedder(name, model)`. Swap in any other model by implementing
  `embed(text) -> list[float]`.
- `decay.py` — the forgetting-curve math: `retention(elapsed, importance,
  access_count)`.
- `summarizer.py` — `claude_summarizer()`, an LLM-backed summarizer for
  `compact()` (optional `llm` extra).
- `vector_index.py` — `VectorIndex`, a sqlite-vec ANN wrapper `recall()` uses
  at scale (optional `vector-index` extra).
- `store.py` — SQLite-backed `MemoryStore`: CRUD (`add`/`get`/`update`/
  `delete`), linking, `recall`, `build_context`, `review_due`, `compact`,
  `list_memories`/`count_memories`, `tags`, `export_data`/`import_data`,
  `reindex`, `stats`.
- `ratelimit.py` — `TokenBucketLimiter`, the in-process rate limiter behind
  `/api/*`.
- `cli.py` — thin argparse wrapper over `MemoryStore`, plus `emem serve`.
- `api.py` / `web/index.html` — FastAPI REST API (API-key auth + rate
  limiting + input limits on `/api/*`) and a dependency-free vanilla-JS
  single-page UI on top of it (optional `web` extra).
- `Dockerfile` / `docker-compose.yml` — containerized `emem serve` with a
  persistent volume for the SQLite store.

## Tests

```bash
pip install -e ".[dev,web]"   # httpx2 + fastapi needed for the API integration tests
pytest -q
ruff check src tests
mypy src/effective_memory
```

Coverage includes the store (`test_store.py`), the CLI end-to-end via
`cli.main([...])` (`test_cli.py`), the REST API end-to-end via FastAPI's
`TestClient` (`test_api.py`), and the rate limiter in isolation
(`test_ratelimit.py`) -- not just manual curl/CLI smoke tests.

CI (`.github/workflows/ci.yml`) has three jobs on every push and PR:
- **test** — the suite on Python 3.10-3.12. Tests for optional extras
  (voyage/openai/sentence-transformers/llm/vector-index) skip themselves
  gracefully when that extra isn't installed.
- **lint** — `ruff check` and `mypy`.
- **docker** — builds the real `Dockerfile`, runs the container, polls
  `docker inspect` for a `healthy` `HEALTHCHECK`, then smoke-tests
  `/healthz`, an authenticated add + stats round trip, and that an
  unauthenticated request is rejected.
