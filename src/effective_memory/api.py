"""FastAPI wrapper around MemoryStore, plus a minimal static web UI.

Run with:

    emem serve --db mem.db --port 8000

or directly:

    uvicorn effective_memory.api:app

Configuration (env vars, all optional):
    EFFECTIVE_MEMORY_DB                    path to the SQLite store
    EFFECTIVE_MEMORY_EMBEDDER              hashing | voyage | openai | sentence-transformers
    EFFECTIVE_MEMORY_EMBEDDING_MODEL       model name for the chosen embedder
    EFFECTIVE_MEMORY_EMBEDDING_CACHE_SIZE  LRU cache size for repeated embed() calls
                                            (default 256); "0" disables it
    EFFECTIVE_MEMORY_API_KEY               comma-separated list of keys accepted on the
                                            "X-API-Key" header for every /api/* request
                                            (a single key works the same as before); set
                                            to "" to disable auth (local/dev use only).
                                            Revoke a key by removing it from the list and
                                            restarting -- keys aren't stored anywhere else.
    EFFECTIVE_MEMORY_VECTOR_INDEX          "1" to use a sqlite-vec ANN index for
                                            recall() (requires the 'vector-index' extra)
    EFFECTIVE_MEMORY_VECTOR_INDEX_OVERFETCH  candidate pool size before decay re-ranking
    EFFECTIVE_MEMORY_RATE_LIMIT             requests per EFFECTIVE_MEMORY_RATE_LIMIT_WINDOW
                                             seconds allowed per API key (or per client
                                             IP when auth is disabled); "0" disables it
    EFFECTIVE_MEMORY_RATE_LIMIT_WINDOW      window length in seconds (default 60)
"""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .embeddings import get_embedder
from .ratelimit import TokenBucketLimiter
from .store import Memory, MemoryStore, RecallResult

MAX_CONTENT_LENGTH = 20_000
MAX_METADATA_BYTES = 10_000

DB_PATH = os.environ.get("EFFECTIVE_MEMORY_DB", "effective_memory.db")
EMBEDDER_NAME = os.environ.get("EFFECTIVE_MEMORY_EMBEDDER", "hashing")
EMBEDDING_MODEL = os.environ.get("EFFECTIVE_MEMORY_EMBEDDING_MODEL")
EMBEDDING_CACHE_SIZE = int(os.environ.get("EFFECTIVE_MEMORY_EMBEDDING_CACHE_SIZE", "256"))
# unset or "" disables auth; comma-separated for multiple independently
# revocable keys (e.g. one per agent/integration)
API_KEYS = [k.strip() for k in (os.environ.get("EFFECTIVE_MEMORY_API_KEY") or "").split(",") if k.strip()]
VECTOR_INDEX = os.environ.get("EFFECTIVE_MEMORY_VECTOR_INDEX", "") == "1"
VECTOR_INDEX_OVERFETCH = int(os.environ.get("EFFECTIVE_MEMORY_VECTOR_INDEX_OVERFETCH", "50"))
RATE_LIMIT = int(os.environ.get("EFFECTIVE_MEMORY_RATE_LIMIT", "120"))
RATE_LIMIT_WINDOW = float(os.environ.get("EFFECTIVE_MEMORY_RATE_LIMIT_WINDOW", "60"))
_STATIC_DIR = Path(__file__).parent / "web"

app = FastAPI(title="effective-memory", description="A local-first memory engine.")
store = MemoryStore(
    DB_PATH,
    embedder=get_embedder(EMBEDDER_NAME, model=EMBEDDING_MODEL, cache_size=EMBEDDING_CACHE_SIZE),
    vector_index=VECTOR_INDEX,
    vector_index_overfetch=VECTOR_INDEX_OVERFETCH,
)


@app.get("/healthz")
def healthz() -> dict:
    """Unauthenticated liveness/readiness probe for container orchestrators."""
    store.stats()  # touches the DB connection so a broken store fails the check
    return {"status": "ok"}


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: Optional[str] = Security(_api_key_header)) -> None:
    if not API_KEYS:
        return  # auth disabled
    if not provided or not any(secrets.compare_digest(provided, key) for key in API_KEYS):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key header")


_rate_limiter = (
    TokenBucketLimiter(rate_per_second=RATE_LIMIT / RATE_LIMIT_WINDOW, capacity=RATE_LIMIT) if RATE_LIMIT > 0 else None
)


def enforce_rate_limit(request: Request, provided: Optional[str] = Security(_api_key_header)) -> None:
    if _rate_limiter is None:
        return
    key = provided or (request.client.host if request.client else "unknown")
    if not _rate_limiter.allow(key):
        raise HTTPException(status_code=429, detail="rate limit exceeded, slow down")


api = APIRouter(prefix="/api", dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)])


# -- schemas --------------------------------------------------------------


def _check_metadata_size(v: Optional[dict]) -> Optional[dict]:
    if v and len(json.dumps(v)) > MAX_METADATA_BYTES:
        raise ValueError(f"metadata too large (max {MAX_METADATA_BYTES} bytes serialized)")
    return v


class AddRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=MAX_CONTENT_LENGTH)
    tags: list[str] = Field(default=[], max_length=100)
    source: Optional[str] = Field(None, max_length=200)
    importance: float = Field(1.0, ge=0.0, le=100.0)
    metadata: dict = {}

    _validate_metadata = field_validator("metadata")(_check_metadata_size)


class UpdateRequest(BaseModel):
    content: Optional[str] = Field(None, min_length=1, max_length=MAX_CONTENT_LENGTH)
    tags: Optional[list[str]] = Field(None, max_length=100)
    source: Optional[str] = Field(None, max_length=200)
    importance: Optional[float] = Field(None, ge=0.0, le=100.0)
    metadata: Optional[dict] = None

    _validate_metadata = field_validator("metadata")(_check_metadata_size)


class LinkRequest(BaseModel):
    src_id: int
    dst_id: int
    relation: str = Field("related", min_length=1, max_length=100)


class CompactRequest(BaseModel):
    retention_threshold: float = Field(0.15, ge=0.0, le=1.0)
    cluster_similarity: float = Field(0.75, ge=-1.0, le=1.0)
    summarizer: str = "none"  # "none" or "claude"
    summarizer_model: Optional[str] = None


class ImportRequest(BaseModel):
    version: int = 1
    memories: list[dict] = []
    links: list[dict] = []


def _memory_out(m: Memory) -> dict:
    return {
        "id": m.id,
        "content": m.content,
        "tags": m.tags,
        "source": m.source,
        "metadata": m.metadata,
        "importance": m.importance,
        "created_at": m.created_at,
        "last_accessed": m.last_accessed,
        "access_count": m.access_count,
        "status": m.status,
    }


def _result_out(r: RecallResult) -> dict:
    return {
        "memory": _memory_out(r.memory),
        "similarity": r.similarity,
        "retention": r.retention,
        "score": r.score,
    }


# -- API --------------------------------------------------------------------


@api.post("/memories")
def add_memory(req: AddRequest) -> dict:
    memory_id = store.add(
        req.content, tags=req.tags, source=req.source, importance=req.importance, metadata=req.metadata
    )
    return {"id": memory_id}


@api.get("/memories")
def list_memories(
    status: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    items = store.list_memories(status=status, tag=tag, limit=limit, offset=offset)
    total = store.count_memories(status=status, tag=tag)
    return {"items": [_memory_out(m) for m in items], "total": total, "limit": limit, "offset": offset}


@api.get("/memories/{memory_id}")
def get_memory(memory_id: int) -> dict:
    memory = store.get(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return _memory_out(memory)


@api.patch("/memories/{memory_id}")
def update_memory(memory_id: int, req: UpdateRequest) -> dict:
    if store.get(memory_id) is None:
        raise HTTPException(status_code=404, detail="memory not found")
    store.update(
        memory_id,
        content=req.content,
        tags=req.tags,
        source=req.source,
        importance=req.importance,
        metadata=req.metadata,
    )
    updated = store.get(memory_id)
    if updated is None:
        # deleted by a concurrent request between the check above and here
        raise HTTPException(status_code=404, detail="memory not found")
    return _memory_out(updated)


@api.delete("/memories/{memory_id}")
def delete_memory(memory_id: int) -> dict:
    if store.get(memory_id) is None:
        raise HTTPException(status_code=404, detail="memory not found")
    store.delete(memory_id)
    return {"ok": True}


@api.get("/memories/{memory_id}/links")
def get_links(memory_id: int) -> list[dict]:
    return [{"relation": rel, "memory": _memory_out(m)} for rel, m in store.links_for(memory_id)]


@api.post("/links")
def add_link(req: LinkRequest) -> dict:
    store.link(req.src_id, req.dst_id, relation=req.relation)
    return {"ok": True}


@api.get("/recall")
def recall(q: str = Query(..., min_length=1, max_length=MAX_CONTENT_LENGTH), k: int = Query(5, ge=1, le=100)) -> list[dict]:
    return [_result_out(r) for r in store.recall(q, k=k)]


@api.get("/context")
def context(
    q: str = Query(..., min_length=1, max_length=MAX_CONTENT_LENGTH),
    budget: int = Query(500, ge=1, le=50_000),
) -> dict:
    return {"context": store.build_context(q, token_budget=budget)}


@api.get("/review")
def review(threshold: float = Query(0.3, ge=0.0, le=1.0), limit: int = Query(10, ge=1, le=500)) -> list[dict]:
    return [_result_out(r) for r in store.review_due(threshold=threshold, limit=limit)]


@api.post("/compact")
def compact(req: CompactRequest) -> dict:
    summarizer = None
    if req.summarizer == "claude":
        from .summarizer import claude_summarizer

        try:
            summarizer = claude_summarizer(model=req.summarizer_model or "claude-opus-5")
        except ImportError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    elif req.summarizer != "none":
        raise HTTPException(status_code=400, detail="summarizer must be 'none' or 'claude'")

    new_ids = store.compact(
        retention_threshold=req.retention_threshold,
        cluster_similarity=req.cluster_similarity,
        summarizer=summarizer,
    )
    return {"new_ids": new_ids}


@api.get("/stats")
def stats() -> dict:
    return store.stats()


@api.get("/tags")
def tags() -> list[dict]:
    return [{"tag": tag, "count": count} for tag, count in store.tags()]


@api.get("/export")
def export_data() -> dict:
    return store.export_data()


@api.post("/import")
def import_data(req: ImportRequest, reset: bool = False) -> dict:
    return store.import_data(req.model_dump(), reset=reset)


@api.post("/reindex")
def reindex() -> dict:
    return {"reindexed": store.reindex()}


@api.get("/config")
def config() -> dict:
    return {
        "db_path": DB_PATH,
        "embedder": EMBEDDER_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dims": store.embedder.dims,
        "embedding_cache_size": EMBEDDING_CACHE_SIZE,
        "vector_index": VECTOR_INDEX,
        "vector_index_overfetch": VECTOR_INDEX_OVERFETCH if VECTOR_INDEX else None,
        "auth_enabled": bool(API_KEYS),
        "api_key_count": len(API_KEYS),
        "rate_limit": {"requests": RATE_LIMIT, "window_seconds": RATE_LIMIT_WINDOW} if _rate_limiter else None,
    }


@api.get("/auth/check")
def auth_check() -> dict:
    """Lets the web UI verify a stored key without touching real data."""
    return {"ok": True}


app.include_router(api)


# -- static web UI ---------------------------------------------------------

if _STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=_STATIC_DIR), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")
