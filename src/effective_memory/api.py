"""FastAPI wrapper around MemoryStore, plus a minimal static web UI.

Run with:

    emem serve --db mem.db --port 8000

or directly:

    uvicorn effective_memory.api:app

Configuration (env vars, all optional):
    EFFECTIVE_MEMORY_DB               path to the SQLite store
    EFFECTIVE_MEMORY_EMBEDDER         hashing | voyage | openai | sentence-transformers
    EFFECTIVE_MEMORY_EMBEDDING_MODEL  model name for the chosen embedder
    EFFECTIVE_MEMORY_API_KEY          required on the "X-API-Key" header for
                                       every /api/* request; set to "" to
                                       disable auth (local/dev use only)
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Security
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .embeddings import get_embedder
from .store import Memory, MemoryStore, RecallResult

DB_PATH = os.environ.get("EFFECTIVE_MEMORY_DB", "effective_memory.db")
EMBEDDER_NAME = os.environ.get("EFFECTIVE_MEMORY_EMBEDDER", "hashing")
EMBEDDING_MODEL = os.environ.get("EFFECTIVE_MEMORY_EMBEDDING_MODEL")
API_KEY = os.environ.get("EFFECTIVE_MEMORY_API_KEY")  # unset or "" disables auth
_STATIC_DIR = Path(__file__).parent / "web"

app = FastAPI(title="effective-memory", description="A local-first memory engine.")
store = MemoryStore(DB_PATH, embedder=get_embedder(EMBEDDER_NAME, model=EMBEDDING_MODEL))

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: Optional[str] = Security(_api_key_header)) -> None:
    if not API_KEY:
        return  # auth disabled
    if not provided or not secrets.compare_digest(provided, API_KEY):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key header")


api = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])


# -- schemas --------------------------------------------------------------


class AddRequest(BaseModel):
    content: str
    tags: list[str] = []
    source: Optional[str] = None
    importance: float = 1.0
    metadata: dict = {}


class LinkRequest(BaseModel):
    src_id: int
    dst_id: int
    relation: str = "related"


class CompactRequest(BaseModel):
    retention_threshold: float = 0.15
    cluster_similarity: float = 0.75


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


@api.get("/memories/{memory_id}")
def get_memory(memory_id: int) -> dict:
    memory = store.get(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return _memory_out(memory)


@api.get("/memories/{memory_id}/links")
def get_links(memory_id: int) -> list[dict]:
    return [{"relation": rel, "memory": _memory_out(m)} for rel, m in store.links_for(memory_id)]


@api.post("/links")
def add_link(req: LinkRequest) -> dict:
    store.link(req.src_id, req.dst_id, relation=req.relation)
    return {"ok": True}


@api.get("/recall")
def recall(q: str, k: int = 5) -> list[dict]:
    return [_result_out(r) for r in store.recall(q, k=k)]


@api.get("/context")
def context(q: str, budget: int = 500) -> dict:
    return {"context": store.build_context(q, token_budget=budget)}


@api.get("/review")
def review(threshold: float = 0.3, limit: int = 10) -> list[dict]:
    return [_result_out(r) for r in store.review_due(threshold=threshold, limit=limit)]


@api.post("/compact")
def compact(req: CompactRequest) -> dict:
    new_ids = store.compact(retention_threshold=req.retention_threshold, cluster_similarity=req.cluster_similarity)
    return {"new_ids": new_ids}


@api.get("/stats")
def stats() -> dict:
    return store.stats()


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
