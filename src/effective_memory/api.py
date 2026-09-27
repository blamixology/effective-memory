"""FastAPI wrapper around MemoryStore, plus a minimal static web UI.

Run with:

    emem serve --db mem.db --port 8000

or directly:

    uvicorn effective_memory.api:app
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .store import Memory, MemoryStore, RecallResult

DB_PATH = os.environ.get("EFFECTIVE_MEMORY_DB", "effective_memory.db")
_STATIC_DIR = Path(__file__).parent / "web"

app = FastAPI(title="effective-memory", description="A local-first memory engine.")
store = MemoryStore(DB_PATH)


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


@app.post("/api/memories")
def add_memory(req: AddRequest) -> dict:
    memory_id = store.add(
        req.content, tags=req.tags, source=req.source, importance=req.importance, metadata=req.metadata
    )
    return {"id": memory_id}


@app.get("/api/memories/{memory_id}")
def get_memory(memory_id: int) -> dict:
    memory = store.get(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return _memory_out(memory)


@app.get("/api/memories/{memory_id}/links")
def get_links(memory_id: int) -> list[dict]:
    return [{"relation": rel, "memory": _memory_out(m)} for rel, m in store.links_for(memory_id)]


@app.post("/api/links")
def add_link(req: LinkRequest) -> dict:
    store.link(req.src_id, req.dst_id, relation=req.relation)
    return {"ok": True}


@app.get("/api/recall")
def recall(q: str, k: int = 5) -> list[dict]:
    return [_result_out(r) for r in store.recall(q, k=k)]


@app.get("/api/context")
def context(q: str, budget: int = 500) -> dict:
    return {"context": store.build_context(q, token_budget=budget)}


@app.get("/api/review")
def review(threshold: float = 0.3, limit: int = 10) -> list[dict]:
    return [_result_out(r) for r in store.review_due(threshold=threshold, limit=limit)]


@app.post("/api/compact")
def compact(req: CompactRequest) -> dict:
    new_ids = store.compact(retention_threshold=req.retention_threshold, cluster_similarity=req.cluster_similarity)
    return {"new_ids": new_ids}


@app.get("/api/stats")
def stats() -> dict:
    return store.stats()


# -- static web UI ---------------------------------------------------------

if _STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=_STATIC_DIR), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")
