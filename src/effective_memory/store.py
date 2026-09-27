"""SQLite-backed memory store.

One store serves three roles at once:
  - agent memory: `recall()` / `build_context()` retrieve the most relevant,
    least-forgotten memories to inject into an LLM prompt under a token budget.
  - personal knowledge base: `add()` + `link()` capture notes and the
    relations between them; `review_due()` surfaces things worth revisiting,
    spaced-repetition style.
  - efficient memory management: `compact()` finds memories that have decayed
    past a forgetting threshold, clusters similar ones, and replaces them
    with a single summary -- bounding storage and context size over time.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .decay import DEFAULT_HALF_LIFE_SECONDS, retention
from .embeddings import Embedder, HashingEmbedder, cosine_similarity

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL,
    embedding TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    source TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    importance REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    dst_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation TEXT NOT NULL DEFAULT 'related'
);

CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_links_src ON links(src_id);
CREATE INDEX IF NOT EXISTS idx_links_dst ON links(dst_id);
"""


@dataclass
class Memory:
    id: int
    content: str
    tags: list[str]
    source: Optional[str]
    metadata: dict
    importance: float
    created_at: float
    last_accessed: float
    access_count: int
    status: str


@dataclass
class RecallResult:
    memory: Memory
    similarity: float
    retention: float
    score: float


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=row["id"],
        content=row["content"],
        tags=json.loads(row["tags"]),
        source=row["source"],
        metadata=json.loads(row["metadata"]),
        importance=row["importance"],
        created_at=row["created_at"],
        last_accessed=row["last_accessed"],
        access_count=row["access_count"],
        status=row["status"],
    )


class MemoryStore:
    def __init__(
        self,
        path: str = "effective_memory.db",
        embedder: Optional[Embedder] = None,
        half_life: float = DEFAULT_HALF_LIFE_SECONDS,
        clock: Callable[[], float] = time.time,
    ):
        self.embedder = embedder or HashingEmbedder()
        self.half_life = half_life
        self._clock = clock
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writing -----------------------------------------------------------

    def add(
        self,
        content: str,
        tags: Optional[list[str]] = None,
        source: Optional[str] = None,
        importance: float = 1.0,
        metadata: Optional[dict] = None,
    ) -> int:
        now = self._clock()
        embedding = self.embedder.embed(content)
        cur = self._conn.execute(
            "INSERT INTO memories (content, embedding, tags, source, metadata, importance, "
            "created_at, last_accessed, access_count, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 'active')",
            (
                content,
                json.dumps(embedding),
                json.dumps(tags or []),
                source,
                json.dumps(metadata or {}),
                importance,
                now,
                now,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def link(self, src_id: int, dst_id: int, relation: str = "related") -> None:
        self._conn.execute(
            "INSERT INTO links (src_id, dst_id, relation) VALUES (?, ?, ?)",
            (src_id, dst_id, relation),
        )
        self._conn.commit()

    def links_for(self, memory_id: int) -> list[tuple[str, Memory]]:
        rows = self._conn.execute(
            "SELECT l.relation as relation, m.* FROM links l "
            "JOIN memories m ON m.id = l.dst_id WHERE l.src_id = ?",
            (memory_id,),
        ).fetchall()
        return [(row["relation"], _row_to_memory(row)) for row in rows]

    def touch(self, memory_id: int) -> None:
        """Record an access: strengthens the memory and slows its decay."""
        self._conn.execute(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (self._clock(), memory_id),
        )
        self._conn.commit()

    # -- reading -------------------------------------------------------------

    def get(self, memory_id: int) -> Optional[Memory]:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return _row_to_memory(row) if row else None

    def _active_rows(self, include_archived: bool = False) -> list[sqlite3.Row]:
        if include_archived:
            return self._conn.execute("SELECT * FROM memories WHERE status != 'compacted'").fetchall()
        return self._conn.execute("SELECT * FROM memories WHERE status = 'active'").fetchall()

    def recall(
        self,
        query: str,
        k: int = 5,
        include_archived: bool = False,
        touch_on_recall: bool = True,
    ) -> list[RecallResult]:
        """Return the top-k memories relevant to `query`, ranked by
        similarity weighted by how well-retained the memory currently is.
        """
        query_vec = self.embedder.embed(query)
        now = self._clock()
        results: list[RecallResult] = []
        for row in self._active_rows(include_archived):
            embedding = json.loads(row["embedding"])
            sim = cosine_similarity(query_vec, embedding)
            elapsed = now - row["last_accessed"]
            ret = retention(elapsed, row["importance"], row["access_count"], self.half_life)
            # never fully zero out a strong semantic match just because it decayed
            score = sim * (0.3 + 0.7 * ret)
            results.append(RecallResult(memory=_row_to_memory(row), similarity=sim, retention=ret, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        top = results[:k]
        if touch_on_recall:
            for r in top:
                self.touch(r.memory.id)
        return top

    def build_context(self, query: str, token_budget: int = 500, k: int = 20) -> str:
        """Build a prompt-ready context block under an approximate token budget.

        Uses a ~4-chars-per-token heuristic to avoid a tokenizer dependency.
        """
        char_budget = token_budget * 4
        chunks: list[str] = []
        used = 0
        for r in self.recall(query, k=k):
            piece = f"- {r.memory.content}"
            if used + len(piece) > char_budget:
                break
            chunks.append(piece)
            used += len(piece)
        return "\n".join(chunks)

    def review_due(self, threshold: float = 0.3, limit: int = 10) -> list[RecallResult]:
        """Memories that have decayed below `threshold` retention -- worth
        rehearsing (like a spaced-repetition review queue) before they're
        swept up by `compact()`.
        """
        now = self._clock()
        due: list[RecallResult] = []
        for row in self._active_rows():
            elapsed = now - row["last_accessed"]
            ret = retention(elapsed, row["importance"], row["access_count"], self.half_life)
            if ret < threshold:
                due.append(RecallResult(memory=_row_to_memory(row), similarity=1.0, retention=ret, score=ret))
        due.sort(key=lambda r: r.retention)
        return due[:limit]

    # -- compaction ------------------------------------------------------

    def compact(
        self,
        retention_threshold: float = 0.15,
        cluster_similarity: float = 0.75,
        summarizer: Optional[Callable[[list[str]], str]] = None,
    ) -> list[int]:
        """Cluster decayed memories by semantic similarity and replace each
        cluster with one summary memory, marking the originals 'compacted'.

        `summarizer` takes the list of clustered contents and returns a
        single string; defaults to a plain extractive join, but pass an
        LLM-backed callable for a real abstractive summary.
        """
        now = self._clock()
        candidates = []
        for row in self._active_rows():
            elapsed = now - row["last_accessed"]
            ret = retention(elapsed, row["importance"], row["access_count"], self.half_life)
            if ret < retention_threshold:
                candidates.append((row, json.loads(row["embedding"])))

        clusters: list[list[sqlite3.Row]] = []
        cluster_vecs: list[list[float]] = []
        for row, vec in candidates:
            placed = False
            for i, cvec in enumerate(cluster_vecs):
                if cosine_similarity(vec, cvec) >= cluster_similarity:
                    clusters[i].append(row)
                    placed = True
                    break
            if not placed:
                clusters.append([row])
                cluster_vecs.append(vec)

        summarizer = summarizer or (lambda contents: " | ".join(contents))
        new_ids: list[int] = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue  # nothing to gain by "compacting" a single memory
            contents = [row["content"] for row in cluster]
            summary_text = summarizer(contents)
            merged_importance = max(row["importance"] for row in cluster)
            new_id = self.add(
                summary_text,
                tags=sorted(set(sum((json.loads(row["tags"]) for row in cluster), []))),
                source="compaction",
                importance=merged_importance,
                metadata={"compacted_from": [row["id"] for row in cluster]},
            )
            for row in cluster:
                self._conn.execute(
                    "UPDATE memories SET status = 'compacted' WHERE id = ?", (row["id"],)
                )
            self._conn.commit()
            new_ids.append(new_id)
        return new_ids

    def stats(self) -> dict:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as n FROM memories GROUP BY status"
        ).fetchall()
        by_status = {row["status"]: row["n"] for row in rows}
        total = sum(by_status.values())
        links = self._conn.execute("SELECT COUNT(*) as n FROM links").fetchone()["n"]
        return {"total": total, "by_status": by_status, "links": links}
