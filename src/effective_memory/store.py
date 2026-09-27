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
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .decay import DEFAULT_HALF_LIFE_SECONDS, retention
from .embeddings import Embedder, HashingEmbedder, cosine_similarity
from .vector_index import VectorIndex

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
        vector_index: bool = False,
        vector_index_overfetch: int = 50,
    ):
        self.embedder = embedder or HashingEmbedder()
        self.half_life = half_life
        self._clock = clock
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        self._vector_index_overfetch = vector_index_overfetch
        self._vector_index: Optional[VectorIndex] = None
        if vector_index:
            self._vector_index = VectorIndex(self._conn, dims=self.embedder.dims)
            self._sync_vector_index()

    def _sync_vector_index(self) -> None:
        """Rebuild the vector index from `memories` if it's out of sync --
        e.g. it was just enabled against a store that already has data.
        """
        active_count = self._conn.execute("SELECT COUNT(*) FROM memories WHERE status = 'active'").fetchone()[0]
        if self._vector_index.count() == active_count:
            return
        self._vector_index.clear()
        for row in self._conn.execute("SELECT id, embedding FROM memories WHERE status = 'active'"):
            self._vector_index.upsert(row["id"], json.loads(row["embedding"]))
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
        with self._lock:
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
            memory_id = cur.lastrowid
            if self._vector_index is not None:
                self._vector_index.upsert(memory_id, embedding)
                self._conn.commit()
            return memory_id

    def update(
        self,
        memory_id: int,
        content: Optional[str] = None,
        tags: Optional[list[str]] = None,
        source: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Edit a memory in place. Re-embeds and re-indexes when `content`
        changes; other fields are updated as given, left alone otherwise.
        """
        with self._lock:
            fields, params = [], []
            if content is not None:
                fields.append("content = ?")
                params.append(content)
                embedding = self.embedder.embed(content)
                fields.append("embedding = ?")
                params.append(json.dumps(embedding))
            if tags is not None:
                fields.append("tags = ?")
                params.append(json.dumps(tags))
            if source is not None:
                fields.append("source = ?")
                params.append(source)
            if importance is not None:
                fields.append("importance = ?")
                params.append(importance)
            if metadata is not None:
                fields.append("metadata = ?")
                params.append(json.dumps(metadata))
            if not fields:
                return
            params.append(memory_id)
            self._conn.execute(f"UPDATE memories SET {', '.join(fields)} WHERE id = ?", params)
            self._conn.commit()
            if content is not None and self._vector_index is not None:
                self._vector_index.upsert(memory_id, embedding)
                self._conn.commit()

    def delete(self, memory_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._conn.commit()
            if self._vector_index is not None:
                self._vector_index.delete(memory_id)
                self._conn.commit()

    def link(self, src_id: int, dst_id: int, relation: str = "related") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO links (src_id, dst_id, relation) VALUES (?, ?, ?)",
                (src_id, dst_id, relation),
            )
            self._conn.commit()

    def links_for(self, memory_id: int) -> list[tuple[str, Memory]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT l.relation as relation, m.* FROM links l "
                "JOIN memories m ON m.id = l.dst_id WHERE l.src_id = ?",
                (memory_id,),
            ).fetchall()
            return [(row["relation"], _row_to_memory(row)) for row in rows]

    def touch(self, memory_id: int) -> None:
        """Record an access: strengthens the memory and slows its decay."""
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (self._clock(), memory_id),
            )
            self._conn.commit()

    # -- reading -------------------------------------------------------------

    def get(self, memory_id: int) -> Optional[Memory]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            return _row_to_memory(row) if row else None

    def _list_filters(self, status: Optional[str], tag: Optional[str]) -> tuple[str, list]:
        clauses, params = [], []
        if status and status != "all":
            clauses.append("status = ?")
            params.append(status)
        if tag:
            clauses.append("tags LIKE ?")
            params.append(f'%"{tag}"%')
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    def list_memories(
        self,
        status: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Memory]:
        """Browse memories most-recent-first. `status` is 'active',
        'compacted', or None/'all' for everything.
        """
        with self._lock:
            where, params = self._list_filters(status, tag)
            rows = self._conn.execute(
                f"SELECT * FROM memories{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()
            return [_row_to_memory(row) for row in rows]

    def count_memories(self, status: Optional[str] = None, tag: Optional[str] = None) -> int:
        with self._lock:
            where, params = self._list_filters(status, tag)
            return self._conn.execute(f"SELECT COUNT(*) FROM memories{where}", params).fetchone()[0]

    def _active_rows(self, include_archived: bool = False) -> list[sqlite3.Row]:
        if include_archived:
            return self._conn.execute("SELECT * FROM memories WHERE status != 'compacted'").fetchall()
        return self._conn.execute("SELECT * FROM memories WHERE status = 'active'").fetchall()

    def _score(self, row: sqlite3.Row, sim: float, now: float) -> RecallResult:
        elapsed = now - row["last_accessed"]
        ret = retention(elapsed, row["importance"], row["access_count"], self.half_life)
        # never fully zero out a strong semantic match just because it decayed
        score = sim * (0.3 + 0.7 * ret)
        return RecallResult(memory=_row_to_memory(row), similarity=sim, retention=ret, score=score)

    def _recall_brute_force(self, query_vec: list[float], k: int, now: float, include_archived: bool) -> list[RecallResult]:
        results = []
        for row in self._active_rows(include_archived):
            sim = cosine_similarity(query_vec, json.loads(row["embedding"]))
            results.append(self._score(row, sim, now))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def _recall_via_vector_index(self, query_vec: list[float], k: int, now: float) -> list[RecallResult]:
        """Overfetch a candidate pool by raw cosine similarity from the ANN
        index, then re-rank that (much smaller) pool by decay-weighted
        score in Python. This avoids scanning every row for large stores,
        at the cost of only ever re-ranking within the top `overfetch`
        candidates by pure similarity.
        """
        overfetch = max(k, self._vector_index_overfetch)
        candidates = self._vector_index.search(query_vec, limit=overfetch)
        if not candidates:
            return []
        similarities = dict(candidates)
        placeholders = ",".join("?" * len(similarities))
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders}) AND status = 'active'",
            list(similarities),
        ).fetchall()
        results = [self._score(row, similarities[row["id"]], now) for row in rows]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

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
        with self._lock:
            query_vec = self.embedder.embed(query)
            now = self._clock()
            if self._vector_index is not None and not include_archived:
                top = self._recall_via_vector_index(query_vec, k, now)
            else:
                top = self._recall_brute_force(query_vec, k, now, include_archived)
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
        with self._lock:
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
        with self._lock:
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
                    if self._vector_index is not None:
                        self._vector_index.delete(row["id"])
                self._conn.commit()
                new_ids.append(new_id)
            return new_ids

    def stats(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) as n FROM memories GROUP BY status"
            ).fetchall()
            by_status = {row["status"]: row["n"] for row in rows}
            total = sum(by_status.values())
            links = self._conn.execute("SELECT COUNT(*) as n FROM links").fetchone()["n"]
            return {"total": total, "by_status": by_status, "links": links}

    def tags(self) -> list[tuple[str, int]]:
        """Distinct tags across all memories, with counts, most-used first."""
        with self._lock:
            counts: dict[str, int] = {}
            for row in self._conn.execute("SELECT tags FROM memories"):
                for tag in json.loads(row["tags"]):
                    counts[tag] = counts.get(tag, 0) + 1
            return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    # -- backup / restore --------------------------------------------------

    def export_data(self) -> dict:
        """Dump every memory and link as plain data (no embeddings -- those
        are re-derived on import, since they're tied to whichever embedder
        made them).
        """
        with self._lock:
            memories = self._conn.execute("SELECT * FROM memories ORDER BY id").fetchall()
            links = self._conn.execute("SELECT src_id, dst_id, relation FROM links ORDER BY id").fetchall()
            return {
                "version": 1,
                "exported_at": self._clock(),
                "memories": [
                    {
                        "id": row["id"],
                        "content": row["content"],
                        "tags": json.loads(row["tags"]),
                        "source": row["source"],
                        "metadata": json.loads(row["metadata"]),
                        "importance": row["importance"],
                        "created_at": row["created_at"],
                        "last_accessed": row["last_accessed"],
                        "access_count": row["access_count"],
                        "status": row["status"],
                    }
                    for row in memories
                ],
                "links": [
                    {"src_id": row["src_id"], "dst_id": row["dst_id"], "relation": row["relation"]}
                    for row in links
                ],
            }

    def import_data(self, data: dict, reset: bool = False) -> dict:
        """Load an export_data() dump, re-embedding every memory with the
        current embedder. If `reset` is True, existing data is wiped first;
        otherwise memories are appended with fresh IDs (links and any
        `metadata.compacted_from` references are remapped to match).
        """
        with self._lock:
            if reset:
                self._conn.execute("DELETE FROM links")
                self._conn.execute("DELETE FROM memories")
                if self._vector_index is not None:
                    self._vector_index.clear()
                self._conn.commit()

            id_map: dict[int, int] = {}
            for mem in data.get("memories", []):
                metadata = dict(mem.get("metadata") or {})
                if "compacted_from" in metadata:
                    metadata["compacted_from"] = [id_map.get(i, i) for i in metadata["compacted_from"]]

                embedding = self.embedder.embed(mem["content"])
                now = self._clock()
                cur = self._conn.execute(
                    "INSERT INTO memories (content, embedding, tags, source, metadata, importance, "
                    "created_at, last_accessed, access_count, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        mem["content"],
                        json.dumps(embedding),
                        json.dumps(mem.get("tags", [])),
                        mem.get("source"),
                        json.dumps(metadata),
                        mem.get("importance", 1.0),
                        mem.get("created_at", now),
                        mem.get("last_accessed", now),
                        mem.get("access_count", 0),
                        mem.get("status", "active"),
                    ),
                )
                new_id = cur.lastrowid
                id_map[mem["id"]] = new_id
                if self._vector_index is not None and mem.get("status", "active") == "active":
                    self._vector_index.upsert(new_id, embedding)

            imported_links = 0
            for link in data.get("links", []):
                src, dst = id_map.get(link["src_id"]), id_map.get(link["dst_id"])
                if src is not None and dst is not None:
                    self._conn.execute(
                        "INSERT INTO links (src_id, dst_id, relation) VALUES (?, ?, ?)",
                        (src, dst, link.get("relation", "related")),
                    )
                    imported_links += 1
            self._conn.commit()
            return {"memories": len(id_map), "links": imported_links}

    def reindex(self) -> int:
        """Re-embed every memory's content with the current embedder and
        refresh the vector index. Use this after switching --embedder on a
        store that already has data.
        """
        with self._lock:
            rows = self._conn.execute("SELECT id, content, status FROM memories").fetchall()
            for row in rows:
                embedding = self.embedder.embed(row["content"])
                self._conn.execute("UPDATE memories SET embedding = ? WHERE id = ?", (json.dumps(embedding), row["id"]))
                if self._vector_index is not None and row["status"] == "active":
                    self._vector_index.upsert(row["id"], embedding)
            self._conn.commit()
            return len(rows)
