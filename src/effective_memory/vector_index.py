"""Optional sqlite-vec backed ANN index, so `recall()` doesn't have to
deserialize and cosine-score every row in Python once a store grows large.

Requires the 'vector-index' extra. Opt-in: MemoryStore only builds one when
asked (vector_index=True), since it adds a dependency and a fixed embedding
dimension for the table's lifetime.
"""

from __future__ import annotations

import sqlite3


class VectorIndex:
    def __init__(self, conn: sqlite3.Connection, dims: int, table: str = "vec_memories"):
        try:
            import sqlite_vec
        except ImportError as e:
            raise ImportError(
                "vector_index=True requires the 'vector-index' extra: "
                "pip install 'effective-memory[vector-index]'"
            ) from e
        self._sqlite_vec = sqlite_vec
        self._table = table
        self._conn = conn

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
            f"memory_id INTEGER PRIMARY KEY, embedding float[{dims}] distance_metric=cosine)"
        )

    def upsert(self, memory_id: int, embedding: list[float]) -> None:
        packed = self._sqlite_vec.serialize_float32(embedding)
        self._conn.execute(f"DELETE FROM {self._table} WHERE memory_id = ?", (memory_id,))
        self._conn.execute(f"INSERT INTO {self._table}(memory_id, embedding) VALUES (?, ?)", (memory_id, packed))

    def delete(self, memory_id: int) -> None:
        self._conn.execute(f"DELETE FROM {self._table} WHERE memory_id = ?", (memory_id,))

    def count(self) -> int:
        return self._conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()[0]

    def clear(self) -> None:
        self._conn.execute(f"DELETE FROM {self._table}")

    def search(self, query_embedding: list[float], limit: int) -> list[tuple[int, float]]:
        """Return up to `limit` (memory_id, cosine_similarity) pairs, best first."""
        packed = self._sqlite_vec.serialize_float32(query_embedding)
        rows = self._conn.execute(
            f"SELECT memory_id, distance FROM {self._table} WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (packed, limit),
        ).fetchall()
        return [(row[0], 1.0 - row[1]) for row in rows]  # cosine distance -> similarity
