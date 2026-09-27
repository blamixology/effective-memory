"""Command-line interface for effective-memory.

    emem add "text" [--tags a,b] [--source name] [--importance 1.0]
    emem recall "query" [--k 5]
    emem context "query" [--budget 500]
    emem review [--threshold 0.3] [--limit 10]
    emem compact [--threshold 0.15]
    emem stats
"""

from __future__ import annotations

import argparse
import sys

from .store import MemoryStore

DEFAULT_DB = "effective_memory.db"


def _add(args: argparse.Namespace) -> None:
    with MemoryStore(args.db) as store:
        tags = args.tags.split(",") if args.tags else []
        memory_id = store.add(args.text, tags=tags, source=args.source, importance=args.importance)
        print(f"added memory #{memory_id}")


def _recall(args: argparse.Namespace) -> None:
    with MemoryStore(args.db) as store:
        results = store.recall(args.query, k=args.k)
        if not results:
            print("(no memories yet)")
            return
        for r in results:
            print(f"[{r.score:.3f}] (#{r.memory.id}, retention={r.retention:.2f}) {r.memory.content}")


def _context(args: argparse.Namespace) -> None:
    with MemoryStore(args.db) as store:
        print(store.build_context(args.query, token_budget=args.budget))


def _review(args: argparse.Namespace) -> None:
    with MemoryStore(args.db) as store:
        due = store.review_due(threshold=args.threshold, limit=args.limit)
        if not due:
            print("nothing due for review")
            return
        for r in due:
            print(f"(#{r.memory.id}, retention={r.retention:.2f}) {r.memory.content}")


def _compact(args: argparse.Namespace) -> None:
    with MemoryStore(args.db) as store:
        new_ids = store.compact(retention_threshold=args.threshold)
        if not new_ids:
            print("nothing to compact")
        else:
            print(f"compacted into {len(new_ids)} summary memories: {new_ids}")


def _stats(args: argparse.Namespace) -> None:
    with MemoryStore(args.db) as store:
        s = store.stats()
        print(f"total: {s['total']}  links: {s['links']}")
        for status, n in s["by_status"].items():
            print(f"  {status}: {n}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="emem", description="A local-first memory engine.")
    parser.add_argument("--db", default=DEFAULT_DB, help="path to the SQLite store")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="store a new memory")
    p_add.add_argument("text")
    p_add.add_argument("--tags", default="")
    p_add.add_argument("--source", default=None)
    p_add.add_argument("--importance", type=float, default=1.0)
    p_add.set_defaults(func=_add)

    p_recall = sub.add_parser("recall", help="semantically search memories")
    p_recall.add_argument("query")
    p_recall.add_argument("--k", type=int, default=5)
    p_recall.set_defaults(func=_recall)

    p_context = sub.add_parser("context", help="build an LLM-ready context block")
    p_context.add_argument("query")
    p_context.add_argument("--budget", type=int, default=500, help="approx token budget")
    p_context.set_defaults(func=_context)

    p_review = sub.add_parser("review", help="list memories due for review (spaced repetition)")
    p_review.add_argument("--threshold", type=float, default=0.3)
    p_review.add_argument("--limit", type=int, default=10)
    p_review.set_defaults(func=_review)

    p_compact = sub.add_parser("compact", help="cluster and summarize decayed memories")
    p_compact.add_argument("--threshold", type=float, default=0.15)
    p_compact.set_defaults(func=_compact)

    p_stats = sub.add_parser("stats", help="show store statistics")
    p_stats.set_defaults(func=_stats)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
