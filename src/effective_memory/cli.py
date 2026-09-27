"""Command-line interface for effective-memory.

    emem add "text" [--tags a,b] [--source name] [--importance 1.0]
    emem recall "query" [--k 5]
    emem context "query" [--budget 500]
    emem update ID [--content ...] [--tags a,b] [--source name] [--importance 1.0]
    emem delete ID
    emem list [--status active|compacted|all] [--tag TAG] [--limit 50] [--offset 0]
    emem tags
    emem review [--threshold 0.3] [--limit 10]
    emem compact [--threshold 0.15] [--summarizer none|claude] [--summarizer-model NAME]
    emem export [--out FILE]
    emem import FILE [--reset]
    emem reindex
    emem stats
    emem serve [--host 127.0.0.1] [--port 8000] [--api-key KEY | --no-auth]
               [--rate-limit 120] [--rate-limit-window 60]

Add --embedder {hashing,voyage,openai,sentence-transformers} (default:
hashing) and --embedding-model NAME before the subcommand to use a real
embedding model instead of the dependency-free default. The same choice
must be used consistently for a given --db, since vectors from different
embedders aren't comparable.

Add --vector-index before the subcommand to use a sqlite-vec ANN index for
recall() instead of scanning every row -- worthwhile once a store holds
more than a few thousand memories (requires the 'vector-index' extra).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .embeddings import get_embedder
from .store import MemoryStore

DEFAULT_DB = os.environ.get("EFFECTIVE_MEMORY_DB", "effective_memory.db")
DEFAULT_EMBEDDER = os.environ.get("EFFECTIVE_MEMORY_EMBEDDER", "hashing")
DEFAULT_EMBEDDING_MODEL = os.environ.get("EFFECTIVE_MEMORY_EMBEDDING_MODEL")
DEFAULT_EMBEDDING_CACHE_SIZE = int(os.environ.get("EFFECTIVE_MEMORY_EMBEDDING_CACHE_SIZE", "256"))
DEFAULT_VECTOR_INDEX = os.environ.get("EFFECTIVE_MEMORY_VECTOR_INDEX", "") == "1"
DEFAULT_VECTOR_INDEX_OVERFETCH = int(os.environ.get("EFFECTIVE_MEMORY_VECTOR_INDEX_OVERFETCH", "50"))
DEFAULT_RATE_LIMIT = int(os.environ.get("EFFECTIVE_MEMORY_RATE_LIMIT", "120"))
DEFAULT_RATE_LIMIT_WINDOW = float(os.environ.get("EFFECTIVE_MEMORY_RATE_LIMIT_WINDOW", "60"))


def _open_store(args: argparse.Namespace) -> MemoryStore:
    embedder = get_embedder(args.embedder, model=args.embedding_model, cache_size=args.embedding_cache_size)
    return MemoryStore(
        args.db,
        embedder=embedder,
        vector_index=args.vector_index,
        vector_index_overfetch=args.vector_index_overfetch,
    )


def _add(args: argparse.Namespace) -> None:
    with _open_store(args) as store:
        tags = args.tags.split(",") if args.tags else []
        memory_id = store.add(args.text, tags=tags, source=args.source, importance=args.importance)
        print(f"added memory #{memory_id}")


def _update(args: argparse.Namespace) -> None:
    with _open_store(args) as store:
        tags = args.tags.split(",") if args.tags is not None else None
        store.update(
            args.id,
            content=args.content,
            tags=tags,
            source=args.source,
            importance=args.importance,
        )
        print(f"updated memory #{args.id}")


def _delete(args: argparse.Namespace) -> None:
    with _open_store(args) as store:
        store.delete(args.id)
        print(f"deleted memory #{args.id}")


def _list(args: argparse.Namespace) -> None:
    with _open_store(args) as store:
        items = store.list_memories(status=args.status, tag=args.tag, limit=args.limit, offset=args.offset)
        total = store.count_memories(status=args.status, tag=args.tag)
        if not items:
            print("(no memories match)")
            return
        for m in items:
            print(f"(#{m.id}, {m.status}) {m.content}")
        print(f"-- {args.offset + len(items)}/{total} --")


def _tags(args: argparse.Namespace) -> None:
    with _open_store(args) as store:
        pairs = store.tags()
        if not pairs:
            print("(no tags yet)")
            return
        for tag, count in pairs:
            print(f"{tag}\t{count}")


def _export(args: argparse.Namespace) -> None:
    with _open_store(args) as store:
        dump = store.export_data()
        text = json.dumps(dump, indent=2)
        if args.out:
            with open(args.out, "w") as f:
                f.write(text)
            print(f"exported {len(dump['memories'])} memories, {len(dump['links'])} links to {args.out}")
        else:
            print(text)


def _import(args: argparse.Namespace) -> None:
    with open(args.file) as f:
        dump = json.load(f)
    with _open_store(args) as store:
        result = store.import_data(dump, reset=args.reset)
        print(f"imported {result['memories']} memories, {result['links']} links")


def _reindex(args: argparse.Namespace) -> None:
    with _open_store(args) as store:
        n = store.reindex()
        print(f"reindexed {n} memories with the '{args.embedder}' embedder")


def _recall(args: argparse.Namespace) -> None:
    with _open_store(args) as store:
        results = store.recall(args.query, k=args.k)
        if not results:
            print("(no memories yet)")
            return
        for r in results:
            print(f"[{r.score:.3f}] (#{r.memory.id}, retention={r.retention:.2f}) {r.memory.content}")


def _context(args: argparse.Namespace) -> None:
    with _open_store(args) as store:
        print(store.build_context(args.query, token_budget=args.budget))


def _review(args: argparse.Namespace) -> None:
    with _open_store(args) as store:
        due = store.review_due(threshold=args.threshold, limit=args.limit)
        if not due:
            print("nothing due for review")
            return
        for r in due:
            print(f"(#{r.memory.id}, retention={r.retention:.2f}) {r.memory.content}")


def _compact(args: argparse.Namespace) -> None:
    summarizer = None
    if args.summarizer == "claude":
        from .summarizer import claude_summarizer

        summarizer = claude_summarizer(model=args.summarizer_model or "claude-opus-5")

    with _open_store(args) as store:
        new_ids = store.compact(retention_threshold=args.threshold, summarizer=summarizer)
        if not new_ids:
            print("nothing to compact")
        else:
            print(f"compacted into {len(new_ids)} summary memories: {new_ids}")


def _stats(args: argparse.Namespace) -> None:
    with _open_store(args) as store:
        s = store.stats()
        print(f"total: {s['total']}  links: {s['links']}")
        for status, n in s["by_status"].items():
            print(f"  {status}: {n}")


def _serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print("the web UI/API needs the 'web' extra: pip install 'effective-memory[web]'", file=sys.stderr)
        raise SystemExit(1)

    os.environ["EFFECTIVE_MEMORY_DB"] = args.db
    os.environ["EFFECTIVE_MEMORY_EMBEDDER"] = args.embedder
    if args.embedding_model:
        os.environ["EFFECTIVE_MEMORY_EMBEDDING_MODEL"] = args.embedding_model
    if args.vector_index:
        os.environ["EFFECTIVE_MEMORY_VECTOR_INDEX"] = "1"
        os.environ["EFFECTIVE_MEMORY_VECTOR_INDEX_OVERFETCH"] = str(args.vector_index_overfetch)
    os.environ["EFFECTIVE_MEMORY_EMBEDDING_CACHE_SIZE"] = str(args.embedding_cache_size)
    os.environ["EFFECTIVE_MEMORY_RATE_LIMIT"] = str(args.rate_limit)
    os.environ["EFFECTIVE_MEMORY_RATE_LIMIT_WINDOW"] = str(args.rate_limit_window)

    if args.no_auth:
        os.environ["EFFECTIVE_MEMORY_API_KEY"] = ""
    elif args.api_key:
        os.environ["EFFECTIVE_MEMORY_API_KEY"] = ",".join(args.api_key)
    elif os.environ.get("EFFECTIVE_MEMORY_API_KEY") is None:
        # only generate one when the env var was never set at all -- an
        # explicitly empty value (e.g. from docker-compose) means "no auth"
        # and must be left alone, not silently overridden
        import secrets

        generated = secrets.token_urlsafe(24)
        os.environ["EFFECTIVE_MEMORY_API_KEY"] = generated
        print(f"generated API key (pass it as 'X-API-Key' header, or --api-key next time): {generated}")

    uvicorn.run("effective_memory.api:app", host=args.host, port=args.port)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    # Every default below falls back to the matching EFFECTIVE_MEMORY_*
    # env var (like --db/DEFAULT_DB already did), not a hardcoded literal.
    # `emem serve` unconditionally re-exports args.X into that same env var
    # for the API process to read -- if the default were hardcoded instead,
    # that write would silently clobber an env var set by Docker/compose
    # whenever the matching flag isn't passed on the command line.
    parser.add_argument("--db", default=DEFAULT_DB, help="path to the SQLite store")
    parser.add_argument(
        "--embedder",
        default=DEFAULT_EMBEDDER,
        choices=["hashing", "voyage", "openai", "sentence-transformers"],
        help="embedding backend (default: hashing, dependency-free)",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="model name for the chosen embedder")
    parser.add_argument(
        "--embedding-cache-size",
        type=int,
        default=DEFAULT_EMBEDDING_CACHE_SIZE,
        help="LRU cache size for repeated embed() calls (helps paid embedders); 0 disables it",
    )
    parser.add_argument(
        "--vector-index",
        action="store_true",
        default=DEFAULT_VECTOR_INDEX,
        help="use a sqlite-vec ANN index for recall() (requires the 'vector-index' extra)",
    )
    parser.add_argument(
        "--vector-index-overfetch",
        type=int,
        default=DEFAULT_VECTOR_INDEX_OVERFETCH,
        help="candidate pool size fetched from the index before decay re-ranking",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="emem", description="A local-first memory engine.")
    _add_common_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="store a new memory")
    p_add.add_argument("text")
    p_add.add_argument("--tags", default="")
    p_add.add_argument("--source", default=None)
    p_add.add_argument("--importance", type=float, default=1.0)
    p_add.set_defaults(func=_add)

    p_update = sub.add_parser("update", help="edit an existing memory")
    p_update.add_argument("id", type=int)
    p_update.add_argument("--content", default=None)
    p_update.add_argument("--tags", default=None, help="replaces all tags")
    p_update.add_argument("--source", default=None)
    p_update.add_argument("--importance", type=float, default=None)
    p_update.set_defaults(func=_update)

    p_delete = sub.add_parser("delete", help="delete a memory")
    p_delete.add_argument("id", type=int)
    p_delete.set_defaults(func=_delete)

    p_list = sub.add_parser("list", help="browse memories (most recent first)")
    p_list.add_argument("--status", default=None, choices=["active", "compacted", "all"])
    p_list.add_argument("--tag", default=None)
    p_list.add_argument("--limit", type=int, default=50)
    p_list.add_argument("--offset", type=int, default=0)
    p_list.set_defaults(func=_list)

    p_tags = sub.add_parser("tags", help="list distinct tags with counts")
    p_tags.set_defaults(func=_tags)

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
    p_compact.add_argument(
        "--summarizer",
        default="none",
        choices=["none", "claude"],
        help="'claude' asks Claude to merge each cluster into a real summary (requires the 'llm' extra); "
        "'none' (default) just concatenates the cluster's contents",
    )
    p_compact.add_argument("--summarizer-model", default=None, help="model name when --summarizer=claude")
    p_compact.set_defaults(func=_compact)

    p_export = sub.add_parser("export", help="dump all memories and links as JSON")
    p_export.add_argument("--out", default=None, help="write to this file instead of stdout")
    p_export.set_defaults(func=_export)

    p_import = sub.add_parser("import", help="load memories and links from an export_data() JSON dump")
    p_import.add_argument("file")
    p_import.add_argument("--reset", action="store_true", help="wipe existing data first")
    p_import.set_defaults(func=_import)

    p_reindex = sub.add_parser("reindex", help="re-embed every memory with the current --embedder")
    p_reindex.set_defaults(func=_reindex)

    p_stats = sub.add_parser("stats", help="show store statistics")
    p_stats.set_defaults(func=_stats)

    p_serve = sub.add_parser("serve", help="run the web UI + REST API (requires the 'web' extra)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument(
        "--api-key",
        action="append",
        default=None,
        help="require this key on the 'X-API-Key' header; repeat for multiple independently "
        "revocable keys (e.g. one per agent/integration)",
    )
    p_serve.add_argument("--no-auth", action="store_true", help="disable API key auth (local/dev use only)")
    p_serve.add_argument(
        "--rate-limit",
        type=int,
        default=DEFAULT_RATE_LIMIT,
        help="max requests per --rate-limit-window per API key/IP on /api/*; 0 disables it",
    )
    p_serve.add_argument(
        "--rate-limit-window", type=float, default=DEFAULT_RATE_LIMIT_WINDOW, help="rate limit window, in seconds"
    )
    p_serve.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
