from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from querdex.services import build_engine


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="querdex")
    parser.add_argument("--db", default="./index_store/querdex.db", help="SQLite database path")

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index a document")
    index_parser.add_argument("file_path", help="Path to input document")
    index_parser.add_argument("--doc-id", default=None, help="Optional explicit document id")

    query_parser = subparsers.add_parser("query", help="Query an indexed document")
    query_parser.add_argument("--doc-id", required=True, help="Document id to query")
    query_parser.add_argument("--query", required=True, help="User query")
    query_parser.add_argument("--session-id", default=None, help="Session id for multi-turn context")

    delete_parser = subparsers.add_parser("delete", help="Delete an indexed document")
    delete_parser.add_argument("--doc-id", required=True, help="Document id to delete")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = build_engine(db_path)
    try:
        if args.command == "index":
            document_index = asyncio.run(engine.index_document(args.file_path, args.doc_id))
            print(f"Indexed doc_id={document_index.doc_id} version={document_index.version}")
            print(f"Nodes={document_index.stats.total_nodes} max_depth={document_index.stats.max_depth}")

        elif args.command == "query":
            result = engine.query_document(args.doc_id, args.query, session_id=args.session_id)
            print(f"Query ID: {result.query_id}")
            print(f"Intent: {result.intent_type} | Cache hit: {result.cache_hit}")
            print(result.answer)

        elif args.command == "delete":
            engine.store.delete_document(args.doc_id)
            print(f"Deleted doc_id={args.doc_id}")

    finally:
        engine.store.close()


if __name__ == "__main__":
    main()
