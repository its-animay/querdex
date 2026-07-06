from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from querdex.extraction import ExtractionExample, ExtractionTask, render_extraction_html
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

    extract_parser = subparsers.add_parser("extract", help="Run structured extraction over an indexed document")
    extract_parser.add_argument("--doc-id", required=True, help="Document id to extract from")
    extract_parser.add_argument("--prompt", required=True, help="Description of what to extract")
    extract_parser.add_argument("--examples", default=None, help="Path to JSON file with few-shot examples")
    extract_parser.add_argument("--passes", type=int, default=1, help="Extraction passes (more passes, more recall)")
    extract_parser.add_argument("--html", default=None, help="Write an interactive HTML review page to this path")

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

        elif args.command == "extract":
            examples = []
            if args.examples:
                raw_examples = json.loads(Path(args.examples).read_text(encoding="utf-8"))
                examples = [ExtractionExample.model_validate(item) for item in raw_examples]
            task = ExtractionTask(description=args.prompt, examples=examples)
            run = engine.extract_document(args.doc_id, task, passes=args.passes)
            stats = run.stats
            print(
                f"Run {run.run_id}: {len(run.extractions)} extractions "
                f"(exact={stats.exact_count} fuzzy={stats.fuzzy_count} unaligned={stats.unaligned_count}) "
                f"in {stats.latency_ms}ms across {stats.chunk_count} chunks"
            )
            for extraction in run.extractions[:20]:
                location = (
                    f"p{extraction.page_number} {extraction.section_id}"
                    if extraction.section_id
                    else "ungrounded"
                )
                print(f"  [{extraction.extraction_class}] {extraction.extraction_text!r} ({location})")
            if len(run.extractions) > 20:
                print(f"  ... and {len(run.extractions) - 20} more")
            if args.html:
                sections = engine.store.sections_for_doc(args.doc_id)
                Path(args.html).write_text(render_extraction_html(run, sections), encoding="utf-8")
                print(f"HTML review written to {args.html}")

    finally:
        engine.store.close()


if __name__ == "__main__":
    main()
