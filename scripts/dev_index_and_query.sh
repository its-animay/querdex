#!/usr/bin/env bash
set -euo pipefail

DB_PATH="${1:-./index_store/querdex.db}"
DOC_PATH="${2:-./README.md}"
DOC_ID="${3:-querdex_docs}"

uv run querdex --db "$DB_PATH" index "$DOC_PATH" --doc-id "$DOC_ID"
uv run querdex --db "$DB_PATH" query --doc-id "$DOC_ID" --query "What does this document say about tiered search?"
