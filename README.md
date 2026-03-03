# Querdex

**Reasoning-first document intelligence system.**

Querdex indexes any document into a hierarchical tree, then uses a two-tier LLM search to answer questions with cited sources. It works without an LLM (keyword heuristics), and optionally plugs in Anthropic or OpenAI for higher-quality results.

---

## Table of Contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick Start (CLI)](#quick-start-cli)
- [LLM Setup](#llm-setup)
- [CLI Reference](#cli-reference)
- [Python API](#python-api)
- [Supported File Types](#supported-file-types)
- [Environment Variables](#environment-variables)
- [Development](#development)
- [Publishing](#publishing)

---

## How it works

```
Document
   │
   ▼
Ingestion ──► parse into pages/sections (Section[])
   │
   ▼
Indexing ───► build hierarchical tree (TreeNode) + entity map + knowledge graph
   │
   ▼
Storage ────► persist to SQLite (sections, tree, entities, graph, query cache)
   │
   ▼
Query
  ├─ Tier 1: LLM (or keyword) batch-prune of tree nodes
  ├─ Tier 2: LLM (or heuristic) per-node relevance scoring
  ├─ Retrieval: pull section text for selected nodes
  └─ Answer: LLM synthesizes answer with source citations
   │
   ▼
Adaptive ───► update node summaries based on query feedback (runs in background)
```

Three query routes are selected automatically:
- **single_doc** — standard hierarchical search on one document
- **multi_doc** — virtual super-tree across up to 3 documents
- **graph** — entity-seeded graph walk for relationship queries ("how does X relate to Y?")

---

## Installation

**Base install** (no LLM, uses keyword heuristics):
```bash
pip install querdex
```

**With Anthropic (Claude):**
```bash
pip install querdex[anthropic]
```

**With OpenAI (GPT):**
```bash
pip install querdex[openai]
```

**Development:**
```bash
git clone <repo>
cd querdex
uv sync --extra dev
# or with an LLM provider:
uv sync --extra dev --extra anthropic
uv sync --extra dev --extra openai
```

**Requirements:** Python 3.11+

---

## Quick Start (CLI)

### 1. Index a document

```bash
querdex index ./report.pdf --doc-id annual-report
```

Output:
```
Indexed doc_id=annual-report version=1
Nodes=12 max_depth=3
```

### 2. Query it

```bash
querdex query --doc-id annual-report --query "What was the Q3 revenue?"
```

Output:
```
Query ID: 3f8a1c...
Intent: single_doc | Cache hit: False
Q3 revenue was $1.2B, up 8% year-over-year (Revenue Analysis, pages 4-6).
```

### 3. Multi-turn conversation (session)

```bash
# First turn
querdex query --doc-id annual-report \
  --query "What were the risk factors?" \
  --session-id session_001

# Second turn — context from first turn is carried over
querdex query --doc-id annual-report \
  --query "Which of those risks materialised?" \
  --session-id session_001
```

### 4. Re-index an updated document

When the document changes, Querdex only rebuilds the affected parts:
```bash
querdex index ./report_v2.pdf --doc-id annual-report
```

### 5. Delete a document

```bash
querdex delete --doc-id annual-report
```

### Custom database path

By default the database is stored at `./index_store/querdex.db`. To change it:
```bash
querdex --db /path/to/my.db index ./report.pdf --doc-id demo
querdex --db /path/to/my.db query --doc-id demo --query "summary?"
```

---

## LLM Setup

Without any LLM configured, Querdex falls back to keyword/heuristic matching — it always produces an answer, just less precise.

### Anthropic (Claude)

```bash
export QUERDEX_LLM_PROVIDER=anthropic
export QUERDEX_LLM_API_KEY=sk-ant-...

# Optional: override model defaults
export QUERDEX_LLM_TIER1_MODEL=claude-haiku-4-5-20251001   # fast, cheap (batch prune)
export QUERDEX_LLM_TIER2_MODEL=claude-sonnet-4-6            # powerful (deep reasoning + answers)
```

### OpenAI (GPT)

```bash
export QUERDEX_LLM_PROVIDER=openai
export QUERDEX_LLM_API_KEY=sk-...

# Optional: override model defaults
export QUERDEX_LLM_TIER1_MODEL=gpt-4o-mini   # fast, cheap
export QUERDEX_LLM_TIER2_MODEL=gpt-4o         # powerful
```

**How the two tiers are used:**

| Tier | Model | Purpose |
|------|-------|---------|
| Tier 1 | cheap/fast | Single batched call to prune all tree nodes to the relevant few |
| Tier 2 | powerful | Per-node deep reasoning to confirm relevance + score confidence |
| Answer | powerful | Synthesise a cited answer from the retrieved section text |

---

## CLI Reference

```
querdex [--db PATH] <command> [options]
```

| Command | Description |
|---------|-------------|
| `index <file>` | Index a document. Auto-detects format from extension. |
| `query` | Query an indexed document. |
| `delete` | Remove a document and all its data from the store. |

### `index`

```
querdex index <file_path> [--doc-id ID]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `file_path` | required | Path to the document to index |
| `--doc-id` | auto-generated from filename+hash | Stable identifier for this document |

### `query`

```
querdex query --doc-id ID --query TEXT [--session-id ID]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--doc-id` | required | Document to query |
| `--query` | required | Natural language question |
| `--session-id` | none | Enables multi-turn context (pass same ID across turns) |

### `delete`

```
querdex delete --doc-id ID
```

---

## Python API

For integration into your own application:

```python
import asyncio
from querdex.services import build_engine

# build_engine reads QUERDEX_LLM_* env vars automatically
engine = build_engine("./index_store/querdex.db")

# Index a document
doc = asyncio.run(engine.index_document("./report.pdf", doc_id="annual-report"))
print(f"Indexed: {doc.doc_id} | nodes={doc.stats.total_nodes}")

# Query
result = engine.query_document("annual-report", "What was Q3 revenue?")
print(result.answer)
print(f"Confidence: {result.confidence:.0%}")
for source in result.source_nodes:
    print(f"  Source: {source.title}, pages {source.pages}")

# Multi-turn query
result2 = engine.query_document(
    "annual-report",
    "What caused that increase?",
    session_id="my-session-001",
)

# Re-index after the document changes
doc_v2 = asyncio.run(engine.reindex_document("./report_v2.pdf", doc_id="annual-report"))

# Delete
engine.store.delete_document("annual-report")

# Always close when done
engine.store.close()
```

### Passing an LLM client directly

```python
from querdex.llm.anthropic_client import AnthropicLLMClient
from querdex.services.engine import QuerdexEngine
from querdex.storage import SQLiteStore

llm = AnthropicLLMClient(
    api_key="sk-ant-...",
    tier1_model="claude-haiku-4-5-20251001",
    tier2_model="claude-sonnet-4-6",
)
store = SQLiteStore("./querdex.db")
engine = QuerdexEngine(store, llm_client=llm)
```

### Using the FakeLLMClient in tests

```python
from querdex.llm.fake_client import FakeLLMClient
from querdex.query.answering import AnswerGenerator

fake = FakeLLMClient(
    default='{"answer": "Revenue was $1.2B.", "confidence": 0.9}'
)
gen = AnswerGenerator(llm_client=fake)
answer, confidence, sources = gen.generate("What was revenue?", chunks)
```

---

## Supported File Types

| Extension | Parser | Notes |
|-----------|--------|-------|
| `.txt` | TextParser | Plain text, split by paragraphs |
| `.md`, `.markdown` | MarkdownParser | Heading-aware section splitting |
| `.html`, `.htm` | HTMLParser | Strips tags, extracts text blocks |
| `.docx` | DOCXParser | Microsoft Word, paragraph-level |
| `.pdf` | PDFParser | Page-level; OCR optional (see below) |
| `.py` | PythonCodeParser | Function/class level chunking |
| `.js`, `.ts`, `.jsx`, `.tsx` | JSCodeParser | Function-level chunking |
| `.csv` | CSVParser | Row-batched sections |
| `.db`, `.sqlite` | SQLiteParser | Table-level sections |
| `.mp3`, `.wav`, `.m4a`, `.mp4`, `.mov` | AudioVideoParser | Transcript-based (requires Whisper or similar) |
| `.url` | URLParser | Fetches and parses the web page at that URL |
| URL string | URLParser | Pass a URL string directly as the file path |

### PDF OCR

For scanned PDFs, enable OCR via environment variables:

```bash
# Tesseract (local)
export QUERDEX_OCR_ENABLED=true
export QUERDEX_OCR_PROVIDER=tesseract         # default when OCR enabled
export QUERDEX_TESSERACT_CMD=tesseract        # path to tesseract binary

# Cloud OCR (custom endpoint)
export QUERDEX_OCR_ENABLED=true
export QUERDEX_OCR_PROVIDER=cloud
export QUERDEX_OCR_ENDPOINT=https://your-ocr-api.com/v1/ocr
export QUERDEX_OCR_API_KEY=your-key
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QUERDEX_LLM_PROVIDER` | _(none)_ | `anthropic` or `openai`. If unset, heuristic mode is used. |
| `QUERDEX_LLM_API_KEY` | _(none)_ | API key for the selected provider |
| `QUERDEX_LLM_TIER1_MODEL` | `claude-haiku-4-5-20251001` / `gpt-4o-mini` | Fast model for batch node pruning |
| `QUERDEX_LLM_TIER2_MODEL` | `claude-sonnet-4-6` / `gpt-4o` | Powerful model for deep reasoning and answers |
| `QUERDEX_OCR_ENABLED` | `false` | Enable OCR for scanned PDFs |
| `QUERDEX_OCR_PROVIDER` | `tesseract` | `tesseract` or `cloud` |
| `QUERDEX_TESSERACT_CMD` | `tesseract` | Path to Tesseract binary |
| `QUERDEX_OCR_ENDPOINT` | _(none)_ | Endpoint URL for cloud OCR provider |
| `QUERDEX_OCR_API_KEY` | _(none)_ | API key for cloud OCR provider |

---

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run tests with coverage output
uv run pytest -v

# Lint
uv run ruff check .

# Type check
uv run mypy src

# Run release gate (integration smoke test)
uv run python scripts/run_release_gate.py \
  --db ./index_store/release_gate.db \
  --doc-path ./tests/fixtures/golden/sample.md \
  --doc-id demo_release
```

### Project structure

```
src/querdex/
├── ingestion/          # File parsers → Section[]
│   └── parsers/        # One file per format
├── indexing/           # Tree builder, entity extractor, graph builder
├── query/              # Analyzer, router, tiered search, answer generator
├── adaptive/           # Feedback-driven node summary updater
├── storage/            # SQLite store, NetworkX graph store
├── llm/                # LLMClient protocol, Anthropic/OpenAI/Fake adapters
├── services/           # QuerdexEngine (main facade), build_engine()
├── schemas/            # Pydantic models (Section, TreeNode, QueryResult, …)
├── utils/              # Tree traversal, LLM JSON validation helpers
├── ops/                # Retry, structured logger, health checker
├── evaluation/         # Evaluation harness and metrics
└── cli.py              # CLI entry point
```

### Adding a new file format

1. Create `src/querdex/ingestion/parsers/my_parser.py` implementing the `Parser` protocol:
   ```python
   from querdex.ingestion.base import Parser
   from querdex.schemas import Section

   class MyParser(Parser):
       def parse(self, path: Path, doc_id: str) -> list[Section]:
           ...
   ```
2. Register it in `IngestionOrchestrator.__init__()` in [ingestion/orchestrator.py](src/querdex/ingestion/orchestrator.py):
   ```python
   ".myext": MyParser(),
   ```

---

## Publishing

```bash
# Install build tools
uv add --dev build twine

# Build wheel + source distribution
uv run python -m build

# Upload to PyPI
uv run twine upload dist/*

# Or test on TestPyPI first
uv run twine upload --repository testpypi dist/*
```

Users then install with:
```bash
pip install querdex                 # heuristic mode
pip install querdex[anthropic]      # with Claude
pip install querdex[openai]         # with GPT
```

---

## License

MIT
