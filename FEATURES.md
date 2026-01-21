# Feature Reference

Complete documentation of all RAG System features.

## Table of Contents

- [Query Modes](#query-modes)
- [Document Ingestion](#document-ingestion)
- [Search & Retrieval](#search--retrieval)
- [Generation Features](#generation-features)
- [Quality & Evaluation](#quality--evaluation)
- [Caching System](#caching-system)
- [Configuration Reference](#configuration-reference)

---

## Query Modes

### Default Mode
Standard balanced query with hybrid search and LLM generation.

```bash
./query.sh "What are the main points in my documents?"
```

- **Duration**: ~60-90 seconds
- **Features**: Hybrid search, basic context
- **Best for**: General queries

### RAG-Only Mode
Retrieval without LLM generation. Returns relevant chunks directly.

```bash
./query.sh --rag-only "project deadlines"
```

- **Duration**: <1 second
- **Features**: Pure vector search
- **Best for**: Finding specific passages, debugging

### Web-Only Mode
Bypass local documents, search the web via SearXNG.

```bash
./query.sh --web-only "Latest AI news 2024"
```

- **Duration**: ~30 seconds
- **Features**: Web search + LLM synthesis
- **Best for**: Current events, external information

### Ultrafast Mode
Minimal features for quick responses.

```bash
./query.sh --ultrafast "simple question"
```

- **Duration**: ~30-45 seconds
- **Features**: Reduced context, shorter response
- **Best for**: Simple factual queries

### Full Mode
All features enabled including multi-pass, citations, and CRAG.

```bash
./query.sh --full "Analyze the contract terms comprehensively"
```

- **Duration**: ~3-5 minutes
- **Features**: Multi-pass retrieval, reranking, citations, CRAG
- **Best for**: Complex analysis, important queries

### Tiered Performance Modes

```bash
./query.sh --mode quick "fast answer needed"
./query.sh --mode default "normal query"
./query.sh --mode deep "thorough research"
```

| Mode | Timeout | Context | Response |
|------|---------|---------|----------|
| Quick | 90s | 3000 chars | 150 tokens |
| Default | 180s | 8000 chars | 800 tokens |
| Deep | 600s | 15000 chars | 2000 tokens |

---

## Document Ingestion

### Supported Formats

| Category | Formats |
|----------|---------|
| Documents | PDF, DOCX, DOC, ODT, RTF, TXT |
| Spreadsheets | XLSX, XLS, CSV, TSV |
| Presentations | PPTX, PPT, ODP |
| Web | HTML, MHTML, XML |
| Data | JSON, YAML |
| Code | MD, RST, PS1, SH, PY, JS |
| Images | PNG, JPG, TIFF (with OCR) |

### Basic Ingestion

```bash
# Single file
./ingest.sh document.pdf

# Directory
./ingest.sh ./documents/

# Force re-ingestion
./ingest.sh --force ./documents/

# Show supported formats
./ingest.sh --formats
```

### Website Ingestion

```bash
# Basic website crawl
./ingest.sh --url https://example.com

# With depth control
./ingest.sh --url https://docs.example.com --max-depth 3

# Stay within domain
./ingest.sh --url https://example.com --same-domain
```

Configuration:
```bash
WEB_CRAWLER_MAX_PAGES=50
WEB_CRAWLER_MAX_DEPTH=3
WEB_CRAWLER_DELAY=1.0
```

### CSV Dual Mode

CSV files are processed in two ways:
1. **Structured**: Column headers + values for precise queries
2. **Natural Language**: Transformed to readable sentences

```bash
# Automatic for CSV/XLSX files
./ingest.sh data.csv
```

Example transformation:
```
# Original CSV:
Name,Role,Department
Alice,Engineer,R&D
Bob,Manager,Sales

# Natural Language:
"Alice works as Engineer in the R&D department."
"Bob works as Manager in the Sales department."
```

### OCR Configuration

```bash
# French + English OCR
UNSTRUCTURED_OCR_LANGUAGES=eng+fra

# Requires:
apt-get install tesseract-ocr tesseract-ocr-fra
```

---

## Search & Retrieval

### Hybrid Search

Combines dense (semantic) and sparse (keyword) vectors using Reciprocal Rank Fusion.

```bash
HYBRID_SEARCH_MODE=native
SPARSE_EMBED_MODEL=prithivida/Splade_PP_en_v1
HYBRID_RRF_K=60
```

Dense vectors capture meaning; sparse vectors capture exact keywords. RRF merges both rankings.

### HyDE (Hypothetical Document Embeddings)

Generates a hypothetical answer first, then searches for similar content.

```bash
HYDE_ENABLED=true
```

Useful for abstract queries where the question doesn't contain domain terms.

### Multi-Pass Retrieval

Multiple search iterations with query variants:
- Original query
- Expanded query
- Rewritten query

```bash
MULTIPASS_ENABLED=true
MULTIPASS_VARIANTS=3
```

### StepBack Query

Generates a broader version of the query for better context retrieval.

```bash
STEPBACK_ENABLED=true
```

Example:
- Query: "What was the Q3 revenue?"
- StepBack: "What financial metrics were reported?"

### Subquery Decomposition

Breaks complex queries into simpler sub-questions.

```bash
SUBQUERY_ENABLED=true
```

### FlashRank Reranking

Neural reranking of retrieved chunks for relevance.

```bash
RERANK_ENABLED=true
RERANK_MODEL=ms-marco-MiniLM-L-12-v2
RERANK_TOP_K=5
```

---

## Generation Features

### Map/Reduce Summarization

Summarize documents of any length by:
1. **Map**: Summarize each chunk
2. **Reduce**: Combine summaries iteratively

```bash
./query.sh --summarize document.pdf
./query.sh --summarize document.pdf "Focus on financial aspects"
```

Configuration:
```bash
MAPREDUCE_ENABLED=true
MAPREDUCE_CHUNK_SIZE=4000
MAPREDUCE_BATCH_SIZE=3
MAPREDUCE_CHUNK_TIMEOUT=120
```

### Extraction Mode

Extract structured data from documents.

```bash
./query.sh --extract "List all people and their roles" document.pdf
./query.sh --extract "Extract all dates and events" report.pdf
./query.sh --extract "Find all product names and prices" catalog.pdf
```

Output format:
```json
{
  "extractions": [
    {"name": "Alice", "role": "CEO"},
    {"name": "Bob", "role": "CTO"}
  ],
  "count": 2,
  "source": "document.pdf"
}
```

Configuration:
```bash
EXTRACTION_ENABLED=true
EXTRACTION_CHUNK_SIZE=3000
EXTRACTION_DEDUP_THRESHOLD=0.85
```

### Self-Reflection

Verify answers for high-stakes queries:
1. Generate initial answer
2. Verify against context
3. Correct if inconsistencies found

```bash
REFLECTION_ENABLED=true
REFLECTION_CONFIDENCE_THRESHOLD=0.7
REFLECTION_MAX_RETRIES=1
REFLECTION_ALWAYS=false

# Triggers for verification:
REFLECTION_KEYWORDS=legal,contract,medical,financial,compliance
```

### CRAG (Corrective RAG)

Falls back to web search when local context is insufficient.

```bash
CRAG_ENABLED=true
CRAG_THRESHOLD=0.4
```

Flow:
1. Retrieve local chunks
2. Score relevance
3. If below threshold → web search
4. Combine results

### Citations

Include source references in responses.

```bash
./query.sh --full --citations "What does the contract say?"
```

Output includes:
```
Based on the documents [1][2], the contract states...

Sources:
[1] contract.pdf (page 3)
[2] amendment.docx (section 2)
```

---

## Quality & Evaluation

### RAGAS Evaluation

Built-in quality metrics:
- **Faithfulness**: Is the answer grounded in context?
- **Answer Relevancy**: Does it address the question?
- **Context Precision**: Are retrieved chunks relevant?

```bash
# Evaluate single query
./evaluate.sh "What is the main topic?" "Expected answer text"

# Batch evaluation
./evaluate.sh --batch questions.json

# Generate test questions
./evaluate.sh --generate 10
```

Configuration:
```bash
RAGAS_ENABLED=true
RAGAS_SLA_THRESHOLD=0.80
```

### Quality Ledger

Tracks query history and quality scores.

```bash
QUALITY_LEDGER_ENABLED=true
```

### Abstention

System declines to answer when confidence is low.

```bash
ABSTENTION_ENABLED=true
```

---

## Caching System

### Layer 1: Qdrant Search Cache

Caches vector search results.

```bash
QDRANT_CACHE_ENABLED=true
QDRANT_CACHE_DIR=./cache/qdrant
QDRANT_CACHE_TTL=3600  # 1 hour
```

### Layer 2: Response Cache

Caches full LLM responses.

```bash
RESPONSE_CACHE_ENABLED=true
RESPONSE_CACHE_DIR=./cache/responses
RESPONSE_CACHE_TTL=86400  # 24 hours
```

### Cache Management

```bash
# View cache statistics
./cache-stats.sh

# Clear volatile caches (keep models)
./clear-cache.sh

# Clear everything
rm -rf ./cache && mkdir -p ./cache
```

### Conversation Memory

Maintains context across queries in a session.

```bash
MEMORY_ENABLED=true
MEMORY_MAX_TURNS=5
MEMORY_FILE=./cache/memory.json
```

---

## Configuration Reference

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | http://localhost:11434 | Ollama server URL |
| `LLM_MODEL` | qwen2.5:3b | LLM model name |
| `TEMPERATURE` | 0.2 | Generation temperature |
| `QDRANT_HOST` | http://localhost:6333 | Qdrant server URL |
| `COLLECTION_NAME` | documents | Vector collection name |

### Embedding Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `FASTEMBED_MODEL` | BAAI/bge-base-en-v1.5 | Embedding model |
| `EMBEDDING_DIMENSION` | 768 | Vector dimension |
| `SPARSE_EMBED_MODEL` | Splade_PP_en_v1 | Sparse model |

### Retrieval Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 600 | Chunk size in chars |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `DEFAULT_TOP_K` | 6 | Number of results |
| `HYBRID_RRF_K` | 60 | RRF constant |

### Timeout Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_TIMEOUT_DEFAULT` | 180 | Default timeout (s) |
| `LLM_TIMEOUT_QUICK` | 90 | Quick mode timeout |
| `LLM_TIMEOUT_DEEP` | 600 | Deep mode timeout |

### Feature Toggles

| Variable | Default | Description |
|----------|---------|-------------|
| `HYDE_ENABLED` | false | Hypothetical documents |
| `RERANK_ENABLED` | false | Neural reranking |
| `CRAG_ENABLED` | false | Web fallback |
| `REFLECTION_ENABLED` | true | Answer verification |
| `CITATIONS_ENABLED` | false | Source citations |

---

## CLI Flags Reference

### ingest.sh

| Flag | Description |
|------|-------------|
| `--force` | Re-ingest all files |
| `--debug` | Verbose output |
| `--recreate` | Delete and recreate collection |
| `--url URL` | Ingest website |
| `--max-depth N` | Crawl depth limit |
| `--formats` | Show supported formats |
| `--help` | Show help |

### query.sh

| Flag | Description |
|------|-------------|
| `--rag-only` | Retrieval only |
| `--web-only` | Web search only |
| `--ultrafast` | Minimal features |
| `--full` | All features |
| `--mode MODE` | quick/default/deep |
| `--summarize FILE` | Map/reduce summary |
| `--extract PROMPT FILE` | Structured extraction |
| `--multipass` | Multi-pass retrieval |
| `--citations` | Include sources |
| `--no-memory` | Disable memory |
| `--no-cache` | Disable cache |
| `--debug` | Verbose output |
| `--help` | Show help |
