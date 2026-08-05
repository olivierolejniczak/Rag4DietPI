# RAG System - Private Document Intelligence

A complete, self-hosted Retrieval-Augmented Generation (RAG) system designed for offline deployment on resource-constrained hardware. Query your private documents using local LLMs with advanced retrieval techniques.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Linux%20Debian-orange.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-green.svg)

## Why This Project?

Beyond the marketing that tends to blur understanding: **AI is a fantastic piece
of software — but it is still software.** Not every question needs an
ultra-fast answer served from a giant datacenter behind a paid subscription. A
great many everyday needs — *"what does this contract say about the deposit?"*,
*"summarize this report"*, *"find every document that mentions X"* — are perfectly
ordinary information-retrieval tasks that a modest local machine can handle on its
own, privately and for free.

So this project set out to answer a concrete question: **how far can you get with a
*local* AI running on an entry-level PC**, for simple, real-world jobs like a RAG?

> **What is a RAG?** *Retrieval-Augmented Generation.* Instead of asking a model to
> answer from its own memory (where it may invent things), you first **retrieve**
> the passages most relevant to the question from *your own* documents, then hand
> them to the model as context. The answer is **grounded in your data** and can
> cite it, rather than being made up. The retrieval half is where most of the
> quality lives — which is the whole point below.

It was also, honestly, a way to **learn**. Building this meant getting hands-on with
a lot of the technology around modern AI — embeddings, vector databases, hybrid
search, rerankers, model quantization, on-demand model serving — and, above all,
understanding that **the pre-processing of the source data is what decides whether
the whole thing is any good.** Chunking, parsing, OCR, how you structure and clean
the reference corpus: that groundwork matters far more to relevance and performance
than any single clever model. This repo is the result of chasing that lesson.

What it's designed for:

- **Complete Privacy**: 100% offline operation, no data leaves your machine
- **Low-Resource Deployment**: Runs on Raspberry Pi 4/5, mini-PCs, or any system with 4GB+ RAM
- **Production-Ready**: Battle-tested on DietPi/Debian systems
- **Advanced Retrieval**: Hybrid search, reranking, CRAG, and more

### The test machine

Everything here is developed and benchmarked on a deliberately **modest,
second-hand office PC** — not a workstation, not a GPU rig:

| Component | Spec |
|-----------|------|
| Machine | Dell OptiPlex 3070 (refurbished small-form-factor desktop) |
| CPU | Intel Core i5-9500T — 6 cores / 6 threads, 2.2 GHz (3.7 GHz boost), **35 W** low-power |
| RAM | 16 GB |
| Storage | 234 GB NVMe SSD |
| GPU | **None** — everything runs on CPU |
| OS | Debian 13 (trixie) / DietPi, headless |
| Swap | 8 GB file (safety net for peaks) |

If it runs comfortably here, it runs on most things. The numbers quoted throughout
this README come from this exact box.

## Why DietPi?

This project was developed and tested primarily on [DietPi](https://dietpi.com/), a highly optimized Debian-based OS for single-board computers (SBCs). DietPi was chosen because:

| Reason | Benefit |
|--------|---------|
| **Minimal footprint** | ~400MB base install vs 2GB+ for standard Debian |
| **Optimized for SBCs** | Pre-configured for Raspberry Pi, Odroid, etc. |
| **Software catalog** | Easy Docker installation via `dietpi-software` |
| **RAM efficiency** | Critical when running LLMs + vector DB on 4-8GB |
| **Headless optimized** | Perfect for server deployments |

### Compatibility with Other Systems

**This project works on any Debian-based Linux distribution:**

- ✅ Debian 11/12 (Bullseye/Bookworm)
- ✅ Ubuntu 22.04/24.04 LTS
- ✅ Raspberry Pi OS (64-bit)
- ✅ DietPi (all supported devices)
- ✅ Linux Mint, Pop!_OS, etc.

For non-DietPi systems, simply:
1. Install Docker manually: `curl -fsSL https://get.docker.com | sh`
2. Adjust data directories in the scripts (default: `/mnt/dietpi_userdata/`)

## Features

### Document Processing
- **20+ File Formats**: PDF, DOCX, XLSX, PPTX, HTML, Markdown, CSV, JSON, XML, and more
- **Smart Chunking**: Semantic-aware document splitting
- **OCR Support**: Tesseract with French + English language packs
- **Legacy Formats**: Microsoft Word 97-2003 (.doc) via antiword
- **CSV Dual Mode**: Structured + natural language representations
- **Web Crawling**: Ingest entire websites with depth control

### Retrieval & Search
- **Hybrid Search**: Dense (semantic) + Sparse (BM25/Splade) vectors
- **Native RRF Fusion**: Reciprocal Rank Fusion in Qdrant
- **HyDE**: Hypothetical Document Embeddings
- **Multi-Pass Retrieval**: Query expansion and rewriting
- **FlashRank Reranking**: Fast neural reranking

### Generation & Quality
- **Map/Reduce Summarization**: Handle documents of any length
- **Extraction Mode**: Structured data extraction from documents
- **Self-Reflection**: Answer verification for high-stakes queries
- **CRAG**: Corrective RAG with web search fallback
- **Citations**: Source attribution in responses
- **RAGAS Evaluation**: Built-in quality metrics

### Optimization
- **CPU Score Profiling**: Automatic hardware detection and tuning
- **Adaptive Models**: Embedding model selection based on RAM
- **Qdrant Low-Memory Mode**: Disk-based storage for systems <8GB RAM
- **Tiered Performance**: Quick/Default/Deep query modes
- **Query Caching**: Query-level response cache with TTL

## Quick Start

### Prerequisites

- Linux (Debian/Ubuntu/DietPi)
- 4GB+ RAM (8GB+ recommended)
- 20GB+ free disk space
- Docker installed

### Installation

```bash
# Clone the repository
git clone https://github.com/olivierolejniczak/Rag4DietPI.git
cd Rag4DietPI

# Run setup scripts (as root or with sudo)
sudo bash setup-rag-core.sh          # base deps (Docker, Qdrant, SearXNG)
sudo bash setup-rag-llm-backend.sh   # migrate LLM backend to llama-swap + llama.cpp
sudo bash setup-rag-ingest.sh
sudo bash setup-rag-query.sh

# Verify installation
./status.sh
```

### Basic Usage

```bash
# Ingest documents
./ingest.sh ./documents/

# Query your documents
./query.sh "What are the main topics in my documents?"

# Web-only search (no local documents)
./query.sh --web-only "Latest news about AI"

# RAG-only (retrieval without LLM)
./query.sh --rag-only "project deadlines"

# Full mode (all features enabled)
./query.sh --full "Summarize the contract terms"
```

### Advanced Features

```bash
# Summarize a long document (map/reduce, its own wrapper)
./summarize.sh ./documents/report.pdf
./summarize.sh ./documents/report.pdf focus on financial data

# Extract structured data (its own wrapper)
./extract.sh ./documents/team.pdf "list all people and their roles"

# Ingest a website (crawl depth/pages come from config.env)
./ingest.sh --url https://example.com

# Tiered performance (fastest → deepest)
./query.sh --ultrafast "simple question"     # 1.5B tier, single pass, ~25s
./query.sh "normal query"                     # adaptive cascade (default)
./query.sh --full "complex analysis"          # 4B tier + CRAG, ~3-5min
```

### Browsing sources and folders

Documents are stored one collection per top-level folder. List them, or browse
the sub-folder tree of a source (deterministic, no LLM) — useful for facts that
live in the directory structure rather than in document text:

```bash
# List available --source values
./query.sh --list-sources

# List a source's sub-folders with file counts (pair with --source)
./query.sh --source contracts --list-folders

# Cap the depth (e.g. 3 path segments) for a summary view
./query.sh --source contracts --list-folders 3
```

## Command reference

### `ingest.sh` — building the index

```
./ingest.sh [options] [path]
```

| Option | Effect |
|--------|--------|
| *(no path)* | Ingest the default `DOCUMENTS_DIR` (`config.env`). |
| `path` | Ingest a single file **or** a directory (recursed). |
| `--url URL` | **Web ingest** — crawl a website and index its pages instead of local files. Crawl scope comes from `config.env`: `WEB_CRAWLER_MAX_PAGES` (default 50), `WEB_CRAWLER_MAX_DEPTH` (3), `WEB_CRAWLER_DELAY` (1.0s, polite rate-limit). |
| `--force` | Re-process **every** file, ignoring the "already ingested" dedup check. Use after changing chunking/parsing settings. |
| `--recreate` | **Delete and recreate** the target collection before ingesting (clean rebuild). |
| `--formats` | Print the full list of supported input formats and exit. |
| `--debug` | Verbose parsing/chunking output. |

**One collection per source folder.** Each top-level folder under `DOCUMENTS_DIR`
becomes its own Qdrant collection, named `documents__<slug>` (e.g.
`documents__maisons`, `documents__contracts`). This keeps sources isolated so a
query can be scoped to one of them with `query.sh --source <name>` — and so
rebuilding one source never touches the others.

**What gets parsed (20+ formats).** PDF, DOCX, legacy `.doc` (via antiword),
PPTX, XLSX, RTF, ODT; plain `txt/md/html`; structured `csv/tsv/json/xml`; email
`eml/msg`; and image OCR (Tesseract, **French + English** packs). CSVs are indexed
in **dual mode** — the raw structured rows *and* a natural-language rendering of
each row (`CSV_NL_LANG=fr`) — so tabular data is retrievable by meaning, not just
exact cell matches. Chunking defaults: `CHUNK_SIZE=700`, `CHUNK_OVERLAP=50`.

Every chunk is embedded **twice**: a dense vector (FastEmbed `bge-base-en-v1.5`,
768-d) for semantic similarity and a sparse SPLADE vector
(`prithivida/Splade_PP_en`) for lexical/keyword matching. Both live in the same
collection so hybrid search can fuse them at query time.

### `query.sh` — asking questions

```
./query.sh [mode] [options] 'your question'
```

By default `query.sh` runs an **adaptive cascade** (multi-level, multi-model): it
starts cheap and only escalates if the retrieved evidence isn't good enough,
swapping up to a bigger model tier as it climbs. This is the "multi-level, multi
model" core — you pay for a big model only when a question actually needs it.

**Modes (pick the entry point):**

| Mode | What it does | Rough time |
|------|--------------|-----------|
| *(default)* | Adaptive cascade: cache → RAG → multipass → web → full, gated by retrieval relevance. | varies |
| `--rag-only` | Retrieval only, **no LLM** — returns the matching chunks. | ~3s |
| `--ultrafast` | Single pass, minimal features, 1.5B tier. | ~25s |
| `--full` | All features + CRAG web fallback, deepest (4B) tier. | ~3–5min |
| `--web-only` | Bypass local documents, answer from web search. | ~30s |

**Options (tune the run):**

| Option | Effect |
|--------|--------|
| `--max-tier N` | Cap cascade escalation: `0`=RAG only, `1`=+multipass, `2`=+web, `3`=full. Bounds worst-case latency. |
| `--no-adaptive` | Disable the cascade — single pass in the current mode. |
| `--multipass` | Force multi-pass retrieval (query rewriting/expansion). |
| `--citations` | Append a **Sources** list (source attribution) to the answer. |
| `--source NAME` | Restrict retrieval to one top-level documents folder (its `documents__<slug>` collection). |
| `--list-sources` | List available `--source` names and exit. |
| `--list-folders [DEPTH]` | List a source's sub-folder tree with file counts (deterministic, no LLM); pair with `--source`. |
| `--no-memory` | Ignore conversation memory for this query. |
| `--no-cache` / `--clear-cache` | Skip the query cache / wipe it. |
| `--whitelist-add TERM`, `--whitelist-show`, `--whitelist-auto` | Manage the spellcheck whitelist (protect proper nouns from "correction"). |
| `--debug` | Show the full pipeline trace (retrieval scores, tier decisions, timings). |

**The multi-model tiers** are served on demand by llama-swap (see
[Architecture](#architecture)); at most one is resident at a time:

| Entry point | llama-swap model | GGUF |
|-------------|------------------|------|
| `--ultrafast` | `rag-quick` | qwen2.5-1.5b-instruct |
| default cascade | `rag-default` | qwen2.5-3b-instruct |
| `--full` (or top of cascade) | `rag-deep` | Qwen3-4B-Instruct-2507 |

### Companion wrappers

```bash
./summarize.sh <document> [focus request]   # map/reduce summary of a long doc
./extract.sh   <document> "what to extract" # structured extraction
./status.sh                                  # health of Qdrant / llama-swap / SearXNG
./monitor.sh                                 # live dashboard
./cache-stats.sh                             # query-cache hit stats
```

## Performance (measured on the test machine)

All figures are on the [OptiPlex 3070 / i5-9500T / 16 GB / CPU-only](#the-test-machine)
box described above — no GPU. Treat them as order-of-magnitude, not guarantees.

| Operation | Throughput / latency | Notes |
|-----------|----------------------|-------|
| **Ingestion** | **~1.9 chunks/second** | End-to-end (parse + dense + SPLADE embed + upsert). **SPLADE sparse embedding dominates** the time; dense bge-base is cheap by comparison. |
| Ingesting ~5,600 chunks | ~45–50 min | e.g. the live `documents__maisons` collection holds **5,589** points. |
| **Retrieval only** (`--rag-only`) | **~3 s / query** | Hybrid dense+sparse search + Qdrant RRF fusion over the collection. |
| **End-to-end answer** (`--ultrafast`, 1.5B) | **~25 s / query** | Includes generation; produces a correct, grounded answer (French corpus). |
| End-to-end answer (`--full`, 4B + CRAG) | minutes | Deepest tier, optional web fallback. |
| Reranking (if `RERANK_ENABLED=true`) | **+~6 s / query** | ~1000× the raw retrieval cost, and it *lowered* accuracy in benchmarks — hence off by default (see below). |

The practical takeaway: **retrieval is fast and cheap; the LLM is the slow part**,
which is exactly why the cascade tries to answer at the smallest tier that works
and only escalates when it must.

## Try it on a public dataset

You don't need private documents to exercise the system. Here's an end-to-end
walkthrough on a **public documentary corpus** — the Project Gutenberg plain-text
of a few classic books — touching every major ingest and query option.

```bash
# 1) Grab a small public corpus (public-domain books, plain text) into a folder
mkdir -p ./documents/gutenberg
curl -L https://www.gutenberg.org/files/1342/1342-0.txt  -o ./documents/gutenberg/pride-and-prejudice.txt   # Austen
curl -L https://www.gutenberg.org/files/11/11-0.txt      -o ./documents/gutenberg/alice-in-wonderland.txt   # Carroll
curl -L https://www.gutenberg.org/files/84/84-0.txt      -o ./documents/gutenberg/frankenstein.txt          # Shelley

# 2) Ingest it. As a top-level folder it becomes its own collection: documents__gutenberg
./ingest.sh ./documents/gutenberg            # first pass
./ingest.sh --recreate ./documents/gutenberg # clean rebuild if you re-tune chunking
./ingest.sh --formats                        # (aside) see every supported format

# 3) Confirm the source is registered and browse its structure
./query.sh --list-sources
./query.sh --source gutenberg --list-folders

# 4) Retrieval only — no LLM, just see what the hybrid search pulls (~3s)
./query.sh --rag-only --source gutenberg "the monster and its creator"

# 5) Fast grounded answer at the smallest tier (~25s)
./query.sh --ultrafast --source gutenberg "Who is Elizabeth Bennet and what is she like?"

# 6) Default adaptive cascade, with citations
./query.sh --citations --source gutenberg "How does Alice get to Wonderland?"

# 7) Bound the work explicitly (cap escalation) and force multi-pass retrieval
./query.sh --max-tier 1 --multipass --source gutenberg "themes of ambition and consequence in Frankenstein"

# 8) Deepest tier + web fallback (slow, needs SearXNG up)
./query.sh --full --source gutenberg "Compare the narrators of these three novels"

# 9) Companion tools on a single file
./summarize.sh ./documents/gutenberg/frankenstein.txt focus on the creature's motivation
./extract.sh   ./documents/gutenberg/pride-and-prejudice.txt "list the main characters and their relationships"

# 10) Web ingest (a different source entirely) — crawl a site into its own collection
./ingest.sh --url https://www.gutenberg.org/ebooks/author/68
```

> Note: the shipped embedding + SPLADE models are English-only, so this English
> corpus is an ideal demo. For a French/other-language corpus, expect retrieval
> quality to drop until a multilingual embedding model is wired in.

## System Requirements

Embeddings are fixed at FastEmbed `bge-base-en-v1.5` (768 dim) on every tier, so
the vector space stays consistent regardless of hardware. Only the LLM tiers that
fit the RAM budget are downloaded.

### Minimum (4GB RAM)
- LLM tiers: `rag-quick` only (qwen2.5-1.5b-instruct)
- Batch size: 32
- Swap: 4GB recommended

### Recommended (8GB RAM)
- LLM tiers: `rag-quick` + `rag-default` (qwen2.5-3b-instruct)
- Batch size: 64

### Optimal (16GB+ RAM)
- LLM tiers: `rag-quick` + `rag-default` + `rag-deep` (Qwen3-4B-Instruct-2507)
- Batch size: 96

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG System                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ llama-swap  │  │   Qdrant    │  │      SearXNG        │ │
│  │ +llama.cpp  │  │ (Vectors)   │  │   (Web Search)      │ │
│  │  :11434     │  │  :6333      │  │      :8085          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐│
│  │                 Python RAG Pipeline                      ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   ││
│  │  │ Ingest   │ │ Hybrid   │ │ Rerank   │ │ Generate │   ││
│  │  │ +Chunk   │→│ Search   │→│ +CRAG    │→│ +Reflect │   ││
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  FastEmbed  │  │  FlashRank  │  │   Unstructured.io   │ │
│  │ (Embeddings)│  │ (Reranking) │  │  (Doc Parsing)      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**LLM backend — llama-swap + llama.cpp.** The system talks to an OpenAI-compatible
API on `:11434` served by [llama-swap](https://github.com/mostlygeek/llama-swap),
which hot-swaps `llama-server` (llama.cpp) instances on demand. The three query
tiers map to three models, loaded only when used and unloaded after a TTL:

| Entry point | llama-swap model | GGUF (Q4_K_M) |
|-------------|------------------|---------------|
| `--ultrafast`         | `rag-quick`   | qwen2.5-1.5b-instruct |
| default cascade       | `rag-default` | qwen2.5-3b-instruct   |
| `--full` / top of cascade | `rag-deep` | Qwen3-4B-Instruct-2507 |

Which models are downloaded depends on the available RAM budget (total RAM minus
a reserve): the 1.5b is always fetched; the 3b and 4b are added only if they fit.
Embeddings use FastEmbed (`bge-base-en-v1.5`, 768-d) as the primary
source; llama-swap serves a matching `bge-base` GGUF (`rag-embed`) only as a
fallback, so the vector space stays consistent and no re-ingest is needed.

**Low memory footprint by design.** Because llama-swap loads a model only on the
first request that needs it and unloads it after an idle TTL, at most one LLM is
resident at a time — so the three tiers cost the RAM of the *largest one in use*,
not the sum. On a constrained box this is what makes multiple model tiers
affordable at all.

### llama-swap web console

llama-swap ships a built-in web UI at **`http://<host>:11434/ui/`** — a zero-cost
way to operate and test the backend without touching the CLI:

- **Models** — see which tier is currently loaded/unloaded, load or evict a model
  on demand.
- **Playground** — a chat panel to try each model (`rag-quick` / `rag-default` /
  `rag-deep`) directly and eyeball quality/speed before wiring it into a query.
- **Activity** — per-request logging: prompt/generation token counts and timing
  for every call the pipeline makes.
- **Performance metrics** — graphical throughput (tokens/sec) and latency per
  request, handy for comparing tiers on your hardware.
- **Logs** — live upstream `llama-server` + proxy logs, so a slow or failed
  generation is diagnosable in the browser.

### Services & endpoints

Once the stack is up (`./status.sh`), these local endpoints are available:

| Service | URL | Purpose |
|---------|-----|---------|
| **llama-swap** — UI | `http://localhost:11434/ui/` | Models, playground, activity, metrics, logs |
| **llama-swap** — API | `http://localhost:11434/v1` | OpenAI-compatible API the pipeline calls |
| **Qdrant** — dashboard | `http://localhost:6333/dashboard` | Browse collections/points, run queries |
| **Qdrant** — REST / gRPC | `http://localhost:6333` / `:6334` | Vector DB API (gRPC is the fast path) |
| **SearXNG** | `http://localhost:8085` | Private metasearch for web-fallback (CRAG) |
| **llama-server** | *internal, dynamic port* | llama.cpp workers — spawned/torn down by llama-swap on `127.0.0.1`, not exposed directly |

Ports come from `config.env` (`QDRANT_HOST`, `SEARXNG_URL`, `LLM_API_BASE`); the
table shows the defaults. The `llama-server` worker port is assigned dynamically
by llama-swap per model — to find the one in use, open the console **Logs** tab
(`:11434/ui/`, the upstream line shows `--port <N>`) or run
`journalctl -u rag-llm.service -f`. You normally never call it directly; go
through the llama-swap API on `:11434`.

## Configuration

All settings are in `config.env`. Key options:

```bash
# Models
LLM_MODEL=qwen2.5:3b
FASTEMBED_MODEL=BAAI/bge-base-en-v1.5

# Performance
QDRANT_BATCH_SIZE=64
CHUNK_SIZE=600
DEFAULT_TOP_K=6

# Features (enable/disable)
HYBRID_SEARCH_MODE=native
RERANK_ENABLED=false
CRAG_ENABLED=false
REFLECTION_ENABLED=true

# Cache
QUERY_CACHE_ENABLED=true
QUERY_CACHE_TTL=3600
```

### Why those feature defaults?

These four defaults are the result of actual measurement on this box (see the
opt-in [`benchmark/`](benchmark/) BEIR harness), not guesses:

| Setting | Default | Why |
|---------|---------|-----|
| `HYBRID_SEARCH_MODE` | `native` | Fusion of dense + sparse results is done **server-side inside Qdrant** (native RRF) rather than in Python. It's the fast path — one query round-trip, fusion happens next to the data — and it's why hybrid search adds negligible latency over dense-only. |
| `RERANK_ENABLED` | `false` | The cross-encoder reranker (`ms-marco-MiniLM`) **consistently *lowered* nDCG in benchmarks** (−0.024 to −0.063 across datasets) while adding **~1000× the retrieval latency** (~6 s/query). It's an English MS-MARCO model, out-of-domain on most real corpora. Off by default; flip it on only if you've measured a gain on *your* data. |
| `CRAG_ENABLED` | `false` | Corrective RAG falls back to a **web search** (SearXNG) when local evidence is weak — but that adds latency and an external dependency, and the local metasearch is the weakest link. Kept off for the default fast/offline path; opt in per-query with `--full`. |
| `REFLECTION_ENABLED` | `true` | A cheap self-check on whether the drafted answer is actually **grounded** in the retrieved context. If it isn't, the pipeline regenerates **at the next tier up** instead of re-rolling the same model. Low-cost hallucination insurance — the one quality feature that earns its keep by default. |

## Scripts Reference

| Script | Description |
|--------|-------------|
| `setup-rag-core.sh` | Install core dependencies (Docker, Qdrant, SearXNG) |
| `setup-rag-llm-backend.sh` | Migrate LLM backend from Ollama to llama-swap + llama.cpp |
| `setup-rag-ingest.sh` | Create document ingestion pipeline |
| `setup-rag-query.sh` | Create query processing pipeline |
| `setup-rag-backup.sh` | Backup and restore utilities |
| `ingest.sh` | Ingest documents |
| `query.sh` | Query the system |
| `status.sh` | Check system status |
| `monitor.sh` | Real-time monitoring dashboard |
| `evaluate.sh` | Run RAGAS quality evaluation |
| `backup.sh` | Create backup |
| `restore.sh` | Restore from backup |

## Comparison with Similar Projects

| Feature | This Project | PrivateGPT | LocalGPT | LightRAG |
|---------|-------------|------------|----------|----------|
| Hybrid Search | ✅ Native | ❌ | ✅ | ✅ |
| Low-RAM Optimization | ✅ | ⚠️ | ❌ | ❌ |
| French OCR | ✅ | ❌ | ❌ | ❌ |
| CRAG Web Fallback | ✅ | ❌ | ❌ | ❌ |
| Map/Reduce Summary | ✅ | ❌ | ❌ | ❌ |
| Self-Reflection | ✅ | ❌ | ✅ | ❌ |
| SBC/ARM Support | ✅ | ⚠️ | ❌ | ❌ |
| Docker Compose | ❌ | ✅ | ✅ | ✅ |

## Troubleshooting

### Out of Memory (OOM)

```bash
# Check swap
free -h

# Create swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Qdrant Not Starting

```bash
# Check container logs
docker logs qdrant

# Verify data directory permissions
sudo chmod 777 /mnt/dietpi_userdata/qdrant
```

### LLM backend (llama-swap) not responding

```bash
# Service logs
journalctl -u rag-llm.service -n 50

# Is the proxy up and which models are exposed?
curl -sf http://127.0.0.1:11434/v1/models | python3 -m json.tool

# Re-run the backend setup (idempotent)
sudo bash setup-rag-llm-backend.sh

# Check disk space (GGUF models live in /var/lib/rag-llm/models)
df -h
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Local LLM inference (CPU)
- [llama-swap](https://github.com/mostlygeek/llama-swap) - On-demand model hot-swap proxy
- [Qdrant](https://qdrant.tech/) - Vector database
- [FastEmbed](https://github.com/qdrant/fastembed) - Fast embeddings
- [Unstructured.io](https://unstructured.io/) - Document parsing
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) - Neural reranking
- [btop](https://github.com/aristocratos/btop) - Resource monitor; handy for watching CPU/RAM during ingest and query
- [DietPi](https://dietpi.com/) - Optimized Linux for SBCs
