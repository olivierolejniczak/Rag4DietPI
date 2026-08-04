# RAG System - Private Document Intelligence

A complete, self-hosted Retrieval-Augmented Generation (RAG) system designed for offline deployment on resource-constrained hardware. Query your private documents using local LLMs with advanced retrieval techniques.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Linux%20Debian-orange.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-green.svg)

## Why This Project?

Most RAG solutions require cloud services or powerful hardware. This system was designed for:

- **Complete Privacy**: 100% offline operation, no data leaves your machine
- **Low-Resource Deployment**: Runs on Raspberry Pi 4/5, mini-PCs, or any system with 4GB+ RAM
- **Production-Ready**: Battle-tested on DietPi/Debian systems
- **Advanced Retrieval**: Hybrid search, reranking, CRAG, and more

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
# Summarize a long document
./query.sh --summarize ./documents/report.pdf

# Extract structured data
./query.sh --extract "List all people and their roles" ./documents/team.pdf

# Ingest a website
./ingest.sh --url https://example.com --max-depth 2

# Tiered performance modes
./query.sh --mode quick "simple question"   # ~30s
./query.sh --mode default "normal query"    # ~90s
./query.sh --mode deep "complex analysis"   # ~3-5min
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

| Tier (`--mode`) | llama-swap model | GGUF (Q4_K_M) |
|-----------------|------------------|---------------|
| `quick` / `--ultrafast` | `rag-quick`   | qwen2.5-1.5b-instruct |
| `default`               | `rag-default` | qwen2.5-3b-instruct   |
| `deep` / `--full`       | `rag-deep`    | Qwen3-4B-Instruct-2507 |

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
table shows the defaults.

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
- [DietPi](https://dietpi.com/) - Optimized Linux for SBCs
