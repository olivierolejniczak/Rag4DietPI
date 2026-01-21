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
| **Software catalog** | Easy Docker/Ollama installation via `dietpi-software` |
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
- **2-Layer Caching**: Search results + LLM response caching

## Quick Start

### Prerequisites

- Linux (Debian/Ubuntu/DietPi)
- 4GB+ RAM (8GB+ recommended)
- 20GB+ free disk space
- Docker installed

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-system.git
cd rag-system

# Run setup scripts (as root or with sudo)
sudo bash setup-rag-core.sh
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

## System Requirements

### Minimum (4GB RAM)
- Model: qwen2.5:1.5b
- Embedding: bge-small-en (384 dim)
- Batch size: 32
- Swap: 4GB recommended

### Recommended (8GB RAM)
- Model: qwen2.5:3b
- Embedding: bge-base-en (768 dim)
- Batch size: 64

### Optimal (16GB+ RAM)
- Model: qwen2.5:7b
- Embedding: bge-large-en (1024 dim)
- Batch size: 96

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG System                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Ollama    │  │   Qdrant    │  │      SearXNG        │ │
│  │  (LLM)      │  │ (Vectors)   │  │   (Web Search)      │ │
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
QDRANT_CACHE_ENABLED=true
RESPONSE_CACHE_ENABLED=true
```

## Scripts Reference

| Script | Description |
|--------|-------------|
| `setup-rag-core.sh` | Install core dependencies (Docker, Qdrant, Ollama, SearXNG) |
| `setup-rag-ingest.sh` | Create document ingestion pipeline |
| `setup-rag-query.sh` | Create query processing pipeline |
| `setup-rag-webui.sh` | Optional web interface |
| `setup-rag-backup.sh` | Backup and restore utilities |
| `ingest.sh` | Ingest documents |
| `query.sh` | Query the system |
| `status.sh` | Check system status |
| `monitor.sh` | Real-time monitoring dashboard |
| `evaluate.sh` | Run RAGAS quality evaluation |
| `backup.sh` | Create backup |
| `restore.sh` | Restore from backup |

## Web Interface (Optional)

```bash
# Install web UI
sudo bash setup-rag-webui.sh

# Start web server
./webui.sh

# Access at http://localhost:5000 or http://<your-ip>:5000
```

Features:
- ChatGPT-style conversation interface
- Mode selection (Standard/RAG-only/Web/Full)
- Source citations
- Mobile-friendly design

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
| Web UI | ✅ | ✅ | ✅ | ❌ |
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

### Ollama Model Download Fails

```bash
# Manual download with progress
ollama pull qwen2.5:3b

# Check disk space
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

- [Ollama](https://ollama.com/) - Local LLM inference
- [Qdrant](https://qdrant.tech/) - Vector database
- [FastEmbed](https://github.com/qdrant/fastembed) - Fast embeddings
- [Unstructured.io](https://unstructured.io/) - Document parsing
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) - Neural reranking
- [DietPi](https://dietpi.com/) - Optimized Linux for SBCs
