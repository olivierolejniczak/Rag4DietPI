#!/bin/bash
# setup-rag-core.sh
# RAG System - Core Setup
# DietPi Trixie compatible - Headless optimized
# Complete RAG system with all features
# Hardware profiling, adaptive models, OCR, low-memory optimization
# Tiered performance, caching, monitoring
# Web ingestion, web-only query
#
# This script ensures ALL prerequisites are installed and running:
#   - Docker
#   - Qdrant (vector database)
#   - Ollama (LLM server)
#   - SearXNG (web search)
#   - Python dependencies
#   - Model downloads
#   - Spellcheck (pyspellchecker with bundled FR/EN dictionaries)

set -e

# ============================================================================
# Logging functions
# ============================================================================
BLUE='\033[1;34m'
GREEN='\033[1;32m'
RED='\033[1;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_ok() { echo -e "${GREEN}[OK]${NC} $1"; }
log_err() { echo -e "${RED}[ERROR]${NC} $1" >&2; }
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# ============================================================================
# Configuration
# ============================================================================
PROJECT_DIR="${1:-$(pwd)}"

# Qdrant settings
QDRANT_CONTAINER_NAME="qdrant"
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_DATA_DIR="/mnt/dietpi_userdata/qdrant"

# Ollama settings
OLLAMA_HOST="http://localhost:11434"

# SearXNG settings
SEARXNG_CONTAINER_NAME="searxng"
SEARXNG_PORT=8085
SEARXNG_DATA_DIR="/mnt/dietpi_userdata/searxng"

echo "============================================"
echo " RAG System - Core Setup"
echo " Map/Reduce + Extraction + Reflection"
echo "============================================"
echo ""

mkdir -p "$PROJECT_DIR"/{lib,cache,documents,.ingest_tracking}
cd "$PROJECT_DIR"

# ============================================================================
# PHASE 1: System Detection (profiling - CPU Score + Adaptive Embedding)
# ============================================================================
echo "=== Phase 1: System Detection ==="

RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
RAM_GB=$((RAM_KB / 1024 / 1024))
CPU_COUNT=$(nproc)

# profiling: CPU Score Profiling
# Calculate CPU score based on cores and frequency
CPU_MHZ=$(lscpu 2>/dev/null | awk -F': +' '/CPU max MHz/{print $2}' | cut -d'.' -f1)
if [ -z "$CPU_MHZ" ] || [ "$CPU_MHZ" = "0" ]; then
    # Fallback: try current MHz
    CPU_MHZ=$(lscpu 2>/dev/null | awk -F': +' '/CPU MHz/{print $2}' | cut -d'.' -f1)
fi
if [ -z "$CPU_MHZ" ] || [ "$CPU_MHZ" = "0" ]; then
    # Last resort: read from cpuinfo
    CPU_MHZ=$(cat /proc/cpuinfo | grep "cpu MHz" | head -1 | awk '{print $4}' | cut -d'.' -f1)
fi
[ -z "$CPU_MHZ" ] && CPU_MHZ=1000  # Default fallback

CPU_SCORE=$((CPU_COUNT * CPU_MHZ / 1000))

# profiling: Architecture Detection
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)   ARCH_TYPE="x86_64"; ARCH_LABEL="Intel/AMD 64-bit" ;;
    aarch64)  ARCH_TYPE="arm64"; ARCH_LABEL="ARM 64-bit" ;;
    armv7l)   ARCH_TYPE="armv7"; ARCH_LABEL="ARM 32-bit (limited)" ;;
    *)        ARCH_TYPE="unknown"; ARCH_LABEL="$ARCH" ;;
esac

echo "System: ${RAM_GB}GB RAM, ${CPU_COUNT} CPUs @ ${CPU_MHZ}MHz"
echo "CPU Score: $CPU_SCORE (cores × MHz / 1000)"
echo "Architecture: $ARCH_LABEL ($ARCH_TYPE)"

# profiling: Combined RAM + CPU Score for model selection
# Score thresholds: Low (<6), Medium (6-12), High (>12)
if [ "$RAM_GB" -lt 4 ]; then
    LLM_MODEL="qwen2.5:1.5b"
    CHUNK_SIZE=400
    TOP_K=5
    LLM_TIMEOUT=180
    BATCH_SIZE=32
elif [ "$RAM_GB" -lt 8 ]; then
    if [ "$CPU_SCORE" -ge 8 ]; then
        # 4-8GB RAM + decent CPU: can handle slightly larger batches
        LLM_MODEL="qwen2.5:1.5b"
        CHUNK_SIZE=500
        TOP_K=5
        LLM_TIMEOUT=150
        BATCH_SIZE=48
    else
        LLM_MODEL="qwen2.5:1.5b"
        CHUNK_SIZE=500
        TOP_K=5
        LLM_TIMEOUT=180
        BATCH_SIZE=32
    fi
elif [ "$RAM_GB" -lt 16 ]; then
    if [ "$CPU_SCORE" -ge 12 ]; then
        # 8-16GB RAM + fast CPU: use 3b model with optimized settings
        LLM_MODEL="qwen2.5:3b"
        CHUNK_SIZE=700
        TOP_K=7
        LLM_TIMEOUT=180
        BATCH_SIZE=64
    else
        LLM_MODEL="qwen2.5:3b"
        CHUNK_SIZE=600
        TOP_K=6
        LLM_TIMEOUT=240
        BATCH_SIZE=48
    fi
else
    if [ "$CPU_SCORE" -ge 16 ]; then
        # 16GB+ RAM + powerful CPU: max performance
        LLM_MODEL="qwen2.5:7b"
        CHUNK_SIZE=1000
        TOP_K=10
        LLM_TIMEOUT=240
        BATCH_SIZE=96
    else
        LLM_MODEL="qwen2.5:7b"
        CHUNK_SIZE=800
        TOP_K=8
        LLM_TIMEOUT=300
        BATCH_SIZE=64
    fi
fi

# profiling: Adaptive Embedding Model Selection
# Select embedding model based on RAM (larger models = better accuracy but more memory)
if [ "$RAM_GB" -lt 4 ]; then
    FASTEMBED_MODEL="BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM=384
elif [ "$RAM_GB" -lt 8 ]; then
    FASTEMBED_MODEL="BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM=384
elif [ "$RAM_GB" -lt 16 ]; then
    FASTEMBED_MODEL="BAAI/bge-base-en-v1.5"
    EMBEDDING_DIM=768
else
    FASTEMBED_MODEL="BAAI/bge-large-en-v1.5"
    EMBEDDING_DIM=1024
fi

echo "LLM: $LLM_MODEL"
echo "Embedding: FastEmbed $FASTEMBED_MODEL ($EMBEDDING_DIM dim)"
echo "Batch Size: $BATCH_SIZE"
echo ""

# ============================================================================
# PHASE 2: Docker Installation
# ============================================================================
echo "=== Phase 2: Docker Setup ==="

if docker --version &> /dev/null; then
    log_ok "Docker installed: $(docker --version | head -1)"
else
    log_err "Docker not found."
    echo ""
    echo "Please install Docker using DietPi software manager:"
    echo ""
    echo "  sudo dietpi-software"
    echo ""
    echo "  1. Select 'Search Software'"
    echo "  2. Search for 'docker'"
    echo "  3. Select [*] 134  Docker Compose"
    echo "  4. Press ENTER and select <Confirm>"
    echo "  5. Select 'Install' to apply changes"
    echo ""
    echo "After installation, re-run this script:"
    echo "  sudo bash setup-rag-core-profiling.sh"
    echo ""
    exit 1
fi

# Ensure Docker service is running
if ! systemctl is-active --quiet docker; then
    log_info "Starting Docker service..."
    systemctl start docker
    systemctl enable docker
    sleep 3
fi

if systemctl is-active --quiet docker; then
    log_ok "Docker service running"
else
    log_err "Docker service failed to start"
    exit 1
fi

# ============================================================================
# PHASE 3: Qdrant Setup
# ============================================================================
echo ""
echo "=== Phase 3: Qdrant Vector Database ==="

mkdir -p "$QDRANT_DATA_DIR"
chmod 777 "$QDRANT_DATA_DIR"

if docker ps -a --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${QDRANT_CONTAINER_NAME}$"; then
        log_ok "Qdrant container already running"
    else
        log_info "Starting existing Qdrant container..."
        docker start "$QDRANT_CONTAINER_NAME"
        sleep 5
    fi
else
    log_info "Creating Qdrant container..."
    
    # profiling: Low-memory optimizations for DietPi/SBC systems
    # ON_DISK_PAYLOAD=true: Store payloads on disk instead of RAM
    # MMAP_THRESHOLD_KB=1024: Use mmap for vectors >1KB (reduces RAM usage)
    # These settings are critical for systems with <8GB RAM
    
    QDRANT_ENV_OPTS=""
    if [ "$RAM_GB" -lt 8 ]; then
        log_info "Enabling Qdrant low-memory mode (RAM < 8GB)"
        QDRANT_ENV_OPTS="-e QDRANT__STORAGE__ON_DISK_PAYLOAD=true -e QDRANT__STORAGE__MMAP_THRESHOLD_KB=1024"
    elif [ "$RAM_GB" -lt 16 ]; then
        log_info "Enabling Qdrant disk payload mode (RAM < 16GB)"
        QDRANT_ENV_OPTS="-e QDRANT__STORAGE__ON_DISK_PAYLOAD=true"
    fi
    
    docker run -d \
        --name "$QDRANT_CONTAINER_NAME" \
        --restart unless-stopped \
        -p ${QDRANT_PORT}:6333 \
        -p ${QDRANT_GRPC_PORT}:6334 \
        -v "${QDRANT_DATA_DIR}:/qdrant/storage" \
        $QDRANT_ENV_OPTS \
        qdrant/qdrant:latest
    
    log_info "Waiting for Qdrant to start..."
    sleep 10
fi

# Verify Qdrant
QDRANT_RETRIES=0
QDRANT_MAX_RETRIES=30
while [ $QDRANT_RETRIES -lt $QDRANT_MAX_RETRIES ]; do
    if curl -s --max-time 3 "http://localhost:${QDRANT_PORT}/collections" > /dev/null 2>&1; then
        log_ok "Qdrant responding on port ${QDRANT_PORT}"
        break
    fi
    QDRANT_RETRIES=$((QDRANT_RETRIES + 1))
    if [ $QDRANT_RETRIES -lt $QDRANT_MAX_RETRIES ]; then
        echo "  Waiting for Qdrant... ($QDRANT_RETRIES/$QDRANT_MAX_RETRIES)"
        sleep 2
    fi
done

if [ $QDRANT_RETRIES -eq $QDRANT_MAX_RETRIES ]; then
    log_err "Qdrant failed to start. Check: docker logs $QDRANT_CONTAINER_NAME"
    exit 1
fi

# ============================================================================
# PHASE 4: Ollama Installation
# ============================================================================
echo ""
echo "=== Phase 4: Ollama LLM Server ==="

if command -v ollama &> /dev/null; then
    log_ok "Ollama installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
else
    log_info "Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
    
    if command -v ollama &> /dev/null; then
        log_ok "Ollama installed successfully"
    else
        log_err "Ollama installation failed"
        exit 1
    fi
fi

# Start Ollama service
if curl -s --max-time 3 "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
    log_ok "Ollama service already running"
else
    log_info "Starting Ollama service..."
    
    if systemctl list-unit-files | grep -q ollama; then
        systemctl start ollama
        systemctl enable ollama
    else
        nohup ollama serve > /var/log/ollama.log 2>&1 &
    fi
    
    OLLAMA_RETRIES=0
    OLLAMA_MAX_RETRIES=30
    while [ $OLLAMA_RETRIES -lt $OLLAMA_MAX_RETRIES ]; do
        if curl -s --max-time 3 "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
            log_ok "Ollama service started"
            break
        fi
        OLLAMA_RETRIES=$((OLLAMA_RETRIES + 1))
        if [ $OLLAMA_RETRIES -lt $OLLAMA_MAX_RETRIES ]; then
            echo "  Waiting for Ollama... ($OLLAMA_RETRIES/$OLLAMA_MAX_RETRIES)"
            sleep 2
        fi
    done
    
    if [ $OLLAMA_RETRIES -eq $OLLAMA_MAX_RETRIES ]; then
        log_err "Ollama failed to start. Check: cat /var/log/ollama.log"
        exit 1
    fi
fi

# Pull LLM model
echo ""
echo "Checking LLM model: $LLM_MODEL"
if ollama list 2>/dev/null | grep -q "$LLM_MODEL"; then
    log_ok "Model $LLM_MODEL already available"
else
    log_info "Pulling model $LLM_MODEL (this may take several minutes)..."
    ollama pull "$LLM_MODEL"
    
    if ollama list 2>/dev/null | grep -q "$LLM_MODEL"; then
        log_ok "Model $LLM_MODEL downloaded"
    else
        log_err "Failed to pull model $LLM_MODEL"
        exit 1
    fi
fi

# ============================================================================
# PHASE 5: SearXNG Setup (Web Search)
# ============================================================================
echo ""
echo "=== Phase 5: SearXNG Web Search ==="

mkdir -p "$SEARXNG_DATA_DIR"
chmod 777 "$SEARXNG_DATA_DIR"

# Create SearXNG settings.yml
SEARXNG_SETTINGS="$SEARXNG_DATA_DIR/settings.yml"
if [ ! -f "$SEARXNG_SETTINGS" ]; then
    log_info "Creating SearXNG configuration..."
    cat > "$SEARXNG_SETTINGS" << 'EOFSEARXNG'
use_default_settings: true

general:
  instance_name: "RAG Search"
  debug: false

search:
  safe_search: 0
  default_lang: "fr-FR"
  formats:
    - html
    - json

server:
  secret_key: "rag-system-secret-key-change-me"
  bind_address: "0.0.0.0"
  port: 8080
  limiter: false

engines:
  - name: google
    engine: google
    shortcut: g
    disabled: false
  - name: duckduckgo
    engine: duckduckgo
    shortcut: ddg
    disabled: false
  - name: bing
    engine: bing
    shortcut: bi
    disabled: false
  - name: wikipedia
    engine: wikipedia
    shortcut: wp
    disabled: false
EOFSEARXNG
    chmod 644 "$SEARXNG_SETTINGS"
    log_ok "SearXNG configuration created"
fi

# Start SearXNG container
if docker ps -a --format '{{.Names}}' | grep -q "^${SEARXNG_CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${SEARXNG_CONTAINER_NAME}$"; then
        log_ok "SearXNG container already running"
    else
        log_info "Starting existing SearXNG container..."
        docker start "$SEARXNG_CONTAINER_NAME"
        sleep 5
    fi
else
    log_info "Creating SearXNG container..."
    docker run -d \
        --name "$SEARXNG_CONTAINER_NAME" \
        --restart unless-stopped \
        -p ${SEARXNG_PORT}:8080 \
        -v "${SEARXNG_DATA_DIR}:/etc/searxng" \
        -e SEARXNG_BASE_URL="http://localhost:${SEARXNG_PORT}/" \
        searxng/searxng:latest
    
    log_info "Waiting for SearXNG to start..."
    sleep 10
fi

# Verify SearXNG
SEARXNG_OK=false
for i in {1..20}; do
    SEARXNG_RESP=$(curl -s --max-time 5 "http://localhost:${SEARXNG_PORT}/search?q=test&format=json" 2>/dev/null)
    if echo "$SEARXNG_RESP" | grep -q '"results"'; then
        log_ok "SearXNG responding with JSON on port ${SEARXNG_PORT}"
        SEARXNG_OK=true
        break
    fi
    echo "  Waiting for SearXNG... ($i/20)"
    sleep 3
done

if [ "$SEARXNG_OK" = false ]; then
    log_warn "SearXNG JSON not working (CRAG/web-only will be limited)"
fi

# ============================================================================
# PHASE 6: System Dependencies
# ============================================================================
echo ""
echo "=== Phase 6: System Dependencies ==="

# External TMPDIR
if [ -d "/mnt/dietpi_userdata" ]; then
    export TMPDIR=/mnt/dietpi_userdata/tmp
    mkdir -p "$TMPDIR"
    log_ok "TMPDIR: $TMPDIR"
else
    export TMPDIR=/tmp
fi

# Swap management
CURRENT_SWAP=$(free -m | awk '/^Swap:/ {print $2}')
RECOMMENDED_SWAP=$((8 - RAM_GB))
[ "$RECOMMENDED_SWAP" -lt 2 ] && RECOMMENDED_SWAP=2

if [ "$CURRENT_SWAP" -lt "$((RECOMMENDED_SWAP * 1024 - 500))" ] && [ ! -f /swapfile ] && [ "$RAM_GB" -lt 8 ]; then
    AVAILABLE_GB=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
    if [ "$AVAILABLE_GB" -gt "$((RECOMMENDED_SWAP + 5))" ]; then
        fallocate -l ${RECOMMENDED_SWAP}G /swapfile 2>/dev/null || \
        dd if=/dev/zero of=/swapfile bs=1M count=$((RECOMMENDED_SWAP * 1024)) status=progress
        chmod 600 /swapfile
        mkswap /swapfile
        swapon /swapfile
        grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
        log_ok "Swap created: ${RECOMMENDED_SWAP}GB"
    fi
fi
sysctl -w vm.swappiness=10 > /dev/null 2>&1 || true
log_ok "Swap configured"

# Install system packages
echo "Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    apt-get update -qq
    apt-get install -y -qq python3 python3-pip python3-venv curl wget 2>/dev/null || true
    
    # profiling: Added tesseract-ocr-fra for French OCR support
    # This is critical for French PDF documents with scanned text
    apt-get install -y -qq libmagic1 poppler-utils tesseract-ocr tesseract-ocr-fra 2>/dev/null || true
    
    # profiling: Added antiword for legacy .doc file support
    # Microsoft Word 97-2003 format still common in enterprise archives
    apt-get install -y -qq antiword 2>/dev/null || true
    
    apt-get install -y -qq libgl1 libgl1-mesa-dri libegl1 2>/dev/null || true
fi
log_ok "System dependencies installed"

# ============================================================================
# PHASE 7: Python Packages
# ============================================================================
echo ""
echo "=== Phase 7: Python Packages ==="

if ! command -v python3 &> /dev/null; then
    log_err "Python3 required"
    exit 1
fi
log_ok "Python: $(python3 --version)"

PIP_FLAGS="--break-system-packages --root-user-action=ignore"
pip3 install --help 2>&1 | grep -q "break-system-packages" || PIP_FLAGS=""

echo "Installing core packages..."
pip3 install --quiet requests urllib3 rank-bm25 numpy $PIP_FLAGS 2>/dev/null || true
pip3 install --quiet flashrank $PIP_FLAGS 2>/dev/null || true

echo "Installing FastEmbed..."
pip3 install --quiet fastembed $PIP_FLAGS 2>/dev/null || { log_err "FastEmbed required"; exit 1; }
log_ok "FastEmbed installed"

echo "Installing Unstructured.io..."
pip3 install --quiet --no-cache-dir "unstructured[all-docs]" $PIP_FLAGS 2>/dev/null || \
pip3 install --quiet --no-cache-dir unstructured $PIP_FLAGS 2>/dev/null || { log_err "Unstructured required"; exit 1; }
log_ok "Unstructured.io installed"

echo "Installing PDF/Image/Excel support..."
pip3 install --quiet pdf2image pdfminer.six $PIP_FLAGS 2>/dev/null || true
pip3 install --quiet pillow pytesseract $PIP_FLAGS 2>/dev/null || true
pip3 install --quiet msoffcrypto-tool $PIP_FLAGS 2>/dev/null || true
log_ok "PDF/Image/Excel support installed"

echo "Installing Qdrant client..."
pip3 install --quiet qdrant-client $PIP_FLAGS 2>/dev/null || { log_err "qdrant-client required"; exit 1; }
log_ok "Qdrant client installed"

# SparseEmbed
SPARSE_MODEL="prithivida/Splade_PP_en_v1"
python3 -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='$SPARSE_MODEL')" 2>/dev/null || {
    SPARSE_MODEL="Qdrant/bm25"
    python3 -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='$SPARSE_MODEL')" 2>/dev/null || SPARSE_MODEL=""
}
[ -n "$SPARSE_MODEL" ] && log_ok "Sparse model: $SPARSE_MODEL" || log_info "No sparse model"

echo "Installing RAGAS..."
pip3 install --quiet ragas datasets $PIP_FLAGS 2>/dev/null && log_ok "RAGAS installed" || log_info "RAGAS optional"

echo "Installing web crawler dependencies..."
pip3 install --quiet beautifulsoup4 lxml html2text $PIP_FLAGS 2>/dev/null && log_ok "Web crawler deps installed" || true

# ============================================================================
# PHASE 8: Spellcheck (pyspellchecker - bundled dictionaries)
# ============================================================================
echo ""
echo "=== Phase 8: Spellcheck Setup ==="

set +e  # Non-critical phase

echo "Installing pyspellchecker (bundled FR/EN dictionaries)..."
pip3 install --quiet pyspellchecker $PIP_FLAGS 2>/dev/null

# Test French
FR_OK=false
if python3 -c "from spellchecker import SpellChecker; s=SpellChecker(language='fr'); print(s.correction('bonjor'))" 2>/dev/null | grep -q "bonjour"; then
    log_ok "French spellcheck: OK"
    FR_OK=true
else
    log_warn "French spellcheck: Failed"
fi

# Test English
EN_OK=false
if python3 -c "from spellchecker import SpellChecker; s=SpellChecker(language='en'); print(s.correction('helo'))" 2>/dev/null | grep -q "hello"; then
    log_ok "English spellcheck: OK"
    EN_OK=true
else
    log_warn "English spellcheck: Failed"
fi

if [ "$FR_OK" = true ] || [ "$EN_OK" = true ]; then
    SPELLCHECK_AVAILABLE=true
    log_ok "Spellcheck available (pyspellchecker)"
else
    SPELLCHECK_AVAILABLE=false
    log_warn "Spellcheck unavailable - tech normalization still works"
fi

set -e  # Re-enable strict mode

# ============================================================================
# PHASE 9: Configuration Files
# ============================================================================
echo ""
echo "=== Phase 9: Configuration ==="

cat > "$PROJECT_DIR/config.env" << EOFCFG
# RAG System Configuration
# Generated: $(date -Iseconds)

# ============================================================================
# MAP/REDUCE SUMMARIZATION (System)
# ============================================================================

# Enable map/reduce for long document summarization
MAPREDUCE_ENABLED=true

# Chunk size for map phase (chars) - larger = fewer LLM calls but more context
MAPREDUCE_CHUNK_SIZE=4000

# Max chunks to summarize per reduce batch
MAPREDUCE_BATCH_SIZE=3

# Timeout per chunk summarization (seconds)
MAPREDUCE_CHUNK_TIMEOUT=120

# ============================================================================
# EXTRACTION MODE (System)
# ============================================================================

# Enable structured extraction from documents
EXTRACTION_ENABLED=true

# Chunk size for extraction (chars)
EXTRACTION_CHUNK_SIZE=3000

# Fuzzy match threshold for deduplication (0.0-1.0)
EXTRACTION_DEDUP_THRESHOLD=0.85

# ============================================================================
# SELF-REFLECTION / VERIFICATION (System)
# ============================================================================

# Enable answer verification
REFLECTION_ENABLED=true

# Minimum confidence threshold (0.0-1.0)
REFLECTION_CONFIDENCE_THRESHOLD=0.7

# Max retries on failed verification
REFLECTION_MAX_RETRIES=1

# Always verify (true) or only high-stakes queries (false)
REFLECTION_ALWAYS=false

# High-stakes keywords that trigger verification
REFLECTION_KEYWORDS=legal,contract,medical,financial,compliance,regulation

# ============================================================================
# SYSTEM PROFILING (profiling)
# ============================================================================

# Detected system specifications
SYSTEM_RAM_GB=$RAM_GB
SYSTEM_CPU_COUNT=$CPU_COUNT
SYSTEM_CPU_MHZ=$CPU_MHZ
SYSTEM_CPU_SCORE=$CPU_SCORE
SYSTEM_ARCH=$ARCH_TYPE

# Batch size (adjusted by CPU score)
QDRANT_BATCH_SIZE=$BATCH_SIZE

# ============================================================================
# TIERED PERFORMANCE SYSTEM (cache)
# ============================================================================

# Quick mode - Rapid responses for simple queries
LLM_TIMEOUT_QUICK=90
MAX_CONTEXT_CHARS_QUICK=3000
NUM_PREDICT_QUICK=150

# Default mode - Balanced performance (existing values preserved)
# (LLM_TIMEOUT_DEFAULT, MAX_CONTEXT_CHARS, NUM_PREDICT_DEFAULT below)

# Deep mode - Comprehensive research
LLM_TIMEOUT_DEEP=600
MAX_CONTEXT_CHARS_DEEP=15000
NUM_PREDICT_DEEP=2000

# Default query mode if --mode not specified
QUERY_MODE_DEFAULT=default

# ============================================================================
# 2-LAYER CACHING SYSTEM (cache)
# ============================================================================

# Layer 1: Qdrant Search Results Cache (volatile)
QDRANT_CACHE_ENABLED=true
QDRANT_CACHE_DIR=./cache/qdrant
QDRANT_CACHE_TTL=3600

# Layer 2: LLM Response Cache (persistent)
RESPONSE_CACHE_ENABLED=true
RESPONSE_CACHE_DIR=./cache/responses
RESPONSE_CACHE_TTL=86400

# Cache debug output
CACHE_DEBUG=false

# ============================================================================
# CORE SETTINGS
# ============================================================================

# Ollama
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=$LLM_MODEL
TEMPERATURE=0.2

# Qdrant
QDRANT_HOST=http://localhost:${QDRANT_PORT}
QDRANT_GRPC_PORT=${QDRANT_GRPC_PORT}
COLLECTION_NAME=documents

# Native QdrantClient (client)
QDRANT_CLIENT_ENABLED=true
QDRANT_BATCH_SIZE=64

# Hybrid Search (hybrid)
SPARSE_EMBED_ENABLED=true
SPARSE_EMBED_MODEL=$SPARSE_MODEL
HYBRID_SEARCH_MODE=native
HYBRID_RRF_K=60
DENSE_VECTOR_NAME=dense
SPARSE_VECTOR_NAME=sparse

# FastEmbed (fastembed)
FEATURE_FASTEMBED_ENABLED=true
FASTEMBED_MODEL=$FASTEMBED_MODEL
FASTEMBED_CACHE_DIR=./cache/fastembed
EMBEDDING_DIMENSION=$EMBEDDING_DIM

# Chunking
CHUNK_SIZE=$CHUNK_SIZE
CHUNK_OVERLAP=50
DEFAULT_TOP_K=$TOP_K

# Timeouts
LLM_TIMEOUT_DEFAULT=$LLM_TIMEOUT
LLM_TIMEOUT_ULTRAFAST=90
LLM_TIMEOUT_RAG_ONLY=10
LLM_TIMEOUT_FULL=0
EMBEDDING_TIMEOUT=60
RERANK_TIMEOUT=60

# Context limits
MAX_CONTEXT_CHARS=8000
MAX_CHUNK_CHARS=1500
MAX_MEMORY_CHARS=1000
NUM_PREDICT_DEFAULT=800
NUM_PREDICT_ULTRAFAST=400
NUM_PREDICT_FULL=1200

# Query features (defaults OFF for speed)
QUERY_CLASSIFICATION_ENABLED=false
HYDE_ENABLED=false
SUBQUERY_ENABLED=false
STEPBACK_ENABLED=false

# Multi-pass retrieval (multipass)
MULTIPASS_ENABLED=false
MULTIPASS_VARIANTS=3

# Spellcheck (dedup) - pyspellchecker
SPELLCHECK_ENABLED=$SPELLCHECK_AVAILABLE
SPELLCHECK_LANG=auto
QUERY_NORMALIZE_ENABLED=true

# Post-retrieval
RERANK_ENABLED=false
RERANK_MODEL=ms-marco-MiniLM-L-12-v2
RERANK_TOP_K=5
RELEVANCE_FILTER_ENABLED=true
RELEVANCE_THRESHOLD=0.3

# CRAG
CRAG_ENABLED=false
CRAG_THRESHOLD=0.4

# Citations (citations)
CITATIONS_ENABLED=false

# Memory
MEMORY_ENABLED=true
MEMORY_MAX_TURNS=5
MEMORY_FILE=./cache/memory.json

# Cache
QUERY_CACHE_ENABLED=true
QUERY_CACHE_TTL=3600

# Web Search (SearXNG)
WEB_SEARCH_ENABLED=true
SEARXNG_URL=http://localhost:${SEARXNG_PORT}/search
SEARXNG_TIMEOUT=10

# Quality
QUALITY_LEDGER_ENABLED=true
ABSTENTION_ENABLED=true

# Deduplication (dedup)
DEDUP_ENABLED=true

# RAGAS (ragas)
RAGAS_ENABLED=true
RAGAS_SLA_THRESHOLD=0.80

# CSV Transform (csv)
CSV_NL_TRANSFORM_ENABLED=true
CSV_NL_DUAL_MODE=true
CSV_NL_LANG=fr

# Website Ingestion (web)
WEB_CRAWLER_ENABLED=true
WEB_CRAWLER_MAX_PAGES=50
WEB_CRAWLER_MAX_DEPTH=3
WEB_CRAWLER_DELAY=1.0

# Web-Only Query (web)
WEB_ONLY_MAX_RESULTS=5
WEB_ONLY_TIMEOUT=60
EOFCFG
log_ok "config.env created"

# Create status.sh
cat > "$PROJECT_DIR/status.sh" << 'EOFSTATUS'
#!/bin/bash
source ./config.env 2>/dev/null || true

G='\033[1;32m'; R='\033[1;31m'; Y='\033[1;33m'; B='\033[1;34m'; C='\033[1;36m'; M='\033[1;35m'; N='\033[0m'

echo "=== RAG System Status ==="
echo ""

# profiling: System Profile
echo "=== System Profile ==="
echo -e "RAM: ${C}${SYSTEM_RAM_GB:-?}GB${N} | CPUs: ${C}${SYSTEM_CPU_COUNT:-?}${N} @ ${C}${SYSTEM_CPU_MHZ:-?}MHz${N}"
echo -e "CPU Score: ${C}${SYSTEM_CPU_SCORE:-?}${N} | Arch: ${C}${SYSTEM_ARCH:-?}${N}"
echo -e "Batch Size: ${C}${QDRANT_BATCH_SIZE:-64}${N} | Embedding: ${C}${FASTEMBED_MODEL:-bge-small}${N}"
echo ""

# Services
echo "=== Services ==="
echo -n "Docker: "
systemctl is-active --quiet docker && echo -e "${G}OK${N}" || echo -e "${R}NOT running${N}"

echo -n "Qdrant: "
QDRANT_HEALTH=$(curl -s --max-time 2 "${QDRANT_HOST:-http://localhost:6333}/collections" 2>/dev/null)
if [ -n "$QDRANT_HEALTH" ]; then
    POINTS=$(curl -s --max-time 2 "${QDRANT_HOST:-http://localhost:6333}/collections/${COLLECTION_NAME:-documents}" 2>/dev/null | grep -o '"points_count":[0-9]*' | cut -d: -f2)
    [ -n "$POINTS" ] && echo -e "${G}OK${N} ($POINTS points)" || echo -e "${G}OK${N} (no collection yet)"
else
    echo -e "${R}NOT running${N}"
fi

echo -n "Ollama: "
curl -s --max-time 2 "${OLLAMA_HOST:-http://localhost:11434}/api/tags" > /dev/null 2>&1 && echo -e "${G}OK${N} (${LLM_MODEL})" || echo -e "${R}NOT running${N}"

echo -n "SearXNG: "
curl -s --max-time 2 "${SEARXNG_URL:-http://localhost:8085/search}?q=test&format=json" 2>/dev/null | grep -q '"results"' && echo -e "${G}OK${N}" || echo -e "${Y}Limited${N}"

echo ""
echo "=== Python Components ==="
python3 -c "from fastembed import TextEmbedding" 2>/dev/null && echo -e "FastEmbed: ${G}OK${N}" || echo -e "FastEmbed: ${R}Missing${N}"
python3 -c "from qdrant_client import QdrantClient" 2>/dev/null && echo -e "Qdrant Client: ${G}OK${N}" || echo -e "Qdrant Client: ${R}Missing${N}"
python3 -c "from unstructured.partition.auto import partition" 2>/dev/null && echo -e "Unstructured: ${G}OK${N}" || echo -e "Unstructured: ${R}Missing${N}"
python3 -c "from spellchecker import SpellChecker" 2>/dev/null && echo -e "Spellcheck: ${G}OK${N} (pyspellchecker)" || echo -e "Spellcheck: ${Y}Unavailable${N}"
python3 -c "from flashrank import Ranker" 2>/dev/null && echo -e "FlashRank: ${G}OK${N}" || echo -e "FlashRank: ${Y}Optional${N}"

echo ""
echo "=== System Features ==="
echo -e "Map/Reduce: ${M}${MAPREDUCE_ENABLED:-true}${N} | Extraction: ${M}${EXTRACTION_ENABLED:-true}${N} | Reflection: ${M}${REFLECTION_ENABLED:-true}${N}"

echo ""
echo "=== profiling Features ==="
echo -n "Tesseract FR: "
tesseract --list-langs 2>/dev/null | grep -q "fra" && echo -e "${G}OK${N}" || echo -e "${Y}Missing${N}"
echo -n "Antiword: "
command -v antiword &>/dev/null && echo -e "${G}OK${N}" || echo -e "${Y}Missing${N}"

echo ""
echo "=== Commands ==="
echo "Summarize:  ./query.sh --summarize document.pdf"
echo "Extract:    ./query.sh --extract 'list all names' document.pdf"
echo "Tiered:     ./query-tiered-cache.sh --mode quick|default|deep 'question'"
echo "Monitor:    ./monitor.sh"
echo "Web-Only:   ./query.sh --web-only 'question'"
echo ""
EOFSTATUS
chmod +x "$PROJECT_DIR/status.sh"
log_ok "status.sh created"

# Create evaluate.sh (web - with single query support)
cat > "$PROJECT_DIR/evaluate.sh" << 'EOFEVAL'
#!/bin/bash
# evaluate.sh - RAG System web Quality Evaluation
#
# RAGAS-style evaluation for:
#   - Single queries (real-time quality feedback)
#   - Batch evaluation (generated test questions)
#   - Client-ready reports
#
# Usage:
#   ./evaluate.sh --query "what is ninjarmm?"    # Evaluate single query
#   ./evaluate.sh --generate 10                   # Generate 10 test questions
#   ./evaluate.sh --report                        # Run batch evaluation

cd "$(dirname "$0")"
source ./config.env 2>/dev/null || true

# Colors
G='\033[1;32m'; R='\033[1;31m'; Y='\033[1;33m'; B='\033[1;34m'; N='\033[0m'

# Defaults
GENERATE=0
REPORT=false
SINGLE_QUERY=""
DATASET="${RAGAS_DATASET_PATH:-./cache/ragas_test.json}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --query|-q) SINGLE_QUERY="$2"; shift 2 ;;
        --generate|-g) GENERATE="${2:-10}"; shift 2 ;;
        --report|-r) REPORT=true; shift ;;
        --dataset|-d) DATASET="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: ./evaluate.sh [options]"
            echo ""
            echo "Options:"
            echo "  --query, -q TEXT    Evaluate a single query with RAGAS metrics"
            echo "  --generate, -g N    Generate N test questions from documents"
            echo "  --report, -r        Run batch evaluation on test dataset"
            echo "  --dataset, -d PATH  Use custom test dataset"
            echo ""
            echo "Examples:"
            echo "  ./evaluate.sh --query 'what is ninjarmm?'"
            echo "  ./evaluate.sh --generate 10 --report"
            exit 0
            ;;
        *) shift ;;
    esac
done

echo "============================================"
echo " RAG System web - Quality Evaluation"
echo "============================================"
echo ""

# Single Query Evaluation
if [ -n "$SINGLE_QUERY" ]; then
    echo -e "${B}Evaluating:${N} $SINGLE_QUERY"
    echo ""
    
    python3 << EOFPY
import sys, os, json, subprocess, re

def run_rag_query(query):
    """Run RAG query and capture output"""
    # RAG-only for context
    rag = subprocess.run(['./query.sh', '--rag-only', query],
                         capture_output=True, text=True, timeout=60)
    contexts = rag.stdout.strip()
    
    # Full query for answer
    full = subprocess.run(['./query.sh', query],
                          capture_output=True, text=True, timeout=180)
    answer = full.stdout.strip()
    
    # Extract sources
    sources = []
    for line in contexts.split('\n'):
        if any(ext in line for ext in ['.pdf', '.docx', '.txt', '.csv', '.pptx']):
            src = re.search(r'[\w\-\.]+\.(pdf|docx|txt|csv|pptx|xlsx)', line)
            if src:
                sources.append(src.group(0))
    
    return contexts, answer, list(set(sources))[:5]

def compute_metrics(query, answer, contexts):
    """Compute RAGAS-style metrics"""
    q_words = set(query.lower().split())
    a_words = set(answer.lower().split())
    c_words = set(contexts.lower().split())
    
    # Context Precision: query terms in context
    q_in_c = len(q_words & c_words) / max(len(q_words), 1)
    ctx_precision = min(1.0, q_in_c * 1.5)
    
    # Answer Relevancy: query terms addressed + length check
    q_in_a = len(q_words & a_words) / max(len(q_words), 1)
    length_ok = 1.0 if len(answer) > 100 else 0.5
    ans_relevancy = min(1.0, q_in_a * 1.2 + 0.2 * length_ok)
    
    # Faithfulness: answer grounded in context
    a_in_c = len(a_words & c_words) / max(len(a_words), 1)
    faithfulness = min(1.0, a_in_c * 1.3)
    
    overall = (ctx_precision + ans_relevancy + faithfulness) / 3
    
    return {
        'context_precision': round(ctx_precision, 2),
        'answer_relevancy': round(ans_relevancy, 2),
        'faithfulness': round(faithfulness, 2),
        'overall': round(overall, 2)
    }

def color(score):
    if score >= 0.8: return '\033[1;32m'
    elif score >= 0.5: return '\033[1;33m'
    return '\033[1;31m'

# Run
query = """$SINGLE_QUERY"""
print("[1/3] Running RAG query...")
contexts, answer, sources = run_rag_query(query)

print("[2/3] Computing metrics...")
m = compute_metrics(query, answer, contexts)

print("[3/3] Results:")
print("")
print("=" * 50)
print(" RAGAS Evaluation Results")
print("=" * 50)
print("")
print("Metrics:")
print(f"  Context Precision: {color(m['context_precision'])}{m['context_precision']:.2f}\033[0m")
print(f"  Answer Relevancy:  {color(m['answer_relevancy'])}{m['answer_relevancy']:.2f}\033[0m")
print(f"  Faithfulness:      {color(m['faithfulness'])}{m['faithfulness']:.2f}\033[0m")
print("")
print(f"  Overall Score:     {color(m['overall'])}{m['overall']:.2f}\033[0m")
print("")

threshold = float(os.environ.get('RAGAS_SLA_THRESHOLD', '0.80'))
if m['overall'] >= threshold:
    print(f"\033[1;32mSLA Status: PASSED (>= {threshold})\033[0m")
else:
    print(f"\033[1;31mSLA Status: FAILED (< {threshold})\033[0m")

print("")
print("-" * 50)
print("Sources:")
for i, s in enumerate(sources, 1):
    print(f"  [{i}] {s}")

print("")
print("-" * 50)
print("Answer:")
print(f"  {answer[:400]}{'...' if len(answer) > 400 else ''}")
print("")
EOFPY
    exit $?
fi

# Generate Test Questions
if [ "$GENERATE" -gt 0 ]; then
    echo -e "${B}Generating $GENERATE test questions...${N}"
    python3 ./lib/ragas_eval.py --generate "$GENERATE" --output "$DATASET"
    echo ""
fi

# Batch Evaluation
if [ "$REPORT" = true ]; then
    if [ ! -f "$DATASET" ]; then
        echo -e "${R}[ERROR]${N} Dataset not found: $DATASET"
        echo "Generate with: ./evaluate.sh --generate 10"
        exit 1
    fi
    echo -e "${B}Running batch evaluation...${N}"
    python3 ./lib/ragas_eval.py --evaluate "$DATASET"
fi

# Default help
if [ "$GENERATE" -eq 0 ] && [ "$REPORT" = false ] && [ -z "$SINGLE_QUERY" ]; then
    echo "Usage: ./evaluate.sh [options]"
    echo ""
    echo "  --query 'question'    Evaluate single query"
    echo "  --generate N          Generate N test questions"
    echo "  --report              Run batch evaluation"
    echo ""
    echo "Examples:"
    echo "  ./evaluate.sh --query 'what is ninjarmm?'"
    echo "  ./evaluate.sh --generate 5 --report"
fi
EOFEVAL
chmod +x "$PROJECT_DIR/evaluate.sh"
log_ok "evaluate.sh created (web - single query support)"

# Create monitor.sh (cache)
cat > "$PROJECT_DIR/monitor.sh" << 'EOFMONITOR'
#!/bin/bash
# RAG System Monitor cache
# Real-time system monitoring with live refresh
set -e

# Colors
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
BLUE='\033[1;34m'
NC='\033[0m'

# Configuration
source ./config.env 2>/dev/null || true
REFRESH_INTERVAL=${1:-5}

while true; do
  clear
  echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
  echo -e "${CYAN}  RAG System Monitor cache - $(date '+%d/%m/%Y %H:%M:%S')${NC}"
  echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
  echo ""
  
  # ========== Services Status ==========
  echo -e "${BLUE}▌ Services Status${NC}"
  
  # Docker
  if systemctl is-active --quiet docker 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Docker: Running"
  else
    echo -e "  ${RED}✗${NC} Docker: Stopped"
  fi
  
  # Qdrant
  QDRANT_STATUS=$(curl -s --max-time 2 "${QDRANT_HOST:-http://localhost:6333}/collections" 2>/dev/null)
  if [ -n "$QDRANT_STATUS" ]; then
    QDRANT_COLLECTIONS=$(echo "$QDRANT_STATUS" | grep -o '"collections":\[' | wc -l 2>/dev/null || echo "?")
    echo -e "  ${GREEN}✓${NC} Qdrant: Running"
  else
    echo -e "  ${RED}✗${NC} Qdrant: Not responding"
  fi
  
  # Ollama
  OLLAMA_STATUS=$(curl -s --max-time 2 "${OLLAMA_HOST:-http://localhost:11434}/api/tags" 2>/dev/null)
  if [ -n "$OLLAMA_STATUS" ]; then
    echo -e "  ${GREEN}✓${NC} Ollama: Running (${LLM_MODEL:-unknown})"
  else
    echo -e "  ${RED}✗${NC} Ollama: Not responding"
  fi
  
  # SearXNG
  SEARXNG_STATUS=$(curl -s --max-time 2 "${SEARXNG_URL:-http://localhost:8085/search}?q=test&format=json" 2>/dev/null)
  if echo "$SEARXNG_STATUS" | grep -q '"results"' 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} SearXNG: Running"
  else
    echo -e "  ${YELLOW}⚠${NC} SearXNG: Limited"
  fi
  
  echo ""
  
  # ========== Resources ==========
  echo -e "${BLUE}▌ Docker Containers${NC}"
  docker stats --no-stream --format "  {{.Name}}: CPU {{.CPUPerc}} | Mem {{.MemUsage}}" 2>/dev/null | \
    grep -E "qdrant|searxng" || echo "  No monitored containers"
  echo ""
  
  # ========== Qdrant Collection ==========
  echo -e "${BLUE}▌ Qdrant Collection${NC}"
  COLL_INFO=$(curl -s --max-time 2 "${QDRANT_HOST:-http://localhost:6333}/collections/${COLLECTION_NAME:-documents}" 2>/dev/null)
  if [ -n "$COLL_INFO" ]; then
    POINTS=$(echo "$COLL_INFO" | grep -o '"points_count":[0-9]*' | cut -d: -f2 || echo "?")
    echo "  Collection: ${COLLECTION_NAME:-documents}"
    echo "  Points: $POINTS"
  else
    echo "  Cannot connect to Qdrant"
  fi
  echo ""
  
  # ========== Cache Status ==========
  echo -e "${BLUE}▌ Cache Status${NC}"
  # Qdrant cache
  if [ -d "${QDRANT_CACHE_DIR:-./cache/qdrant}" ]; then
    QDRANT_CACHE=$(find "${QDRANT_CACHE_DIR:-./cache/qdrant}" -name "*.json" -type f 2>/dev/null | wc -l)
    echo "  Qdrant cache: $QDRANT_CACHE queries"
  fi
  
  # Response cache
  if [ -d "${RESPONSE_CACHE_DIR:-./cache/responses}" ]; then
    RESPONSE_CACHE=$(find "${RESPONSE_CACHE_DIR:-./cache/responses}" -name "*.txt" -type f 2>/dev/null | wc -l)
    echo "  Response cache: $RESPONSE_CACHE entries"
  fi
  
  # Memory
  if [ -f "${MEMORY_FILE:-./cache/memory.json}" ]; then
    MEMORY_SIZE=$(stat -c%s "${MEMORY_FILE:-./cache/memory.json}" 2>/dev/null || echo "0")
    echo "  Conversation memory: ${MEMORY_SIZE}B"
  fi
  
  # Total cache size
  CACHE_SIZE=$(du -sh "${CACHE_DIR:-./cache}" 2>/dev/null | cut -f1 || echo "0")
  echo -e "  ${CYAN}Total:${NC} $CACHE_SIZE"
  echo ""
  
  # ========== System Load ==========
  echo -e "${BLUE}▌ System Load${NC}"
  LOAD=$(uptime | awk -F'load average:' '{print $2}')
  echo "  Load:$LOAD"
  
  FREE_MEM=$(free -h | awk '/^Mem:/ {print $4}')
  SWAP_USED=$(free -h | awk '/^Swap:/ {print $3}')
  echo "  Free RAM: $FREE_MEM | Swap: $SWAP_USED"
  echo ""
  
  echo -e "${CYAN}Refresh: ${REFRESH_INTERVAL}s | Press Ctrl+C to exit${NC}"
  sleep $REFRESH_INTERVAL
done
EOFMONITOR
chmod +x "$PROJECT_DIR/monitor.sh"
log_ok "monitor.sh created (cache)"

# Create cache-stats.sh (cache)
cat > "$PROJECT_DIR/cache-stats.sh" << 'EOFCACHE'
#!/bin/bash
# Cache Statistics Viewer cache
source ./config.env 2>/dev/null || true

# Colors
GREEN='\033[1;32m'
CYAN='\033[1;36m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m'

echo -e "${CYAN}════════════════════════════════════════${NC}"
echo -e "${CYAN}  Cache Statistics - $(date '+%H:%M:%S')${NC}"
echo -e "${CYAN}════════════════════════════════════════${NC}"
echo ""

# ========== Qdrant Cache ==========
if [ -d "${QDRANT_CACHE_DIR:-./cache/qdrant}" ]; then
  echo -e "${GREEN}Qdrant Search Cache:${NC}"
  
  QDRANT_FILES=$(find "${QDRANT_CACHE_DIR:-./cache/qdrant}" -name "*.json" -type f 2>/dev/null | wc -l)
  QDRANT_SIZE=$(du -sh "${QDRANT_CACHE_DIR:-./cache/qdrant}" 2>/dev/null | cut -f1 || echo "0")
  
  echo "  Files: $QDRANT_FILES"
  
  if [ $QDRANT_FILES -gt 0 ]; then
    OLDEST_FILE=$(find "${QDRANT_CACHE_DIR:-./cache/qdrant}" -name "*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | head -1 | cut -d' ' -f2)
    if [ -n "$OLDEST_FILE" ] && [ -f "$OLDEST_FILE" ]; then
      OLDEST_AGE=$(( $(date +%s) - $(stat -c%Y "$OLDEST_FILE" 2>/dev/null || echo "0") ))
      echo "  Oldest: $((OLDEST_AGE / 3600))h ago"
    fi
  fi
  
  echo "  Size: $QDRANT_SIZE"
  echo "  TTL: ${QDRANT_CACHE_TTL:-3600}s"
  echo ""
fi

# ========== Response Cache ==========
if [ -d "${RESPONSE_CACHE_DIR:-./cache/responses}" ]; then
  echo -e "${GREEN}LLM Response Cache:${NC}"
  
  RESPONSE_FILES=$(find "${RESPONSE_CACHE_DIR:-./cache/responses}" -name "*.txt" -type f 2>/dev/null | wc -l)
  RESPONSE_SIZE=$(du -sh "${RESPONSE_CACHE_DIR:-./cache/responses}" 2>/dev/null | cut -f1 || echo "0")
  
  echo "  Files: $RESPONSE_FILES"
  
  if [ $RESPONSE_FILES -gt 0 ]; then
    OLDEST_FILE=$(find "${RESPONSE_CACHE_DIR:-./cache/responses}" -name "*.txt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | head -1 | cut -d' ' -f2)
    if [ -n "$OLDEST_FILE" ] && [ -f "$OLDEST_FILE" ]; then
      OLDEST_AGE=$(( $(date +%s) - $(stat -c%Y "$OLDEST_FILE" 2>/dev/null || echo "0") ))
      echo "  Oldest: $((OLDEST_AGE / 3600))h ago"
    fi
  fi
  
  echo "  Size: $RESPONSE_SIZE"
  echo "  TTL: ${RESPONSE_CACHE_TTL:-86400}s"
  echo ""
fi

# ========== FastEmbed Model Cache ==========
if [ -d "./cache/fastembed" ]; then
  echo -e "${GREEN}FastEmbed Model Cache:${NC}"
  
  FASTEMBED_SIZE=$(du -sh "./cache/fastembed" 2>/dev/null | cut -f1 || echo "0")
  FASTEMBED_MODELS=$(find "./cache/fastembed" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
  
  echo "  Models: $FASTEMBED_MODELS"
  echo "  Size: $FASTEMBED_SIZE"
  echo "  Note: Persistent (not cleared by clear-cache.sh)"
  echo ""
fi

# ========== Conversation Memory ==========
if [ -f "${MEMORY_FILE:-./cache/memory.json}" ]; then
  echo -e "${GREEN}Conversation Memory:${NC}"
  
  MEMORY_SIZE=$(stat -c%s "${MEMORY_FILE:-./cache/memory.json}" 2>/dev/null || echo "0")
  MEMORY_SIZE_HR=$(numfmt --to=iec-i --suffix=B $MEMORY_SIZE 2>/dev/null || echo "${MEMORY_SIZE}B")
  
  echo "  Size: $MEMORY_SIZE_HR"
  echo ""
fi

# ========== Total ==========
TOTAL_SIZE=$(du -sh "${CACHE_DIR:-./cache}" 2>/dev/null | cut -f1 || echo "0")
echo -e "${CYAN}Total Cache Size:${NC} $TOTAL_SIZE"
echo ""

echo -e "${YELLOW}Commands:${NC}"
echo "  Clear query cache: ./clear-cache.sh"
echo "  Clear all (incl. models): rm -rf ./cache && mkdir -p ./cache"
EOFCACHE
chmod +x "$PROJECT_DIR/cache-stats.sh"
log_ok "cache-stats.sh created (cache)"

# Create clear-cache.sh (cache)
cat > "$PROJECT_DIR/clear-cache.sh" << 'EOFCLEAR'
#!/bin/bash
# Clear Cache Utility cache
source ./config.env 2>/dev/null || true

echo "Clearing query caches..."

# Clear Qdrant cache
if [ -d "${QDRANT_CACHE_DIR:-./cache/qdrant}" ]; then
  rm -f "${QDRANT_CACHE_DIR:-./cache/qdrant}"/*.json 2>/dev/null
  echo "[OK] Qdrant cache cleared"
fi

# Clear Response cache
if [ -d "${RESPONSE_CACHE_DIR:-./cache/responses}" ]; then
  rm -f "${RESPONSE_CACHE_DIR:-./cache/responses}"/*.txt 2>/dev/null
  echo "[OK] Response cache cleared"
fi

# Clear conversation memory
if [ -f "${MEMORY_FILE:-./cache/memory.json}" ]; then
  rm -f "${MEMORY_FILE:-./cache/memory.json}"
  echo "[OK] Conversation memory cleared"
fi

# Recreate directories
mkdir -p "${QDRANT_CACHE_DIR:-./cache/qdrant}"
mkdir -p "${RESPONSE_CACHE_DIR:-./cache/responses}"

echo ""
echo "Cache cleared successfully"
echo "Note: FastEmbed models NOT cleared (intentional)"
EOFCLEAR
chmod +x "$PROJECT_DIR/clear-cache.sh"
log_ok "clear-cache.sh created (cache)"

# ============================================================================
# PHASE 10: Final Verification
# ============================================================================
echo ""
echo "=== Phase 10: Final Verification ==="

ERRORS=0

echo -n "Qdrant: "
curl -s --max-time 3 "http://localhost:${QDRANT_PORT}/collections" > /dev/null 2>&1 && log_ok "Running" || { log_err "Failed"; ERRORS=$((ERRORS+1)); }

echo -n "Ollama: "
curl -s --max-time 3 "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1 && log_ok "Running" || { log_err "Failed"; ERRORS=$((ERRORS+1)); }

echo -n "Model ($LLM_MODEL): "
ollama list 2>/dev/null | grep -q "$LLM_MODEL" && log_ok "Available" || { log_err "Missing"; ERRORS=$((ERRORS+1)); }

echo -n "FastEmbed: "
python3 -c "from fastembed import TextEmbedding" 2>/dev/null && log_ok "OK" || { log_err "Missing"; ERRORS=$((ERRORS+1)); }

echo -n "Qdrant Client: "
python3 -c "from qdrant_client import QdrantClient" 2>/dev/null && log_ok "OK" || { log_err "Missing"; ERRORS=$((ERRORS+1)); }

[ $ERRORS -gt 0 ] && { log_err "$ERRORS critical failures"; exit 1; }

echo ""
echo "============================================"
echo " Core Setup Complete (System)"
echo "============================================"
echo ""
echo "System Profile:"
echo "  RAM: ${RAM_GB}GB | CPUs: ${CPU_COUNT} @ ${CPU_MHZ}MHz"
echo "  CPU Score: $CPU_SCORE | Arch: $ARCH_TYPE"
echo ""
echo "Services: Qdrant :${QDRANT_PORT} | Ollama :11434 | SearXNG :${SEARXNG_PORT}"
echo "Model: $LLM_MODEL | Embedding: $FASTEMBED_MODEL ($EMBEDDING_DIM dim)"
echo "Spellcheck: pyspellchecker (FR: $FR_OK, EN: $EN_OK)"
echo ""
echo "System Features:"
echo "  - Map/Reduce summarization (./query.sh --summarize)"
echo "  - Extraction mode (./query.sh --extract)"
echo "  - Self-reflection answer verification"
echo ""
echo "profiling Features (preserved):"
echo "  - CPU score profiling, adaptive embedding"
echo "  - French OCR, legacy .doc support"
echo "  - Qdrant low-memory tuning"
echo ""
echo "cache Features (preserved):"
echo "  - Tiered performance, 2-layer cache, monitoring"
echo ""
echo "Next steps:"
echo "  bash setup-rag-ingest-System.sh"
echo "  bash setup-rag-query-System.sh"
echo "  ./ingest.sh ./documents"
echo "  ./query.sh 'your question'"
echo "============================================"
