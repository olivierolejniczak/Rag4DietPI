#!/bin/bash
# ============================================================================
# RAG System v44 - Core Setup
# Features: SearXNG self-hosted, Docker auto-deploy, systemd service
# ============================================================================

set -e

echo "============================================================================"
echo "   RAG System v44 - Core Setup"
echo "============================================================================"
echo ""

PROJECT_DIR="${1:-$(pwd)}"
cd "$PROJECT_DIR"

# ============================================================================
# Utility Functions
# ============================================================================
log_ok() { echo "[OK] $1"; }
log_err() { echo "[ERROR] $1" >&2; }
log_info() { echo "[INFO] $1"; }

# ============================================================================
# System Detection
# ============================================================================
RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
RAM_GB=$((RAM_KB / 1024 / 1024))
CPU_COUNT=$(nproc)

echo "System: ${RAM_GB}GB RAM, ${CPU_COUNT} CPUs"
echo ""

# ============================================================================
# Profile Selection Based on RAM
# ============================================================================
if [ "$RAM_GB" -lt 4 ]; then
    PROFILE="minimal"
    LLM_MODEL="qwen2.5:1.5b"  # 1.5b works with swap, 0.5b hallucinates
    CHUNK_SIZE=400
    TOP_K=5
    LLM_TIMEOUT=180
    ULTRAFAST_TIMEOUT=90
    RERANK_DEFAULT=false
    HYDE_DEFAULT=false
elif [ "$RAM_GB" -lt 8 ]; then
    PROFILE="balanced"
    LLM_MODEL="qwen2.5:1.5b"
    CHUNK_SIZE=500
    TOP_K=5
    LLM_TIMEOUT=180
    ULTRAFAST_TIMEOUT=90
    RERANK_DEFAULT=false
    HYDE_DEFAULT=false
elif [ "$RAM_GB" -lt 16 ]; then
    PROFILE="performance"
    LLM_MODEL="qwen2.5:3b"
    CHUNK_SIZE=600
    TOP_K=6
    LLM_TIMEOUT=240
    ULTRAFAST_TIMEOUT=120
    RERANK_DEFAULT=true
    HYDE_DEFAULT=false
else
    PROFILE="full"
    LLM_MODEL="qwen2.5:7b"
    CHUNK_SIZE=800
    TOP_K=8
    LLM_TIMEOUT=300
    ULTRAFAST_TIMEOUT=150
    RERANK_DEFAULT=true
    HYDE_DEFAULT=true
fi

echo "Selected profile: $PROFILE"
echo "  LLM Model: $LLM_MODEL"
echo "  Chunk Size: $CHUNK_SIZE"
echo ""

# ============================================================================
# Swap Management (critical for low-RAM systems)
# ============================================================================
echo "Checking swap..."
CURRENT_SWAP=$(free -m | awk '/^Swap:/ {print $2}')
RECOMMENDED_SWAP=$((8 - RAM_GB))  # Aim for ~8GB total (RAM + swap)
[ "$RECOMMENDED_SWAP" -lt 2 ] && RECOMMENDED_SWAP=2

if [ "$CURRENT_SWAP" -lt "$((RECOMMENDED_SWAP * 1024 - 500))" ]; then
    echo "  Current swap: ${CURRENT_SWAP}MB, Recommended: ${RECOMMENDED_SWAP}GB"
    
    # Try to create swap if not enough
    if [ ! -f /swapfile ] && [ "$RAM_GB" -lt 8 ]; then
        echo "  Creating ${RECOMMENDED_SWAP}GB swap file..."
        
        # Check available disk space
        AVAILABLE_GB=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
        if [ "$AVAILABLE_GB" -gt "$((RECOMMENDED_SWAP + 5))" ]; then
            fallocate -l ${RECOMMENDED_SWAP}G /swapfile 2>/dev/null || dd if=/dev/zero of=/swapfile bs=1M count=$((RECOMMENDED_SWAP * 1024)) status=progress
            chmod 600 /swapfile
            mkswap /swapfile
            swapon /swapfile
            
            # Make persistent
            if ! grep -q '/swapfile' /etc/fstab; then
                echo '/swapfile none swap sw 0 0' >> /etc/fstab
            fi
            
            log_ok "Swap created: ${RECOMMENDED_SWAP}GB"
        else
            echo "  [WARN] Not enough disk space for swap"
        fi
    elif [ -f /swapfile ]; then
        # Swapfile exists but might not be active
        if ! swapon --show | grep -q '/swapfile'; then
            swapon /swapfile 2>/dev/null || true
        fi
        log_ok "Swap file exists"
    else
        log_ok "Swap: ${CURRENT_SWAP}MB (sufficient)"
    fi
else
    log_ok "Swap: ${CURRENT_SWAP}MB"
fi

# Optimize swappiness for RAG workload
CURRENT_SWAPPINESS=$(cat /proc/sys/vm/swappiness)
if [ "$CURRENT_SWAPPINESS" -gt 30 ]; then
    echo "  Optimizing swappiness (${CURRENT_SWAPPINESS} -> 10)..."
    sysctl -w vm.swappiness=10 > /dev/null
    echo "vm.swappiness=10" > /etc/sysctl.d/99-rag-swap.conf 2>/dev/null || true
fi

echo ""

# ============================================================================
# Install Python3 (if not present)
# ============================================================================
echo "Checking Python..."

if ! command -v python3 &> /dev/null; then
    log_info "Python3 not found. Installing..."
    
    if command -v apt-get &> /dev/null; then
        apt-get update -qq
        apt-get install -y -qq python3 python3-pip python3-venv
    elif command -v apk &> /dev/null; then
        apk add --quiet python3 py3-pip
    elif command -v dnf &> /dev/null; then
        dnf install -y -q python3 python3-pip
    elif command -v yum &> /dev/null; then
        yum install -y -q python3 python3-pip
    else
        log_err "Cannot install Python3. Please install manually."
        exit 1
    fi
fi

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    log_ok "Python: $PYTHON_VERSION"
else
    log_err "Python3 installation failed"
    exit 1
fi

# ============================================================================
# Install Python Packages
# ============================================================================
echo "Installing Python packages..."

# For Python 3.11+ we need --break-system-packages
# Also suppress root user warning
PIP_FLAGS="--break-system-packages --root-user-action=ignore"

# Test if flags are supported (older pip versions)
if ! pip3 install --help 2>&1 | grep -q "break-system-packages"; then
    PIP_FLAGS=""
fi

# Don't upgrade pip itself (causes issues with system-managed Python)
pip3 install --quiet requests urllib3 $PIP_FLAGS 2>/dev/null || true
pip3 install --quiet pypdfium2 python-docx openpyxl python-pptx $PIP_FLAGS 2>/dev/null || true
pip3 install --quiet pdfplumber $PIP_FLAGS 2>/dev/null || true
pip3 install --quiet rank-bm25 $PIP_FLAGS 2>/dev/null || { log_err "rank-bm25 required"; exit 1; }
pip3 install --quiet flashrank $PIP_FLAGS 2>/dev/null || log_info "FlashRank optional (reranking disabled)"
pip3 install --quiet numpy $PIP_FLAGS 2>/dev/null || true
pip3 install --quiet pillow pytesseract $PIP_FLAGS 2>/dev/null || log_info "OCR optional"

log_ok "Python packages installed"

# ============================================================================
# Check/Install Ollama
# ============================================================================
echo "Checking Ollama..."

if ! command -v ollama &> /dev/null; then
    log_info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama if not running
if ! pgrep -x "ollama" > /dev/null 2>&1; then
    log_info "Starting Ollama..."
    ollama serve &>/dev/null &
    sleep 5
fi

log_ok "Ollama ready"

# Pull required models
echo "Checking models..."

if ! ollama list 2>/dev/null | grep -q "nomic-embed-text"; then
    log_info "Pulling embedding model..."
    ollama pull nomic-embed-text
fi
log_ok "Embedding model: nomic-embed-text"

if ! ollama list 2>/dev/null | grep -q "$LLM_MODEL"; then
    log_info "Pulling LLM model: $LLM_MODEL..."
    ollama pull "$LLM_MODEL"
fi
log_ok "LLM model: $LLM_MODEL"

# ============================================================================
# Install Docker if not present
# ============================================================================
echo "Checking Docker..."
if ! command -v docker &> /dev/null; then
    log_info "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    log_ok "Docker installed"
else
    log_ok "Docker already installed"
fi

# ============================================================================
# Install Docker Compose if not present
# ============================================================================
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    log_info "Installing Docker Compose..."
    apt-get update && apt-get install -y docker-compose-plugin || {
        # Fallback to standalone
        curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
    }
    log_ok "Docker Compose installed"
else
    log_ok "Docker Compose already installed"
fi

# ============================================================================
# Create Directory Structure
# ============================================================================
echo ""
echo "Creating directory structure..."
mkdir -p "$PROJECT_DIR"/{documents,cache,lib,searxng,logs}
log_ok "Directories created"

# ============================================================================
# Create SearXNG Configuration
# ============================================================================
echo "Creating SearXNG configuration..."

cat > "$PROJECT_DIR/searxng/settings.yml" << 'EOFSEARX'
# SearXNG Settings - RAG v44
# Only non-commercial, privacy-respecting engines

use_default_settings: false

general:
  instance_name: "RAG-SearXNG"
  debug: false
  privacypolicy_url: false
  donation_url: false
  contact_url: false
  enable_metrics: false

search:
  safe_search: 0
  autocomplete: ""
  default_lang: "en"
  max_page: 0
  formats:
    - html
    - json

server:
  port: 8080
  bind_address: "0.0.0.0"
  secret_key: "rag-searxng-secret-key-change-me"
  limiter: false
  public_instance: false

ui:
  static_use_hash: true
  default_theme: simple
  results_on_new_tab: false

# WHITELISTED ENGINES ONLY
# NO commercial engines: google, bing, yahoo, yandex, baidu, qwant
engines:
  # Wikipedia / Knowledge
  - name: wikipedia
    engine: wikipedia
    shortcut: wp
    disabled: false
    weight: 1.5

  - name: wikidata
    engine: wikidata
    shortcut: wd
    disabled: false
    weight: 1.2

  # Technical / Programming
  - name: stackoverflow
    engine: stackoverflow
    shortcut: so
    disabled: false
    weight: 1.5

  - name: stackexchange
    engine: stackexchange
    shortcut: se
    disabled: false
    sites: superuser.com,serverfault.com,askubuntu.com,unix.stackexchange.com
    weight: 1.3

  - name: mdn
    engine: mdn
    shortcut: mdn
    disabled: false
    weight: 1.4

  # Archives / Libraries
  - name: archive
    engine: archive_is
    shortcut: ar
    disabled: false
    weight: 1.0

  - name: openlibrary
    engine: openlibrary
    shortcut: ol
    disabled: false
    weight: 1.0

  # Linux Documentation
  - name: debian
    engine: debian
    shortcut: deb
    disabled: false
    categories: [it, packages]
    weight: 1.2

  - name: archlinux
    engine: arch_linux_wiki
    shortcut: arch
    disabled: false
    weight: 1.3

  - name: gentoo
    engine: gentoo
    shortcut: gt
    disabled: false
    weight: 1.1

  # DuckDuckGo Lite (non-tracking version)
  - name: duckduckgo_lite
    engine: duckduckgo
    shortcut: ddg
    disabled: false
    weight: 1.0

# Explicitly disable all commercial engines
  - name: google
    disabled: true
  - name: bing
    disabled: true
  - name: yahoo
    disabled: true
  - name: yandex
    disabled: true
  - name: baidu
    disabled: true
  - name: qwant
    disabled: true

outgoing:
  request_timeout: 10.0
  max_request_timeout: 15.0
  useragent_suffix: ""
  pool_connections: 100
  pool_maxsize: 10
EOFSEARX

log_ok "SearXNG settings.yml created"

# ============================================================================
# Create Docker Compose File
# ============================================================================
echo "Creating docker-compose.yml..."

cat > "$PROJECT_DIR/docker-compose.yml" << 'EOFDOCKER'
services:
  # Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    container_name: rag-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_data:/qdrant/storage
    restart: unless-stopped
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334

  # SearXNG - Privacy-respecting metasearch
  searxng:
    image: searxng/searxng:latest
    container_name: rag-searxng
    ports:
      - "8085:8080"
    volumes:
      - ./searxng:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=http://localhost:8085/
    restart: unless-stopped
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - SETGID
      - SETUID

volumes:
  qdrant_data:
EOFDOCKER

log_ok "docker-compose.yml created"

# ============================================================================
# Create Systemd Service for SearXNG
# ============================================================================
echo "Creating systemd service..."

cat > /etc/systemd/system/rag-services.service << EOFSVC
[Unit]
Description=RAG System Services (Qdrant + SearXNG)
Requires=docker.service
After=docker.service network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$PROJECT_DIR
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=120

[Install]
WantedBy=multi-user.target
EOFSVC

systemctl daemon-reload
systemctl enable rag-services.service
log_ok "Systemd service created and enabled"

# ============================================================================
# Create config.env
# ============================================================================
echo "Creating config.env..."

cat > "$PROJECT_DIR/config.env" << EOFCFG
# ============================================================================
# RAG System v44 Configuration
# Profile: $PROFILE (${RAM_GB}GB RAM, ${CPU_COUNT} CPUs)
# Generated: $(date)
# ============================================================================

# =========================
# SERVICES
# =========================
OLLAMA_HOST=http://localhost:11434
QDRANT_HOST=http://localhost:6333
COLLECTION_NAME=documents

# =========================
# MODELS
# =========================
LLM_MODEL=$LLM_MODEL
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

# =========================
# CHUNKING
# =========================
CHUNK_SIZE=$CHUNK_SIZE
CHUNK_OVERLAP=80
MIN_CHUNK_SIZE=100
MAX_CHUNK_SIZE=1200
USE_SEMANTIC_CHUNKING=true
USE_SECTIONS=true
CONTEXTUAL_HEADERS=true
PARENT_CHILD_ENABLED=true

# =========================
# RETRIEVAL
# =========================
DEFAULT_TOP_K=$TOP_K
HYBRID_SEARCH_ENABLED=true
HYBRID_ALPHA=0.5
BM25_K1=1.5
BM25_B=0.75

# =========================
# QUERY ENHANCEMENT
# =========================
QUERY_CLASSIFICATION_ENABLED=false
QUERY_REWRITE_ENABLED=false
HYDE_ENABLED=$HYDE_DEFAULT
SUBQUERY_ENABLED=false
STEPBACK_ENABLED=false
SPELL_CORRECTION_ENABLED=true

# =========================
# POST-RETRIEVAL
# =========================
RERANK_ENABLED=$RERANK_DEFAULT
RERANK_MODEL=ms-marco-MiniLM-L-12-v2
RERANK_TOP_K=10
RELEVANCE_FILTER_ENABLED=true
RELEVANCE_THRESHOLD=0.001
DIVERSITY_FILTER_ENABLED=true
DIVERSITY_THRESHOLD=0.85

# =========================
# CRAG (Corrective RAG)
# =========================
CRAG_ENABLED=true
CRAG_MIN_RELEVANCE=0.40
CRAG_WEB_FALLBACK=true

# =========================
# WEB SEARCH - SearXNG ONLY
# =========================
WEB_SEARCH_PROVIDER=searxng
WEB_SEARCH_ENABLED=true
WEB_SEARCH_MODE=auto
SEARXNG_URL=http://localhost:8085/search
SEARXNG_TIMEOUT=10
SEARXNG_MAX_RESULTS=5
SEARXNG_ALLOWED_ENGINES=wikipedia,wikidata,stackexchange,stackoverflow,archive,openlibrary,mdn,debian,archlinux,gentoo,duckduckgo_lite

# =========================
# GENERATION
# =========================
CITATIONS_ENABLED=false
GROUNDING_CHECK_ENABLED=false

# =========================
# MEMORY
# =========================
MEMORY_ENABLED=true
MEMORY_MAX_TURNS=3
MEMORY_FILE=cache/memory.json

# =========================
# TIMEOUTS (seconds)
# Default mode: reasonable timeouts
# Full mode: NO timeouts (set to 0 = unlimited)
# =========================
LLM_TIMEOUT_DEFAULT=$LLM_TIMEOUT
LLM_TIMEOUT_ULTRAFAST=$ULTRAFAST_TIMEOUT
LLM_TIMEOUT_FULL=0
EMBEDDING_TIMEOUT=60
EMBEDDING_TIMEOUT_FULL=0
RERANK_TIMEOUT=60
RERANK_TIMEOUT_FULL=0
SEARXNG_TIMEOUT=10
SEARXNG_TIMEOUT_FULL=30

# =========================
# CONTEXT SIZE
# =========================
MAX_CONTEXT_CHARS=5000
MAX_CONTEXT_CHARS_FULL=15000
MAX_CHUNK_CHARS=1000
MAX_MEMORY_CHARS=500

# =========================
# GENERATION PARAMETERS
# =========================
NUM_PREDICT_DEFAULT=500
NUM_PREDICT_ULTRAFAST=150
NUM_PREDICT_FULL=2000
TEMPERATURE=0.2

# =========================
# CACHE
# =========================
CACHE_ENABLED=true
CACHE_DIR=./cache
QUERY_CACHE_ENABLED=true
QUERY_CACHE_TTL=3600
DEDUP_ENABLED=true

# =========================
# QUALITY FEEDBACK LOOP (v44)
# =========================
QUALITY_LEDGER_ENABLED=true
QUALITY_LEDGER_FILE=cache/quality_ledger.sqlite
QUALITY_LEDGER_FORMAT=sqlite

# Confidence thresholds (0-1)
RETRIEVAL_CONFIDENCE_MIN=0.3
ANSWER_COVERAGE_MIN=0.2
GROUNDING_SCORE_MIN=0.4

# Abstention behavior
ABSTENTION_ENABLED=true
ABSTENTION_MESSAGE="Je n'ai pas assez d'informations fiables pour répondre avec confiance."

# Human feedback
FEEDBACK_ENABLED=true

# =========================
# DEBUG
# =========================
DEBUG=false
DEBUG_WEB=false
DEBUG_RETRIEVAL=false
DEBUG_LLM=false

# =========================
# DIRECTORIES
# =========================
DOCUMENTS_DIR=./documents
EOFCFG

log_ok "config.env created"

# ============================================================================
# Create Helper Scripts
# ============================================================================
echo "Creating helper scripts..."

# start-services.sh
cat > "$PROJECT_DIR/start-services.sh" << 'EOFSTART'
#!/bin/bash
# Start all RAG services

echo "Starting RAG services..."

# Start Ollama
if command -v ollama &> /dev/null; then
    if ! pgrep -x "ollama" > /dev/null; then
        echo "Starting Ollama..."
        ollama serve &>/dev/null &
        sleep 3
    fi
    echo "[OK] Ollama running"
else
    echo "[WARN] Ollama not found"
fi

# Start Docker services (Qdrant + SearXNG)
echo "Starting Docker services..."
docker compose up -d

# Wait for services
echo "Waiting for services to be ready..."
sleep 5

# Check Qdrant
if curl -s http://localhost:6333/health | grep -q "ok"; then
    echo "[OK] Qdrant ready"
else
    echo "[WAIT] Qdrant starting..."
    sleep 5
fi

# Check SearXNG
if curl -s "http://localhost:8085/search?q=test&format=json" | grep -q "results"; then
    echo "[OK] SearXNG ready"
else
    echo "[WAIT] SearXNG starting..."
    sleep 5
fi

echo ""
echo "All services started!"
EOFSTART
chmod +x "$PROJECT_DIR/start-services.sh"

# stop-services.sh
cat > "$PROJECT_DIR/stop-services.sh" << 'EOFSTOP'
#!/bin/bash
# Stop all RAG services

echo "Stopping RAG services..."
docker compose down
pkill -f "ollama serve" 2>/dev/null || true
echo "Services stopped."
EOFSTOP
chmod +x "$PROJECT_DIR/stop-services.sh"

# status.sh
cat > "$PROJECT_DIR/status.sh" << 'EOFSTATUS'
#!/bin/bash
# Check status of all RAG services

source ./config.env 2>/dev/null

echo "============================================"
echo "   RAG System v44 - Service Status"
echo "============================================"
echo ""

# Ollama
echo -n "Ollama:   "
if curl -s "${OLLAMA_HOST:-http://localhost:11434}/api/tags" > /dev/null 2>&1; then
    MODELS=$(curl -s "${OLLAMA_HOST:-http://localhost:11434}/api/tags" | grep -o '"name":"[^"]*"' | head -3 | cut -d'"' -f4 | tr '\n' ', ')
    echo "✓ Running (models: ${MODELS%,})"
else
    echo "✗ Not running"
fi

# Qdrant
echo -n "Qdrant:   "
if curl -s "${QDRANT_HOST:-http://localhost:6333}/health" | grep -q "ok"; then
    COLLECTIONS=$(curl -s "${QDRANT_HOST:-http://localhost:6333}/collections" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | tr '\n' ', ')
    echo "✓ Running (collections: ${COLLECTIONS%,})"
else
    echo "✗ Not running"
fi

# SearXNG
echo -n "SearXNG:  "
SEARXNG_URL="${SEARXNG_URL:-http://localhost:8085/search}"
if curl -s "${SEARXNG_URL}?q=test&format=json" 2>/dev/null | grep -q "results"; then
    echo "✓ Running at ${SEARXNG_URL%/search}"
else
    echo "✗ Not running"
fi

# Docker
echo -n "Docker:   "
if docker ps --format '{{.Names}}' | grep -E "rag-|searxng|qdrant" > /dev/null 2>&1; then
    CONTAINERS=$(docker ps --format '{{.Names}}' | grep -E "rag-|searxng|qdrant" | tr '\n' ', ')
    echo "✓ Containers: ${CONTAINERS%,}"
else
    echo "○ No RAG containers running"
fi

echo ""
echo "Config: LLM=$LLM_MODEL, Embedding=$EMBEDDING_MODEL"
echo "============================================"
EOFSTATUS
chmod +x "$PROJECT_DIR/status.sh"

# verify.sh
cat > "$PROJECT_DIR/verify.sh" << 'EOFVERIFY'
#!/bin/bash
# Verify RAG system installation

source ./config.env 2>/dev/null

echo "============================================"
echo "   RAG System v44 - Verification"
echo "============================================"
echo ""

ERRORS=0

# Check Ollama
echo -n "Checking Ollama... "
if curl -s "${OLLAMA_HOST:-http://localhost:11434}/api/tags" > /dev/null 2>&1; then
    echo "✓"
else
    echo "✗ FAILED"
    ((ERRORS++))
fi

# Check Qdrant
echo -n "Checking Qdrant... "
if curl -s "${QDRANT_HOST:-http://localhost:6333}/health" | grep -q "ok"; then
    echo "✓"
else
    echo "✗ FAILED"
    ((ERRORS++))
fi

# Check SearXNG
echo -n "Checking SearXNG... "
SEARXNG_BASE="${SEARXNG_URL:-http://localhost:8085/search}"
if curl -s "${SEARXNG_BASE}?q=test&format=json" 2>/dev/null | grep -q "results"; then
    echo "✓"
else
    echo "✗ FAILED"
    ((ERRORS++))
fi

# Check LLM model
echo -n "Checking LLM model ($LLM_MODEL)... "
if curl -s "${OLLAMA_HOST:-http://localhost:11434}/api/tags" | grep -q "$LLM_MODEL"; then
    echo "✓"
else
    echo "✗ NOT FOUND (run: ollama pull $LLM_MODEL)"
    ((ERRORS++))
fi

# Check embedding model
echo -n "Checking embedding model ($EMBEDDING_MODEL)... "
if curl -s "${OLLAMA_HOST:-http://localhost:11434}/api/tags" | grep -q "$EMBEDDING_MODEL"; then
    echo "✓"
else
    echo "✗ NOT FOUND (run: ollama pull $EMBEDDING_MODEL)"
    ((ERRORS++))
fi

# Check Python
echo -n "Checking Python... "
if command -v python3 &> /dev/null; then
    echo "✓ ($(python3 --version))"
else
    echo "✗ NOT FOUND"
    ((ERRORS++))
fi

# Check required Python packages
echo -n "Checking Python packages... "
MISSING=""
python3 -c "import requests" 2>/dev/null || MISSING="$MISSING requests"
python3 -c "import rank_bm25" 2>/dev/null || MISSING="$MISSING rank_bm25"
if [ -z "$MISSING" ]; then
    echo "✓"
else
    echo "✗ Missing:$MISSING"
    ((ERRORS++))
fi

# Check directories
echo -n "Checking directories... "
if [ -d "./documents" ] && [ -d "./cache" ] && [ -d "./lib" ]; then
    echo "✓"
else
    echo "✗ Missing directories"
    ((ERRORS++))
fi

echo ""
if [ $ERRORS -eq 0 ]; then
    echo "✓ All checks passed!"
else
    echo "✗ $ERRORS check(s) failed"
fi
echo "============================================"
exit $ERRORS
EOFVERIFY
chmod +x "$PROJECT_DIR/verify.sh"

log_ok "Helper scripts created"

# ============================================================================
# Start Services
# ============================================================================
echo ""
echo "Starting services..."
cd "$PROJECT_DIR"
docker compose up -d

# Wait for services
echo "Waiting for services to initialize..."
sleep 10

# Verify SearXNG
echo -n "Verifying SearXNG... "
for i in {1..6}; do
    if curl -s "http://localhost:8085/search?q=test&format=json" | grep -q "results"; then
        echo "✓ Ready"
        break
    fi
    sleep 5
    echo -n "."
done

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================================"
echo "   Core Setup Complete!"
echo "============================================================================"
echo ""
echo "Profile: $PROFILE"
echo "LLM Model: $LLM_MODEL"
echo "Embedding: nomic-embed-text (768 dim)"
echo ""
echo "Services:"
echo "  • Qdrant:   http://localhost:6333"
echo "  • SearXNG:  http://localhost:8085"
echo "  • Ollama:   http://localhost:11434"
echo ""
echo "SearXNG Engines (whitelisted only):"
echo "  wikipedia, wikidata, stackoverflow, stackexchange,"
echo "  mdn, archive, openlibrary, debian, archlinux, gentoo"
echo ""
echo "Next steps:"
echo "  1. bash setup-rag-ingest-v44.sh"
echo "  2. bash setup-rag-query-v44.sh"
echo "  3. ./ingest.sh"
echo "  4. ./query.sh 'your question'"
echo ""
echo "============================================================================"
