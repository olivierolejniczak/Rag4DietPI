#!/bin/bash

# ============================================================================
# RAG System v44 - Core Setup
# Features: SearXNG self-hosted, Docker auto-deploy, systemd service
# ============================================================================

set -e

echo "============================================================================"
echo " RAG System v44 - Core Setup"
echo "============================================================================"
echo ""

PROJECT_DIR="${1:-$(pwd)}"
cd "$PROJECT_DIR"

# ============================================================================
# Utility Functions
# ============================================================================

log_ok()   { echo "[OK] $1"; }
log_err()  { echo "[ERROR] $1" >&2; }
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
    LLM_MODEL="qwen2.5:1.5b" # 1.5b works with swap, 0.5b hallucinates
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
echo " LLM Model: $LLM_MODEL"
echo " Chunk Size: $CHUNK_SIZE"
echo ""

# ============================================================================
# Swap Management (critical for low-RAM systems)
# ============================================================================

echo "Checking swap..."
CURRENT_SWAP=$(free -m | awk '/^Swap:/ {print $2}')
RECOMMENDED_SWAP=$((8 - RAM_GB)) # Aim for ~8GB total (RAM + swap)
[ "$RECOMMENDED_SWAP" -lt 2 ] && RECOMMENDED_SWAP=2

if [ "$CURRENT_SWAP" -lt "$((RECOMMENDED_SWAP * 1024 - 500))" ]; then
    echo " Current swap: ${CURRENT_SWAP}MB, Recommended: ${RECOMMENDED_SWAP}GB"
    # Try to create swap if not enough
    if [ ! -f /swapfile ] && [ "$RAM_GB" -lt 8 ]; then
        echo " Creating ${RECOMMENDED_SWAP}GB swap file..."
        # Check available disk space
        AVAILABLE_GB=$(df -BG / | awk 'NR==2 {print $4}' | tr -d 'G')
        if [ "$AVAILABLE_GB" -gt "$((RECOMMENDED_SWAP + 5))" ]; then
            fallocate -l ${RECOMMENDED_SWAP}G /swapfile 2>/dev/null || \
                dd if=/dev/zero of=/swapfile bs=1M count=$((RECOMMENDED_SWAP * 1024)) status=progress
            chmod 600 /swapfile
            mkswap /swapfile
            swapon /swapfile
            # Make persistent
            if ! grep -q '/swapfile' /etc/fstab; then
                echo '/swapfile none swap sw 0 0' >> /etc/fstab
            fi
            log_ok "Swap created: ${RECOMMENDED_SWAP}GB"
        else
            echo " [WARN] Not enough disk space for swap"
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
    echo " Optimizing swappiness (${CURRENT_SWAPPINESS} -> 10)..."
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

PIP_FLAGS="--break-system-packages --root-user-action=ignore"
if ! pip3 install --help 2>&1 | grep -q "break-system-packages"; then
    PIP_FLAGS=""
fi

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
        curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
            -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
    }
    log_ok "Docker Compose installed"
else
    log_ok "Docker Compose already installed"
fi

# Normalize compose command wrapper (v2 plugin or standalone)
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE_CMD=(docker-compose)
else
    DOCKER_COMPOSE_CMD=(docker compose)
fi

# ============================================================================
# Create Directory Structure
# ============================================================================

echo ""
echo "Creating directory structure..."
mkdir -p "$PROJECT_DIR"/{documents,cache,lib,searxng,logs}
log_ok "Directories created"

# ============================================================================
# SearXNG Configuration (integrated patch)
# ============================================================================

configure_searxng() {
    log_info "[SearXNG] Ensure ./searxng config directory exists"
    mkdir -p ./searxng

    log_info "[SearXNG] Write minimal ./searxng/settings.yml"
    cat > ./searxng/settings.yml << 'EOF'
use_default_settings: true

server:
  # base_url comes from SEARXNG_BASE_URL env (docker-compose.yml)
  secret_key: "change_this_secret_key"
  limiter: false
  image_proxy: true

search:
  # Allow HTML UI and JSON API
  formats:
    - html
    - json
EOF

    log_info "[SearXNG] Ensure ./searxng/searxng-privacy.html exists"
    if [ ! -s ./searxng/searxng-privacy.html ]; then
        cat > ./searxng/searxng-privacy.html << 'EOF'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Privacy Policy - Local SearXNG</title>
  </head>
  <body>
    <h1>Privacy Policy</h1>
    <p>This private SearXNG instance is used only for local RAG experimentation.</p>
    <p>No user-identifiable logs are intentionally kept beyond default SearXNG behavior.</p>
  </body>
</html>
EOF
        log_ok "[SearXNG] Created default searxng-privacy.html"
    else
        log_info "[SearXNG] Existing searxng-privacy.html detected, leaving as is"
    fi
}

# ============================================================================
# Docker Compose File
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
# Systemd Service for RAG stack
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
# config.env
# ============================================================================

echo "Creating config.env..."
cat > "$PROJECT_DIR/config.env" << EOFCFG
# ============================================================================
# RAG System v44 Configuration
# Profile: $PROFILE (${RAM_GB}GB RAM, ${CPU_COUNT} CPUs)
# Generated: $(date)
# ============================================================================

# SERVICES
OLLAMA_HOST=http://localhost:11434
QDRANT_HOST=http://localhost:6333
COLLECTION_NAME=documents

# MODELS
LLM_MODEL=$LLM_MODEL
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSION=768

# CHUNKING
CHUNK_SIZE=$CHUNK_SIZE
CHUNK_OVERLAP=80
MIN_CHUNK_SIZE=100
MAX_CHUNK_SIZE=1200
USE_SEMANTIC_CHUNKING=true
USE_SECTIONS=true
CONTEXTUAL_HEADERS=true
PARENT_CHILD_ENABLED=true

# RETRIEVAL
DEFAULT_TOP_K=$TOP_K
HYBRID_SEARCH_ENABLED=true
HYBRID_ALPHA=0.5
BM25_K1=1.5
BM25_B=0.75

# QUERY ENHANCEMENT
QUERY_CLASSIFICATION_ENABLED=false
QUERY_REWRITE_ENABLED=false
HYDE_ENABLED=$HYDE_DEFAULT
SUBQUERY_ENABLED=false
STEPBACK_ENABLED=false
SPELL_CORRECTION_ENABLED=true

# POST-RETRIEVAL
RERANK_ENABLED=$RERANK_DEFAULT
RERANK_MODEL=ms-marco-MiniLM-L-12-v2
RERANK_TOP_K=10
RELEVANCE_FILTER_ENABLED=true
RELEVANCE_THRESHOLD=0.001
DIVERSITY_FILTER_ENABLED=true
DIVERSITY_THRESHOLD=0.85

# CRAG (Corrective RAG)
CRAG_ENABLED=true
CRAG_MIN_RELEVANCE=0.40
CRAG_WEB_FALLBACK=true

# WEB SEARCH - SearXNG ONLY
WEB_SEARCH_PROVIDER=searxng
WEB_SEARCH_ENABLED=true
WEB_SEARCH_MODE=auto
SEARXNG_URL=http://localhost:8085/search
SEARXNG_TIMEOUT=10
SEARXNG_MAX_RESULTS=5
SEARXNG_ALLOWED_ENGINES=wikipedia,wikidata,stackexchange,stackoverflow,archive,openlibrary,mdn,debian,archlinux,gentoo,duckduckgo_lite

# GENERATION
CITATIONS_ENABLED=false
GROUNDING_CHECK_ENABLED=false

# MEMORY
MEMORY_ENABLED=true
MEMORY_MAX_TURNS=3
MEMORY_FILE=cache/memory.json

# TIMEOUTS
LLM_TIMEOUT_DEFAULT=$LLM_TIMEOUT
LLM_TIMEOUT_ULTRAFAST=$ULTRAFAST_TIMEOUT
LLM_TIMEOUT_FULL=0
EMBEDDING_TIMEOUT=60
EMBEDDING_TIMEOUT_FULL=0
RERANK_TIMEOUT=60
RERANK_TIMEOUT_FULL=0
SEARXNG_TIMEOUT=10
SEARXNG_TIMEOUT_FULL=30

# CONTEXT SIZE
MAX_CONTEXT_CHARS=5000
MAX_CONTEXT_CHARS_FULL=15000
MAX_CHUNK_CHARS=1000
MAX_MEMORY_CHARS=500

# GENERATION PARAMETERS
NUM_PREDICT_DEFAULT=500
NUM_PREDICT_ULTRAFAST=150
NUM_PREDICT_FULL=2000
TEMPERATURE=0.2

# CACHE
CACHE_ENABLED=true
CACHE_DIR=./cache
QUERY_CACHE_ENABLED=true
QUERY_CACHE_TTL=3600
DEDUP_ENABLED=true

# QUALITY FEEDBACK LOOP (v44)
QUALITY_LEDGER_ENABLED=true
QUALITY_LEDGER_FILE=cache/quality_ledger.sqlite
QUALITY_LEDGER_FORMAT=sqlite
RETRIEVAL_CONFIDENCE_MIN=0.3
ANSWER_COVERAGE_MIN=0.2
GROUNDING_SCORE_MIN=0.4
ABSTENTION_ENABLED=true
ABSTENTION_MESSAGE="Je n'ai pas assez d'informations fiables pour répondre avec confiance."
FEEDBACK_ENABLED=true

# DEBUG
DEBUG=false
DEBUG_WEB=false
DEBUG_RETRIEVAL=false
DEBUG_LLM=false

# DIRECTORIES
DOCUMENTS_DIR=./documents
EOFCFG

log_ok "config.env created"

# ============================================================================
# Helper Scripts (start/stop/status/verify) – keep as-is or adjust later
# ============================================================================

echo "Creating helper scripts..."

# start-services.sh
cat > "$PROJECT_DIR/start-services.sh" << 'EOFSTART'
#!/bin/bash
echo "Starting RAG services..."

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

echo "Starting Docker services..."
docker compose up -d
echo "Waiting for services to be ready..."
sleep 5

echo "[INFO] Basic start complete (detailed health via ./status.sh)"
EOFSTART
chmod +x "$PROJECT_DIR/start-services.sh"

# stop-services.sh
cat > "$PROJECT_DIR/stop-services.sh" << 'EOFSTOP'
#!/bin/bash
echo "Stopping RAG services..."
docker compose down
pkill -f "ollama serve" 2>/dev/null || true
echo "Services stopped."
EOFSTOP
chmod +x "$PROJECT_DIR/stop-services.sh"

# status.sh
cat > "$PROJECT_DIR/status.sh" << 'EOFSTATUS'
#!/bin/bash
source ./config.env 2>/dev/null

echo "============================================"
echo " RAG System v44 - Service Status"
echo "============================================"
echo ""

# Ollama
echo -n "Ollama: "
if curl -s "${OLLAMA_HOST:-http://localhost:11434}/api/tags" > /dev/null 2>&1; then
  MODELS=$(curl -s "${OLLAMA_HOST:-http://localhost:11434}/api/tags" | grep -o '"name":"[^"]*"' | head -3 | cut -d'"' -f4 | tr '\n' ', ')
  echo "✓ Running (models: ${MODELS%,})"
else
  echo "✗ Not running"
fi

# Qdrant via /collections
echo -n "Qdrant: "
QDRANT_URL="${QDRANT_HOST:-http://localhost:6333}"
HTTP_CODE_QD="$(curl -s -o /dev/null -w "%{http_code}" "${QDRANT_URL}/collections" || true)"
if [ "$HTTP_CODE_QD" = "200" ]; then
  COLLECTIONS=$(curl -s "${QDRANT_URL}/collections" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | tr '\n' ', ')
  echo "✓ Running (collections: ${COLLECTIONS%,})"
else
  echo "✗ Not running"
fi

# SearXNG JSON
echo -n "SearXNG: "
SEARXNG_URL_RUNTIME="${SEARXNG_URL:-http://localhost:8085/search}"
HTTP_CODE_SX="$(curl -s -o /dev/null -w "%{http_code}" "${SEARXNG_URL_RUNTIME}?q=test&format=json" || true)"
if [ "$HTTP_CODE_SX" = "200" ]; then
  echo "✓ Running at ${SEARXNG_URL_RUNTIME%/search}"
else
  echo "✗ Not running"
fi

# Docker
echo -n "Docker: "
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

# verify.sh (kept mostly as-is, but Qdrant/SearXNG checks can be updated similarly)
cat > "$PROJECT_DIR/verify.sh" << 'EOFVERIFY'
#!/bin/bash
source ./config.env 2>/dev/null

echo "============================================"
echo " RAG System v44 - Verification"
echo "============================================"
echo ""

ERRORS=0

echo -n "Checking Ollama... "
if curl -s "${OLLAMA_HOST:-http://localhost:11434}/api/tags" > /dev/null 2>&1; then
  echo "✓"
else
  echo "✗ FAILED"
  ((ERRORS++))
fi

echo -n "Checking Qdrant... "
QDRANT_URL="${QDRANT_HOST:-http://localhost:6333}"
if [ "$(curl -s -o /dev/null -w "%{http_code}" "${QDRANT_URL}/collections" || true)" = "200" ]; then
  echo "✓"
else
  echo "✗ FAILED"
  ((ERRORS++))
fi

echo -n "Checking SearXNG... "
SEARXNG_BASE="${SEARXNG_URL:-http://localhost:8085/search}"
if [ "$(curl -s -o /dev/null -w "%{http_code}" "${SEARXNG_BASE}?q=test&format=json" || true)" = "200" ]; then
  echo "✓"
else
  echo "✗ FAILED"
  ((ERRORS++))
fi

echo -n "Checking LLM model ($LLM_MODEL)... "
if curl -s "${OLLAMA_HOST:-http://localhost:11434}/api/tags" | grep -q "$LLM_MODEL"; then
  echo "✓"
else
  echo "✗ NOT FOUND (run: ollama pull $LLM_MODEL)"
  ((ERRORS++))
fi

echo -n "Checking embedding model ($EMBEDDING_MODEL)... "
if curl -s "${OLLAMA_HOST:-http://localhost:11434}/api/tags" | grep -q "$EMBEDDING_MODEL"; then
  echo "✓"
else
  echo "✗ NOT FOUND (run: ollama pull $EMBEDDING_MODEL)"
  ((ERRORS++))
fi

echo -n "Checking Python... "
if command -v python3 &> /dev/null; then
  echo "✓ ($(python3 --version))"
else
  echo "✗ NOT FOUND"
  ((ERRORS++))
fi

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
# Start Services (integrated SearXNG config + compose)
# ============================================================================

echo ""
echo "Starting services..."

cd "$PROJECT_DIR"
configure_searxng

${DOCKER_COMPOSE_CMD[@]} down || true
${DOCKER_COMPOSE_CMD[@]} up -d

echo "Waiting 5 seconds for services to warm up..."
sleep 5

# Non-fatal health checks (like patch)
set +e

echo "[Health] Check Qdrant via /collections"
QDRANT_URL="${QDRANT_HOST:-http://localhost:6333}"
HTTP_CODE_QD="$(curl -s -o /dev/null -w "%{http_code}" "${QDRANT_URL}/collections" || true)"
if [ "$HTTP_CODE_QD" = "200" ]; then
  COLLECTIONS=$(curl -s "${QDRANT_URL}/collections" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | tr '\n' ', ')
  echo "    -> Qdrant OK (collections: ${COLLECTIONS%,})"
else
  echo "    -> WARNING: Qdrant /collections returned HTTP $HTTP_CODE_QD (check logs)."
fi

echo "[Health] Check SearXNG JSON endpoint"
SEARXNG_URL_RUNTIME="${SEARXNG_URL:-http://localhost:8085}"
HTTP_CODE_SX="$(curl -s -o /dev/null -w "%{http_code}" "${SEARXNG_URL_RUNTIME}/search?q=test&format=json" || true)"
if [ "$HTTP_CODE_SX" = "200" ]; then
  echo "    -> SearXNG JSON API OK at ${SEARXNG_URL_RUNTIME}"
else
  echo "    -> WARNING: SearXNG JSON API returned HTTP $HTTP_CODE_SX (check logs)."
fi

set -e

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================================================"
echo " Core Setup Complete!"
echo "============================================================================"
echo ""
echo "Profile: $PROFILE"
echo "LLM Model: $LLM_MODEL"
echo "Embedding: nomic-embed-text (768 dim)"
echo ""
echo "Services:"
echo " • Qdrant:  http://localhost:6333"
echo " • SearXNG: http://localhost:8085"
echo " • Ollama:  http://localhost:11434"
echo ""
echo "Next steps:"
echo " 1. bash setup-rag-ingest-v44.sh"
echo " 2. bash setup-rag-query-v44.sh"
echo " 3. ./ingest.sh"
echo " 4. ./query.sh 'your question'"
echo ""
echo "============================================================================"
