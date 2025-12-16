#!/bin/bash
# ============================================================================
# RAG System v44 - Complete Fresh Installation Script
# ============================================================================
# This script performs a complete fresh installation of the RAG system
# with all CRAG fixes included.
#
# Usage:
#   ./fresh-install.sh [installation_directory]
#
# Example:
#   ./fresh-install.sh /root
#   ./fresh-install.sh /home/user/rag
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m'

log_ok() { echo -e "[${GREEN}✓${NC}] $1"; }
log_warn() { echo -e "[${YELLOW}!${NC}] $1"; }
log_error() { echo -e "[${RED}✗${NC}] $1"; }
log_info() { echo -e "[${BLUE}i${NC}] $1"; }

# Get installation directory
INSTALL_DIR="${1:-/root}"

echo "============================================================================"
echo "   RAG System v44 - Fresh Installation"
echo "============================================================================"
echo ""
log_info "Installation directory: $INSTALL_DIR"
echo ""

# Verify we're in the Rag4DietPI directory
if [ ! -f "setup-rag-core-v44.sh" ]; then
    log_error "Must run from Rag4DietPI directory!"
    log_info "Example: cd ~/Rag4DietPI && ./fresh-install.sh /root"
    exit 1
fi

# Check prerequisites
log_info "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    log_error "Python3 not found! Install: apt install python3 python3-pip python3-venv"
    exit 1
fi
log_ok "Python3 installed"

# Check Docker
if ! command -v docker &> /dev/null; then
    log_warn "Docker not found - SearXNG will not be available"
    log_info "To install Docker: curl -fsSL https://get.docker.com | sh"
    SKIP_SEARXNG=true
else
    log_ok "Docker installed"
    SKIP_SEARXNG=false
fi

echo ""
log_info "Step 1/7: Running core setup..."
./setup-rag-core-v44.sh "$INSTALL_DIR"
log_ok "Core setup complete"

echo ""
log_info "Step 2/7: Running ingest setup..."
./setup-rag-ingest-v44.sh "$INSTALL_DIR"
log_ok "Ingest setup complete"

echo ""
log_info "Step 3/7: Running query setup (with CRAG fixes)..."
./setup-rag-query-v44.sh "$INSTALL_DIR"
log_ok "Query setup complete (includes all CRAG fixes)"

echo ""
if [ "$SKIP_SEARXNG" = false ]; then
    log_info "Step 4/7: Installing SearXNG..."

    # Check if SearXNG already running
    if docker ps | grep -q searxng; then
        log_ok "SearXNG already running"
    else
        # Remove old container if exists
        docker rm -f searxng 2>/dev/null || true

        # Start SearXNG
        docker run -d \
            --name searxng \
            -p 8085:8080 \
            -e SEARXNG_BASE_URL=http://localhost:8085/ \
            --restart unless-stopped \
            searxng/searxng:latest

        log_ok "SearXNG started"
        log_info "Waiting 20 seconds for SearXNG startup..."
        sleep 20

        # Test SearXNG
        if curl -s "http://localhost:8085/search?q=test&format=json" > /dev/null; then
            log_ok "SearXNG is responding"
        else
            log_warn "SearXNG may not be ready yet (try testing manually later)"
        fi
    fi
else
    log_warn "Step 4/7: Skipping SearXNG (Docker not available)"
    log_info "CRAG web search will not work without SearXNG"
fi

echo ""
log_info "Step 5/7: Creating documents directory..."
cd "$INSTALL_DIR"
mkdir -p documents
log_ok "Documents directory created: $INSTALL_DIR/documents"

echo ""
log_warn "Step 6/7: Document ingestion"
echo ""
echo "  You need to add your documents to: $INSTALL_DIR/documents/"
echo ""
echo "  Options:"
echo "    A) Copy your documents now and run: cd $INSTALL_DIR && ./ingest.sh"
echo "    B) Download sample documents (Harry Potter chapters):"
echo "       wget -P $INSTALL_DIR/documents/ https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%201%20-%20The%20Philosopher's%20Stone.txt"
echo ""
read -p "  Do you want to download sample Harry Potter text for testing? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Downloading sample document..."
    wget -q -O "$INSTALL_DIR/documents/harry-potter-sample.txt" \
        "https://raw.githubusercontent.com/formcept/whiteboard/master/nbviewer/notebooks/data/harrypotter/Book%201%20-%20The%20Philosopher's%20Stone.txt" \
        2>/dev/null || {
        log_warn "Download failed, skipping sample data"
    }

    if [ -f "$INSTALL_DIR/documents/harry-potter-sample.txt" ]; then
        log_ok "Sample document downloaded"
        log_info "Running ingestion..."
        cd "$INSTALL_DIR"
        ./ingest.sh
        log_ok "Ingestion complete"
    fi
else
    log_info "Skipping sample data - add your documents manually and run: cd $INSTALL_DIR && ./ingest.sh"
fi

echo ""
log_info "Step 7/7: Verification"
cd "$INSTALL_DIR"

# Check services
./status.sh

echo ""
echo "============================================================================"
echo "   Installation Complete!"
echo "============================================================================"
echo ""
log_ok "Installation directory: $INSTALL_DIR"
echo ""
echo "Next steps:"
echo ""
echo "  1. Add documents (if not done already):"
echo "     cp /path/to/your/files/* $INSTALL_DIR/documents/"
echo ""
echo "  2. Run ingestion:"
echo "     cd $INSTALL_DIR && ./ingest.sh"
echo ""
echo "  3. Test queries:"
echo "     cd $INSTALL_DIR"
echo "     ./query.sh \"your question\""
echo "     ./query.sh --debug --full \"test CRAG with unknown term\""
echo ""
echo "  4. Test CRAG specifically:"
echo "     ./query.sh --debug --full \"Alticap\""
echo "     (Should show CRAG triggering web search)"
echo ""
echo "Available commands:"
echo "  ./query.sh [options] \"query\"   - Ask questions"
echo "  ./ingest.sh                     - Ingest/re-ingest documents"
echo "  ./status.sh                     - Check system status"
echo "  ./verify.sh                     - Run verification tests"
echo ""
echo "Query options:"
echo "  --debug                         - Show detailed debug output"
echo "  --full                          - Enable all features (CRAG, reranking, etc.)"
echo "  --clear-cache                   - Clear query cache"
echo ""
if [ "$SKIP_SEARXNG" = true ]; then
    log_warn "SearXNG not installed - CRAG web search will not work"
    echo "  To install Docker: curl -fsSL https://get.docker.com | sh"
    echo "  Then run: docker run -d --name searxng -p 8085:8080 searxng/searxng:latest"
    echo ""
fi
echo "Documentation:"
echo "  - Fresh install guide: ~/Rag4DietPI/FRESH-INSTALL-GUIDE.md"
echo "  - Fix guide: ~/Rag4DietPI/FIX-ALTICAP-QUERY.md"
echo "  - Main README: ~/Rag4DietPI/README-v44.md"
echo ""
log_ok "All done!"
echo "============================================================================"
