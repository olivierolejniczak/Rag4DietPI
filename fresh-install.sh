#!/bin/bash
# ============================================================================
# RAG System v44 - Complete Fresh Installation Script
# ============================================================================
# This script performs a complete fresh installation of the RAG system
# with all CRAG fixes included.
#
# Usage Option 1 - Download and install in one command:
#   bash <(curl -s https://raw.githubusercontent.com/olivierolejniczak/Rag4DietPI/claude/debug-rag-query-F6HAr/fresh-install.sh)
#
# Usage Option 2 - Clone first, then run:
#   git clone https://github.com/olivierolejniczak/Rag4DietPI.git
#   cd Rag4DietPI
#   git checkout claude/debug-rag-query-F6HAr
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

# Check if we need to clone the repository
if [ ! -f "setup-rag-core-v44.sh" ]; then
    log_info "Repository not found, cloning from GitHub..."

    # Check git is installed
    if ! command -v git &> /dev/null; then
        log_error "Git not found! Install: apt install git"
        exit 1
    fi

    # Clone to temporary directory in user's home
    REPO_DIR="$HOME/Rag4DietPI"

    if [ -d "$REPO_DIR" ]; then
        log_info "Using existing repository at $REPO_DIR"
        cd "$REPO_DIR"
        git fetch origin
        git checkout claude/debug-rag-query-F6HAr
        # Reset any local changes or deleted files
        git reset --hard origin/claude/debug-rag-query-F6HAr
        log_ok "Repository updated and cleaned"
    else
        log_info "Cloning to $REPO_DIR..."
        git clone https://github.com/olivierolejniczak/Rag4DietPI.git "$REPO_DIR"
        cd "$REPO_DIR"
        git checkout claude/debug-rag-query-F6HAr
    fi

    log_ok "Repository ready at $REPO_DIR"
    echo ""
fi

# Now verify we have the setup scripts
if [ ! -f "setup-rag-core-v44.sh" ]; then
    log_error "Setup scripts not found! Cannot continue."
    exit 1
fi

SCRIPT_DIR="$(pwd)"
log_ok "Using scripts from: $SCRIPT_DIR"
echo ""

# Check prerequisites
log_info "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    log_error "Python3 not found! Install: apt install python3 python3-pip python3-venv"
    exit 1
fi
log_ok "Python3 installed"

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
log_info "Step 4/7: Installing DuckDuckGo search library..."
cd "$INSTALL_DIR"
if [ -d "./venv" ]; then
    source ./venv/bin/activate
    pip install -q duckduckgo-search
    log_ok "DuckDuckGo search installed (CRAG web search enabled)"
else
    log_warn "No venv found, skipping DuckDuckGo installation"
    log_info "CRAG web search will not work without duckduckgo-search"
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
echo "Documentation:"
echo "  - Fresh install guide: $SCRIPT_DIR/FRESH-INSTALL-GUIDE.md"
echo "  - Fix guide: $SCRIPT_DIR/FIX-ALTICAP-QUERY.md"
echo "  - Main README: $SCRIPT_DIR/README-v44.md"
echo ""
echo "Repository location: $SCRIPT_DIR"
echo ""
log_ok "All done!"
echo "============================================================================"
