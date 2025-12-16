#!/bin/bash

# SearXNG Diagnostic and Setup Helper

SEARXNG_URL="${SEARXNG_URL:-http://localhost:8085}"
SEARCH_ENDPOINT="${SEARXNG_URL}/search"

echo "============================================================"
echo "SearXNG Diagnostic & Setup"
echo "============================================================"
echo "SearXNG URL: $SEARXNG_URL"
echo ""

# Check if SearXNG is running
echo "[1] Checking if SearXNG is accessible..."
if curl -sf "${SEARCH_ENDPOINT}?q=test&format=json" 2>/dev/null | grep -q "results"; then
    echo "✓ SearXNG is RUNNING and accessible"
    echo ""
    echo "Testing a sample search..."
    RESULT=$(curl -sf "${SEARCH_ENDPOINT}?q=test+query&format=json" 2>/dev/null)
    NUM_RESULTS=$(echo "$RESULT" | python3 -c "import sys, json; print(len(json.load(sys.stdin).get('results', [])))" 2>/dev/null || echo "0")
    echo "  Sample query returned: $NUM_RESULTS results"

    if [ "$NUM_RESULTS" -gt 0 ]; then
        echo ""
        echo "✓ SearXNG is working correctly"
        echo "  CRAG web search should function properly"
        exit 0
    else
        echo "⚠️  SearXNG responded but returned 0 results"
        echo "  This might indicate configuration issues"
    fi
else
    echo "✗ SearXNG is NOT accessible"
fi

echo ""
echo "============================================================"
echo "Impact"
echo "============================================================"
echo "Without SearXNG:"
echo "  - CRAG (Corrective RAG) cannot perform web fallback"
echo "  - Queries for content not in your database will fail"
echo "  - Low-confidence retrievals won't trigger web search"
echo ""
echo "Example: Query 'Alticap' (not in database)"
echo "  - With BM25 + Vector: Retrieves irrelevant results"
echo "  - CRAG detects low quality (score < 0.4)"
echo "  - CRAG tries web search → FAILS (SearXNG not running)"
echo "  - User gets wrong answer with low confidence"
echo ""

# Check Docker
echo "============================================================"
echo "Setup Options"
echo "============================================================"
echo ""
echo "[Option 1] Docker (Recommended)"
echo "-----------------------------------------------------------"

if command -v docker &> /dev/null; then
    echo "✓ Docker is installed"

    # Check if container exists
    if docker ps -a --format '{{.Names}}' | grep -q "searxng"; then
        STATUS=$(docker ps --filter "name=searxng" --format '{{.Status}}')
        if [ -n "$STATUS" ]; then
            echo "⚠️  SearXNG container exists and is running but not accessible"
            echo "  Container status: $STATUS"
            echo "  Check port mapping: docker ps | grep searxng"
            echo ""
            echo "  Try restarting:"
            echo "    docker restart searxng"
        else
            echo "⚠️  SearXNG container exists but is stopped"
            echo ""
            echo "  Start it with:"
            echo "    docker start searxng"
        fi
    else
        echo "SearXNG container not found"
        echo ""
        echo "Quick start (single command):"
        echo "-----------------------------------------------------------"
        echo "docker run -d \\"
        echo "  --name searxng \\"
        echo "  -p 8085:8080 \\"
        echo "  --restart unless-stopped \\"
        echo "  searxng/searxng:latest"
        echo ""
        echo "After starting, wait 10-15 seconds then test:"
        echo "  curl \"http://localhost:8085/search?q=test&format=json\""
        echo ""
        echo "Or use this script:"
        echo "  ./start-searxng.sh  (if available)"
    fi
else
    echo "✗ Docker is not installed"
    echo ""
    echo "Install Docker first:"
    echo "  curl -fsSL https://get.docker.com | sh"
    echo "  usermod -aG docker \$USER"
fi

echo ""
echo "[Option 2] Manual Installation"
echo "-----------------------------------------------------------"
echo "See: https://docs.searxng.org/admin/installation.html"
echo ""
echo "For DietPi/Debian:"
echo "  1. Install dependencies"
echo "  2. Clone SearXNG repo"
echo "  3. Configure and run"
echo ""
echo "(Docker method is much simpler)"
echo ""

echo "============================================================"
echo "Configuration"
echo "============================================================"
echo ""
echo "SearXNG settings in config.env:"
echo "  SEARXNG_URL=$SEARXNG_URL/search"
echo "  SEARXNG_TIMEOUT=10"
echo "  SEARXNG_ALLOWED_ENGINES=wikipedia,duckduckgo,..."
echo ""
echo "CRAG settings in config.env:"
echo "  CRAG_ENABLED=true"
echo "  CRAG_THRESHOLD=0.4  (lower = more aggressive web search)"
echo ""

echo "============================================================"
echo "Alternative: Disable CRAG"
echo "============================================================"
echo ""
echo "If you don't need web search fallback, disable CRAG:"
echo ""
echo "In config.env:"
echo "  CRAG_ENABLED=false"
echo ""
echo "Or use command-line:"
echo "  CRAG_ENABLED=false ./query.sh \"your query\""
echo ""
echo "Note: Without CRAG, queries for content not in your"
echo "database will return low-confidence or irrelevant results."
echo ""

echo "============================================================"
echo "Testing After Setup"
echo "============================================================"
echo ""
echo "1. Check SearXNG is running:"
echo "   ./check-searxng.sh"
echo ""
echo "2. Test CRAG with debug:"
echo "   ./query.sh --debug --full \"search term not in database\""
echo ""
echo "3. Look for these lines in output:"
echo "   [CRAG] Retrieval quality evaluation:"
echo "   [CRAG] Score: 0.1234 | Threshold: 0.4"
echo "   [CRAG] Decision: ✗ INSUFFICIENT - triggering web search"
echo "   [WEB SEARCH] Got 3 results from web"
echo ""

exit 1
