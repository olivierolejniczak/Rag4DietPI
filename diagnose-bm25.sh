#!/bin/bash

echo "============================================================"
echo "BM25 Diagnostic Tool"
echo "============================================================"

# Check if running from /root
if [ "$PWD" != "/root" ]; then
    echo "⚠️  Warning: Should run from /root directory"
    echo "   Current: $PWD"
fi

echo ""
echo "[1] Checking Qdrant status..."
QDRANT_STATUS=$(curl -sf http://localhost:6333/collections/documents 2>/dev/null)
if [ $? -eq 0 ]; then
    POINTS=$(echo "$QDRANT_STATUS" | python3 -c "import sys, json; print(json.load(sys.stdin).get('result', {}).get('points_count', 0))" 2>/dev/null)
    echo "✓ Qdrant is running"
    echo "  Documents in collection: $POINTS"
else
    echo "✗ Qdrant not accessible at http://localhost:6333"
    exit 1
fi

echo ""
echo "[2] Checking BM25 index..."
if [ -f "cache/bm25_index.pkl" ]; then
    SIZE=$(ls -lh cache/bm25_index.pkl | awk '{print $5}')
    echo "✓ BM25 index exists: cache/bm25_index.pkl ($SIZE)"

    # Try to load and check it
    python3 << 'EOFPY'
import pickle
try:
    with open("cache/bm25_index.pkl", "rb") as f:
        data = pickle.load(f)
    print(f"  Documents in BM25 index: {len(data.get('doc_ids', []))}")
    if len(data.get('doc_ids', [])) == 0:
        print("  ⚠️  Index is EMPTY - needs rebuild")
except Exception as e:
    print(f"  ✗ Error loading index: {e}")
    print("  ⚠️  Index is CORRUPTED - needs rebuild")
EOFPY
else
    echo "✗ BM25 index NOT FOUND: cache/bm25_index.pkl"
    echo "  This explains why BM25 search returns 0 results"
fi

echo ""
echo "[3] Checking rebuild script..."
if [ -f "rebuild-bm25.sh" ]; then
    echo "✓ rebuild-bm25.sh exists"
else
    echo "✗ rebuild-bm25.sh NOT FOUND"
    echo "  Will be created by running fix script"
fi

echo ""
echo "[4] Checking vocabulary.json..."
if [ -f "cache/vocabulary.json" ]; then
    SIZE=$(ls -lh cache/vocabulary.json | awk '{print $5}')
    echo "✓ Vocabulary exists: cache/vocabulary.json ($SIZE)"
else
    echo "⚠️  Vocabulary not found (used for spell correction)"
fi

echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"

if [ ! -f "cache/bm25_index.pkl" ]; then
    echo "ISSUE: BM25 index is missing"
    echo ""
    echo "Impact:"
    echo "  - BM25 keyword search returns 0 results"
    echo "  - Hybrid search falls back to vector-only"
    echo "  - Keyword queries (like 'Alticap') may retrieve wrong content"
    echo ""
    echo "Solution:"
    echo "  Run: ./rebuild-bm25.sh"
    echo "  Or: ./fix-bm25.sh (if rebuild script missing)"
else
    echo "✓ BM25 system appears functional"
fi

echo ""
