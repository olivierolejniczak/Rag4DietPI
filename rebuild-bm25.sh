#!/bin/bash

# BM25 Index Rebuild Script
# Creates BM25 index from Qdrant documents

set -e

QDRANT="${QDRANT_HOST:-http://localhost:6333}"
COLLECTION="${COLLECTION_NAME:-documents}"

echo "============================================================"
echo "BM25 Index Rebuild"
echo "============================================================"
echo "Qdrant: $QDRANT"
echo "Collection: $COLLECTION"
echo ""

python3 << 'EOFPY'
import json
import requests
import os
import pickle
import re

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("✗ Error: rank-bm25 not installed")
    print("  Install with: pip3 install rank-bm25")
    exit(1)

QDRANT = os.environ.get("QDRANT_HOST", "http://localhost:6333")
COLLECTION = os.environ.get("COLLECTION_NAME", "documents")

print("Fetching documents from Qdrant...")

# Fetch all documents
try:
    resp = requests.post(f"{QDRANT}/collections/{COLLECTION}/points/scroll", json={
        "limit": 10000,
        "with_payload": True,
        "with_vector": False
    }, timeout=30)

    if resp.status_code != 200:
        print(f"✗ Error fetching documents: HTTP {resp.status_code}")
        exit(1)

    data = resp.json()
    points = data.get("result", {}).get("points", [])

    if not points:
        print("✗ No documents found in Qdrant")
        exit(1)

    print(f"✓ Fetched {len(points)} documents")

except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Tokenize function (must match query.sh tokenization)
def tokenize(text):
    """
    Tokenize text for BM25.
    IMPORTANT: Must use same tokenization as query.sh
    """
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)

    # Stopwords (keep in sync with query.sh)
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'shall',
        'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
        'because', 'until', 'while', 'this', 'that', 'these', 'those'
    }

    # Remove short tokens and stopwords
    tokens = [t for t in tokens if len(t) > 2 and t not in stopwords]
    return tokens

print("\nBuilding BM25 index...")

# Build corpus
corpus = []
doc_ids = []
doc_texts = {}
empty_docs = 0

for p in points:
    text = p.get("payload", {}).get("text", "")
    if text and text.strip():
        tokens = tokenize(text)
        if tokens:
            corpus.append(tokens)
            doc_ids.append(p["id"])
            doc_texts[p["id"]] = text
        else:
            empty_docs += 1
    else:
        empty_docs += 1

if not corpus:
    print("✗ No valid documents for BM25 (all empty or no text)")
    exit(1)

if empty_docs > 0:
    print(f"⚠️  Skipped {empty_docs} empty documents")

print(f"✓ Prepared {len(corpus)} documents for indexing")

# Build BM25 index
print("\nBuilding BM25Okapi index...")
bm25 = BM25Okapi(corpus)

# Save index
os.makedirs("cache", exist_ok=True)
index_path = "cache/bm25_index.pkl"

print(f"\nSaving index to {index_path}...")
with open(index_path, "wb") as f:
    pickle.dump({
        "bm25": bm25,
        "doc_ids": doc_ids,
        "corpus": corpus,
        "doc_texts": doc_texts
    }, f)

# Get file size
import os
size_bytes = os.path.getsize(index_path)
size_mb = size_bytes / (1024 * 1024)

print(f"✓ BM25 index built successfully")
print(f"  Documents: {len(doc_ids)}")
print(f"  Index size: {size_mb:.2f} MB")

# Build vocabulary for spell correction
print("\n" + "="*60)
print("Building vocabulary for spell correction...")
print("="*60)

from collections import Counter
import json

vocab = Counter()
for tokens in corpus:
    vocab.update(tokens)

# Save top 10000 words
vocab_dict = {word: count for word, count in vocab.most_common(10000)}
vocab_path = "cache/vocabulary.json"

with open(vocab_path, "w") as f:
    json.dump(vocab_dict, f)

print(f"✓ Vocabulary saved: {len(vocab_dict)} words")
print(f"  Path: {vocab_path}")

print("\n" + "="*60)
print("✓ BM25 rebuild complete!")
print("="*60)
EOFPY

echo ""
echo "You can now run queries with BM25 keyword search enabled"
echo ""
