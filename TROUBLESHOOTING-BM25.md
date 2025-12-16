# BM25 Search Issue - Troubleshooting Guide

## Problem Description

When running queries, you may see this in debug output:

```
[HYBRID] Vector search returned 10 results
[HYBRID] BM25 search returned 0 results
```

This causes:
- **Keyword queries fail**: Terms like "Alticap" don't match via BM25
- **Hybrid search degrades**: Falls back to vector-only search
- **Irrelevant results**: Vector search alone may retrieve wrong content
- **Low grounding scores**: Retrieved content doesn't match the query

## Root Cause

The **BM25 index file is missing**: `cache/bm25_index.pkl`

The BM25 search function silently fails when this file doesn't exist, returning an empty result set. This can happen when:

1. The ingestion process didn't complete successfully
2. The `rebuild-bm25.sh` script was never created or run
3. The cache directory was cleared but index not rebuilt
4. The setup process was interrupted

## Diagnosis

Run the diagnostic script:

```bash
cd /root  # Or wherever your RAG system is installed
./diagnose-bm25.sh
```

Expected output if BM25 is missing:

```
============================================================
BM25 Diagnostic Tool
============================================================

[1] Checking Qdrant status...
✓ Qdrant is running
  Documents in collection: 1234

[2] Checking BM25 index...
✗ BM25 index NOT FOUND: cache/bm25_index.pkl
  This explains why BM25 search returns 0 results

[3] Checking rebuild script...
✗ rebuild-bm25.sh NOT FOUND
  Will be created by running fix script

============================================================
Summary
============================================================
ISSUE: BM25 index is missing

Impact:
  - BM25 keyword search returns 0 results
  - Hybrid search falls back to vector-only
  - Keyword queries (like 'Alticap') may retrieve wrong content

Solution:
  Run: ./rebuild-bm25.sh
```

## Solution

### Step 1: Get the rebuild script

If `rebuild-bm25.sh` doesn't exist, pull the latest changes:

```bash
cd ~/Rag4DietPI
git pull origin claude/debug-rag-query-F6HAr
cp rebuild-bm25.sh /root/
chmod +x /root/rebuild-bm25.sh
```

Or download from GitHub:
```bash
cd /root
curl -O https://raw.githubusercontent.com/olivierolejniczak/Rag4DietPI/claude/debug-rag-query-F6HAr/rebuild-bm25.sh
chmod +x rebuild-bm25.sh
```

### Step 2: Rebuild the BM25 index

```bash
cd /root
./rebuild-bm25.sh
```

Expected output:

```
============================================================
BM25 Index Rebuild
============================================================
Qdrant: http://localhost:6333
Collection: documents

Fetching documents from Qdrant...
✓ Fetched 1234 documents

Building BM25 index...
✓ Prepared 1234 documents for indexing

Building BM25Okapi index...

Saving index to cache/bm25_index.pkl...
✓ BM25 index built successfully
  Documents: 1234
  Index size: 2.45 MB

============================================================
Building vocabulary for spell correction...
============================================================
✓ Vocabulary saved: 10000 words
  Path: cache/vocabulary.json

============================================================
✓ BM25 rebuild complete!
============================================================
```

### Step 3: Verify the fix

Clear the query cache and test:

```bash
./query.sh --clear-cache
./query.sh --debug --full "Alticap"
```

You should now see:

```
[HYBRID] Vector search returned 10 results
[HYBRID] BM25 search returned 5 results   ← BM25 working!
[HYBRID] Top 5 after RRF fusion + filename boost
```

## Understanding the BM25 Index

### What is BM25?

BM25 (Best Matching 25) is a keyword-based search algorithm that:
- Finds exact term matches
- Ranks by term frequency and document frequency
- Works like traditional search engines
- Complements semantic vector search

### How the index is built

1. **Fetch documents** from Qdrant (all chunks)
2. **Tokenize** text:
   - Convert to lowercase
   - Extract words (alphanumeric sequences)
   - Remove stopwords (the, a, is, etc.)
   - Remove short tokens (< 3 characters)
3. **Build BM25Okapi** index from tokenized corpus
4. **Save** to `cache/bm25_index.pkl` with document IDs

### Tokenization example

Input text:
```
"Harry, this is Professor Quirrell"
```

Tokenization:
```
harry, this, professor, quirrell
          ↓ (remove stopword)
harry, professor, quirrell
```

Query "Alticap" → tokenized to `["alticap"]` → searches BM25 index

## Prevention

### Ensure BM25 is built during ingestion

When running the ingestion script:

```bash
./ingest.sh
```

Check the output includes:

```
✓ BM25 index building
```

If you see errors or warnings, BM25 may not have been built.

### Verify after ingestion

Always verify BM25 was created:

```bash
ls -lh cache/bm25_index.pkl
```

Should show a file (typically 1-10 MB depending on corpus size).

### Re-run setup if needed

If `rebuild-bm25.sh` is missing, the ingestion setup may be incomplete:

```bash
cd ~/Rag4DietPI
./setup-rag-ingest-v44.sh
```

This will create `rebuild-bm25.sh` in your project directory.

## Technical Details

### BM25 search code location

File: `/root/query.sh` (created by `setup-rag-query-v44.sh`)

Function: `bm25_search()` (around line 418-440)

```python
def bm25_search(query, top_k=10):
    """BM25 keyword search"""
    try:
        with open("cache/bm25_index.pkl", "rb") as f:
            data = pickle.load(f)

        bm25 = data["bm25"]
        doc_ids = data["doc_ids"]

        # Tokenize query (same as indexing)
        tokens = re.findall(r'\b\w+\b', query.lower())
        stopwords = {'the', 'a', 'an', ...}
        tokens = [t for t in tokens if len(t) > 2 and t not in stopwords]

        if not tokens:
            return []

        scores = bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)),
                           key=lambda i: scores[i],
                           reverse=True)[:top_k]

        return [(doc_ids[i], float(scores[i]))
                for i in top_indices if scores[i] > 0]
    except Exception as e:
        return []  # Silent failure if index missing
```

**Note**: The function silently returns `[]` on any exception, including missing file.

### Hybrid search logic

File: `/root/query.sh`

Function: `hybrid_search()` (around line 525-600)

```python
# Vector search (always runs)
vector_results = vector_search(query_embedding, top_k=50)

# BM25 search (may return empty if index missing)
bm25_results = bm25_search(query, top_k=50)

# RRF fusion - combines both results
# If BM25 returns 0 results, only vector results are used
```

### Dependencies

BM25 requires:
- Python package: `rank-bm25`
- Qdrant running with documents indexed
- Write access to `cache/` directory

Check if installed:
```bash
python3 -c "from rank_bm25 import BM25Okapi; print('✓ rank-bm25 installed')"
```

If not installed:
```bash
pip3 install rank-bm25
```

## Common Questions

### Q: Why doesn't BM25 error visibly when index is missing?

A: The `bm25_search()` function uses a try/except block that returns an empty list on any error. This is by design to allow graceful degradation - if BM25 fails, vector search continues to work.

However, this makes the issue harder to diagnose without debug output.

### Q: Can I use vector search only?

A: Yes, but you'll lose keyword matching benefits:
- Exact term matches (names, codes, IDs)
- Rare terms that embeddings might miss
- Better performance on specific entity queries

Hybrid search (BM25 + Vector) is recommended for best results.

### Q: How often should I rebuild BM25?

A: Rebuild after:
- Adding new documents via ingestion
- Clearing the cache
- Updating the document collection

The index is static - it doesn't auto-update when Qdrant changes.

### Q: What if rebuild fails?

Common issues:

1. **Qdrant not running**:
   ```bash
   systemctl status qdrant
   # Or
   docker ps | grep qdrant
   ```

2. **No documents in Qdrant**:
   ```bash
   curl http://localhost:6333/collections/documents | python3 -m json.tool
   ```

   If `points_count` is 0, run ingestion first.

3. **Permission issues**:
   ```bash
   mkdir -p cache
   chmod 755 cache
   ```

4. **rank-bm25 not installed**:
   ```bash
   pip3 install rank-bm25
   ```

## Summary

**Problem**: BM25 index missing → BM25 returns 0 results → Hybrid search fails

**Solution**: Run `./rebuild-bm25.sh` to rebuild the index from Qdrant data

**Prevention**: Ensure ingestion completes successfully and verify `cache/bm25_index.pkl` exists

For more help, see:
- `./diagnose-bm25.sh` - Automated diagnostics
- `./query.sh --debug` - See BM25 result counts
- `README-v44.md` - Full system documentation
