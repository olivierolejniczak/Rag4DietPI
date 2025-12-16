# Fix: "Alticap" Query Returns Irrelevant Results

## Executive Summary

Your query for "Alticap" returns Harry Potter dialogue instead of relevant content. This is caused by **TWO independent issues** that need to be fixed:

1. **BM25 index missing** → BM25 returns 0 results
2. **SearXNG not running** → CRAG can't trigger web search

Both need to be fixed for optimal results.

## Quick Fix (5-10 minutes)

```bash
# Step 1: Pull the fixes
cd ~/Rag4DietPI
git pull origin claude/debug-rag-query-F6HAr

# Step 2: Fix BM25
cd /root
cp ~/Rag4DietPI/rebuild-bm25.sh .
chmod +x rebuild-bm25.sh
./rebuild-bm25.sh

# Step 3: Install SearXNG (Docker required)
docker run -d --name searxng -p 8085:8080 \
  --restart unless-stopped searxng/searxng:latest
sleep 15  # Wait for startup

# Step 4: Apply CRAG debug patch
cp ~/Rag4DietPI/patch-crag-debug.sh .
chmod +x patch-crag-debug.sh
./patch-crag-debug.sh

# Step 5: Test
./query.sh --clear-cache
./query.sh --debug --full "Alticap"
```

## Understanding the Issues

### Issue 1: BM25 Index Missing

**Debug output showed:**
```
[HYBRID] Vector search returned 10 results
[HYBRID] BM25 search returned 0 results  ← Problem!
```

**What this means:**
- BM25 keyword search isn't working
- Only vector (semantic) search is running
- Exact keyword matches like "Alticap" don't work
- System falls back to semantically similar content (Harry Potter)

**Why it happened:**
- `/root/cache/bm25_index.pkl` doesn't exist
- Should have been created during ingestion
- BM25 search silently fails when index is missing

**Impact:**
- Keyword queries retrieve wrong content
- Hybrid search degrades to vector-only
- Precision suffers for exact term matching

**Fix:** Run `./rebuild-bm25.sh` to create the index

**Details:** See [TROUBLESHOOTING-BM25.md](./TROUBLESHOOTING-BM25.md)

### Issue 2: CRAG Web Search Failing

**What should happen:**
```
Query "Alticap" (not in database)
  ↓
Retrieval returns Harry Potter (irrelevant)
  ↓
CRAG detects low quality (score 0.0008 < threshold 0.4)
  ↓
CRAG triggers web search via SearXNG
  ↓
Web results for "Alticap" added to context
  ↓
LLM generates answer using web sources
  ↓
Result: Correct answer with web citations
```

**What actually happens:**
```
Query "Alticap"
  ↓
Retrieval returns Harry Potter (irrelevant)
  ↓
CRAG detects low quality → tries web search
  ↓
SearXNG not running → ConnectionError
  ↓
Web search returns [] (empty, silently)
  ↓
LLM generates answer using only Harry Potter context
  ↓
Result: Wrong answer with LOW_CONFIDENCE
```

**Why you don't see it:**
- CRAG fails gracefully (no visible error)
- Web search exception returns empty list
- Only shows error if `DEBUG_WEB=true`
- System continues with local results only

**Fix:** Install SearXNG + apply debug patch

**Details:** See [TROUBLESHOOTING-CRAG.md](./TROUBLESHOOTING-CRAG.md)

## Detailed Steps

### Step 1: Diagnose Current State

```bash
cd /root

# Check BM25
ls -lh cache/bm25_index.pkl
# Expected: "No such file or directory"

# Check SearXNG
curl -sf "http://localhost:8085/search?q=test&format=json" | head -20
# Expected: "curl: (7) Failed to connect"
```

Both missing? Continue to Step 2.

### Step 2: Rebuild BM25 Index

```bash
cd ~/Rag4DietPI
git pull origin claude/debug-rag-query-F6HAr

cd /root
cp ~/Rag4DietPI/rebuild-bm25.sh .
cp ~/Rag4DietPI/diagnose-bm25.sh .
chmod +x *.sh

# Diagnose first
./diagnose-bm25.sh

# Expected output:
# ✗ BM25 index NOT FOUND: cache/bm25_index.pkl

# Rebuild
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
✓ BM25 rebuild complete!
============================================================
```

Verify:
```bash
ls -lh cache/bm25_index.pkl
# Should show file size (e.g., 2.5M)
```

### Step 3: Install SearXNG

#### Option A: Docker (5 minutes, recommended)

```bash
# Check if Docker is installed
docker --version

# If not installed:
# curl -fsSL https://get.docker.com | sh

# Run SearXNG
docker run -d \
  --name searxng \
  -p 8085:8080 \
  --restart unless-stopped \
  searxng/searxng:latest

# Wait for startup
echo "Waiting 15 seconds for SearXNG to start..."
sleep 15

# Test
curl "http://localhost:8085/search?q=test&format=json" | python3 -m json.tool | head -30
```

Expected output:
```json
{
  "results": [
    {
      "title": "...",
      "url": "...",
      "content": "..."
    },
    ...
  ],
  "query": "test"
}
```

#### Option B: Check if Already Running

```bash
docker ps | grep searxng
```

If you see a container, restart it:
```bash
docker restart searxng
```

#### Option C: Use Diagnostic Script

```bash
cp ~/Rag4DietPI/check-searxng.sh /root/
chmod +x check-searxng.sh
./check-searxng.sh
```

Follow the instructions provided.

### Step 4: Apply CRAG Debug Patch

This patch adds visibility into CRAG's operation:

```bash
cd /root
cp ~/Rag4DietPI/patch-crag-debug.sh .
chmod +x patch-crag-debug.sh
./patch-crag-debug.sh
```

Expected output:
```
============================================================
CRAG Debug Patch
============================================================
...
Backing up query.sh...
✓ Backup created: query.sh.backup-crag-debug

Applying patch...
✓ Patched evaluate_retrieval_quality with debug logging
✓ Patched search_web to show errors in DEBUG mode

============================================================
✓ Patch applied successfully
============================================================
```

This adds debug output like:
```
[CRAG] Retrieval quality evaluation:
  Score: 0.0008 | Threshold: 0.4
  Decision: ✗ INSUFFICIENT - triggering web search
[WEB SEARCH] Got 3 results from web
```

### Step 5: Test the Fix

```bash
cd /root

# Clear cache to force fresh query
./query.sh --clear-cache

# Test with debug output
./query.sh --debug --full "Alticap"
```

#### Expected Output - BEFORE Fixes

```
============================================================
RAG Query v44
============================================================
Query: Alticap...
Mode: FULL | HyDE=True | Rerank=True
============================================================

  [VECTOR] Got 10 results from Qdrant
  [HYBRID] Vector search returned 10 results
  [HYBRID] BM25 search returned 0 results  ← BM25 broken

[SEARCH] Found 5 results

[POST-RETRIEVAL] 5 chunks after filtering
  [1] Dialogue.csv: see you there. Harry...

[QUALITY SCORES]
  Retrieval confidence: 0.413
  Grounding:            0.180  ← Low grounding
  Decision: low_confidence

============================================================
ANSWER
============================================================
(Wrong answer about Harry Potter)

⚠️ Confidence: LOW_CONFIDENCE
```

#### Expected Output - AFTER Fixes

```
============================================================
RAG Query v44
============================================================
Query: Alticap...
Mode: FULL | HyDE=True | Rerank=True
============================================================

  [VECTOR] Got 10 results from Qdrant
  [HYBRID] Vector search returned 10 results
  [HYBRID] BM25 search returned 5 results  ← BM25 working!

[SEARCH] Found 10 results

[CRAG] Retrieval quality evaluation:  ← CRAG debug output
  Score: 0.0012 | Threshold: 0.4
  Decision: ✗ INSUFFICIENT - triggering web search

[WEB SEARCH] Got 3 results from web  ← Web search working!

[POST-RETRIEVAL] 13 chunks after filtering

[QUALITY SCORES]
  Retrieval confidence: 0.682
  Grounding:            0.754  ← Better grounding
  Decision: confident

============================================================
ANSWER
============================================================
(Correct answer based on web results)

✅ Confidence: CONFIDENT

---
Sources:
[WEB:wikipedia] Alticap - Wikipedia
[WEB:duckduckgo] ...
```

### Step 6: Verify Both Fixes

Test BM25:
```bash
./query.sh --debug "test keyword in your database"
```

Look for:
```
[HYBRID] BM25 search returned X results  (X > 0)
```

Test CRAG:
```bash
./query.sh --debug --full "something not in database"
```

Look for:
```
[CRAG] Decision: ✗ INSUFFICIENT - triggering web search
[WEB SEARCH] Got 3 results from web
```

## Configuration

After fixing, you can tune the system in `config.env`:

```bash
# BM25 Settings
BM25_K1=1.5        # Term frequency saturation
BM25_B=0.75        # Length normalization

# CRAG Settings
CRAG_ENABLED=true         # Enable web fallback
CRAG_THRESHOLD=0.4        # Lower = more aggressive web search

# SearXNG Settings
SEARXNG_URL=http://localhost:8085/search
SEARXNG_TIMEOUT=10
SEARXNG_ALLOWED_ENGINES=wikipedia,duckduckgo,stackoverflow
```

### Tuning CRAG Threshold

| Value | Behavior | Use Case |
|-------|----------|----------|
| 0.2   | Very aggressive | Always double-check with web |
| 0.4   | Balanced (default) | Web for clear mismatches |
| 0.6   | Conservative | Only obvious failures |
| 0.8   | Very conservative | Almost never use web |

Lower = More web searches = Slower but more comprehensive

## Troubleshooting

### BM25 Still Returns 0 Results

```bash
# Check Qdrant has documents
curl -s http://localhost:6333/collections/documents | python3 -m json.tool

# Re-diagnose
./diagnose-bm25.sh

# Check rank-bm25 is installed
python3 -c "from rank_bm25 import BM25Okapi; print('OK')"
```

### SearXNG Connection Fails

```bash
# Check container is running
docker ps | grep searxng

# Check logs
docker logs searxng | tail -50

# Restart
docker restart searxng
sleep 15

# Test manually
curl "http://localhost:8085/search?q=test&format=json"
```

### CRAG Still Not Triggering

```bash
# Check debug patch applied
grep "CRAG] Retrieval quality" query.sh

# Check CRAG enabled
grep CRAG_ENABLED config.env

# Test with explicit enable
CRAG_ENABLED=true ./query.sh --debug --full "test query"
```

## Alternative: Disable Features

If you can't fix something, you can disable it:

### Disable CRAG (keep local-only RAG)

```bash
# In config.env
CRAG_ENABLED=false

# Or per query
CRAG_ENABLED=false ./query.sh "query"
```

**Impact**: No web fallback for unknown queries

### Use BM25-only (without fixing)

Not recommended, but possible:

```bash
# Disable vector search, use only BM25
# (Not actually possible with current implementation)
# Better to fix BM25 index
```

## Files Provided

| File | Purpose |
|------|---------|
| `rebuild-bm25.sh` | Rebuild BM25 index from Qdrant |
| `diagnose-bm25.sh` | Diagnose BM25 issues |
| `check-searxng.sh` | Check SearXNG status |
| `patch-crag-debug.sh` | Add CRAG debug logging |
| `TROUBLESHOOTING-BM25.md` | BM25 troubleshooting guide |
| `TROUBLESHOOTING-CRAG.md` | CRAG troubleshooting guide |
| `FIX-ALTICAP-QUERY.md` | This file |

## Summary

**Two issues identified:**
1. BM25 index missing → no keyword search
2. SearXNG not running → no web fallback

**Fixes:**
1. Run `./rebuild-bm25.sh`
2. Install SearXNG with Docker
3. Apply CRAG debug patch for visibility

**Result:**
- BM25 + Vector hybrid search works
- CRAG triggers web search for unknown content
- Debug mode shows what's happening
- Queries like "Alticap" get web-augmented answers

**Time required:** 5-10 minutes

**Benefits:**
- Accurate answers for content in database
- Web fallback for content not in database
- Visible debug info for troubleshooting
- Quality scores indicate confidence

## Next Steps

After fixing:

1. **Test with known content** (should not trigger CRAG):
   ```bash
   ./query.sh --debug "content you know is in database"
   ```

2. **Test with unknown content** (should trigger CRAG):
   ```bash
   ./query.sh --debug --full "recent news or unknown topic"
   ```

3. **Test with Alticap** (if it exists anywhere):
   ```bash
   ./query.sh --debug --full "Alticap"
   ```

4. **Monitor quality ledger**:
   ```bash
   ./query.sh --ledger-recent
   ./query.sh --ledger-stats
   ```

5. **Tune thresholds** based on results

## Questions?

See detailed documentation:
- [TROUBLESHOOTING-BM25.md](./TROUBLESHOOTING-BM25.md)
- [TROUBLESHOOTING-CRAG.md](./TROUBLESHOOTING-CRAG.md)
- [README-v44.md](./README-v44.md)

Or run diagnostics:
- `./diagnose-bm25.sh`
- `./check-searxng.sh`
- `./query.sh --debug`
