# CRAG Web Search Not Triggering - Troubleshooting

## Problem Description

When querying for content that doesn't exist in your database (like "Alticap"), you expect:

1. **Low retrieval quality** → CRAG detects irrelevant results
2. **CRAG triggers** → Web search as fallback
3. **Web results** → Supplement or replace local results

But instead, you see:
- No web search happening
- Irrelevant results returned
- Low confidence score (correctly detected)
- No indication that CRAG tried to help

## Root Causes

There are **TWO independent issues** that both contribute to this problem:

### Issue 1: BM25 Index Missing ✓ FIXED

**Problem**: BM25 keyword search returns 0 results

**Impact**:
- Hybrid search falls back to vector-only
- Keyword queries don't match via BM25
- Poor retrieval results for exact terms

**Solution**: See [TROUBLESHOOTING-BM25.md](./TROUBLESHOOTING-BM25.md)

Quick fix:
```bash
./rebuild-bm25.sh
```

### Issue 2: SearXNG Not Running ← YOU ARE HERE

**Problem**: CRAG tries to trigger web search but SearXNG is not accessible

**Impact**:
- CRAG silently fails when trying web search
- No fallback for queries about content not in database
- Users get wrong answers instead of web-augmented answers

**Solution**: Install and run SearXNG (details below)

## How CRAG Should Work

```
Query: "Alticap" (not in database)
  ↓
1. Hybrid Search (BM25 + Vector)
   → Returns: Harry Potter chunks (irrelevant)
  ↓
2. CRAG Evaluates Retrieval Quality
   - Keyword overlap: 0% (no "alticap" in results)
   - RRF scores: ~0.008 (low)
   - Quality score: 0.0008
   - Threshold: 0.4
   - Decision: 0.0008 < 0.4 → INSUFFICIENT
  ↓
3. CRAG Triggers Web Search
   → Calls: search_web("Alticap", max_results=3)
  ↓
4. SearXNG Searches Web
   → Returns: 3 web results about Alticap
  ↓
5. Combine Results
   → Context = Local chunks (low quality) + Web results
  ↓
6. Generate Answer
   → LLM uses web results to answer correctly
```

## What's Happening Now

```
Query: "Alticap"
  ↓
1. Hybrid Search
   → Returns: Harry Potter chunks (irrelevant)
  ↓
2. CRAG Evaluates Retrieval Quality
   - Quality score: 0.0008
   - Decision: INSUFFICIENT → Trigger web search
  ↓
3. CRAG Calls search_web()
   → Tries to connect to http://localhost:8085/search
   → ConnectionError: SearXNG not running
   → Returns: [] (empty, silently)
  ↓
4. No Web Results
   → Context = Only local chunks (irrelevant)
  ↓
5. Generate Answer
   → LLM tries to answer using irrelevant context
   → Result: Wrong answer with LOW_CONFIDENCE
```

## Diagnosis

### Step 1: Check if SearXNG is Running

```bash
cd /root  # Or your RAG installation directory
./check-searxng.sh
```

Expected output if SearXNG is **not** running:

```
============================================================
SearXNG Diagnostic & Setup
============================================================
SearXNG URL: http://localhost:8085

[1] Checking if SearXNG is accessible...
✗ SearXNG is NOT accessible

============================================================
Impact
============================================================
Without SearXNG:
  - CRAG (Corrective RAG) cannot perform web fallback
  - Queries for content not in your database will fail
  - Low-confidence retrievals won't trigger web search
```

### Step 2: Apply CRAG Debug Patch

To see exactly what CRAG is doing:

```bash
cd /root
cp ~/Rag4DietPI/patch-crag-debug.sh .
chmod +x patch-crag-debug.sh
./patch-crag-debug.sh
```

This adds debug logging to show:
- CRAG quality evaluation scores
- When CRAG triggers web search
- Web search errors (SearXNG not accessible)

### Step 3: Test with Debug Mode

```bash
./query.sh --clear-cache
./query.sh --debug --full "Alticap"
```

**Before patch**, you see:
```
[HYBRID] BM25 search returned 0 results
[SEARCH] Found 5 results
[POST-RETRIEVAL] 5 chunks after filtering
(no CRAG messages)
```

**After patch with SearXNG NOT running**, you see:
```
[HYBRID] BM25 search returned 0 results
[SEARCH] Found 5 results
[CRAG] Retrieval quality evaluation:
  Score: 0.0008 | Threshold: 0.4
  Decision: ✗ INSUFFICIENT - triggering web search
[WEB SEARCH] ✗ SearXNG not accessible at http://localhost:8085/search
[WEB SEARCH] Install: docker run -d -p 8085:8080 searxng/searxng
[POST-RETRIEVAL] 5 chunks after filtering
```

**After fixing SearXNG**, you see:
```
[HYBRID] BM25 search returned 5 results
[SEARCH] Found 10 results
[CRAG] Retrieval quality evaluation:
  Score: 0.0012 | Threshold: 0.4
  Decision: ✗ INSUFFICIENT - triggering web search
[WEB SEARCH] Got 3 results from web
[POST-RETRIEVAL] 13 chunks after filtering (10 local + 3 web)
```

## Solution: Install SearXNG

### Option 1: Docker (Recommended)

Fastest and easiest method:

```bash
# Pull and run SearXNG
docker run -d \
  --name searxng \
  -p 8085:8080 \
  --restart unless-stopped \
  searxng/searxng:latest

# Wait 10-15 seconds for startup
sleep 15

# Test
curl "http://localhost:8085/search?q=test&format=json"
```

Expected response:
```json
{
  "results": [
    {"title": "...", "url": "...", "content": "..."},
    ...
  ],
  "query": "test"
}
```

### Option 2: Docker Compose

If you want persistent configuration:

```bash
cd /root
mkdir -p searxng
cd searxng

cat > docker-compose.yml << 'EOF'
version: '3.7'

services:
  searxng:
    image: searxng/searxng:latest
    container_name: searxng
    ports:
      - "8085:8080"
    volumes:
      - ./searxng:/etc/searxng:rw
    environment:
      - SEARXNG_BASE_URL=http://localhost:8085/
    restart: unless-stopped
EOF

docker-compose up -d
```

### Option 3: Manual Installation

See official docs: https://docs.searxng.org/admin/installation.html

For Debian/DietPi:
```bash
apt-get update
apt-get install git python3-dev python3-babel python3-venv \
                uwsgi uwsgi-plugin-python3 \
                libapache2-mod-uwsgi nginx
# ... (follow official guide)
```

Docker is much simpler.

## Configuration

### System Configuration (config.env)

After installing SearXNG, update your RAG config:

```bash
# config.env

# Web Search
SEARXNG_URL=http://localhost:8085/search
SEARXNG_TIMEOUT=10
SEARXNG_MAX_RESULTS=5

# CRAG (Corrective RAG)
CRAG_ENABLED=true
CRAG_THRESHOLD=0.4  # Lower = more aggressive (triggers more often)
```

### CRAG Threshold Tuning

The `CRAG_THRESHOLD` controls when web search is triggered:

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.2 | Very aggressive | Always try web for any doubt |
| 0.4 | Balanced (default) | Web search for clear mismatches |
| 0.6 | Conservative | Only obvious failures |
| 0.8 | Very conservative | Almost never triggers |

Lower threshold = More web searches = Slower but more comprehensive

### SearXNG Engine Configuration

Customize which search engines SearXNG uses:

```bash
# config.env
SEARXNG_ALLOWED_ENGINES=wikipedia,duckduckgo,stackoverflow,archive
```

Available engines:
- **General**: duckduckgo, bing, google, brave
- **Knowledge**: wikipedia, wikidata
- **Technical**: stackoverflow, github, mdn, archlinux
- **Archive**: archive.org, openlibrary

## Verification

### 1. Check SearXNG

```bash
./check-searxng.sh
```

Should show:
```
✓ SearXNG is RUNNING and accessible
  Sample query returned: 10 results
✓ SearXNG is working correctly
  CRAG web search should function properly
```

### 2. Test CRAG Trigger

Query something NOT in your database:

```bash
./query.sh --clear-cache
./query.sh --debug --full "quantum computing breakthrough 2024"
```

Look for:
```
[CRAG] Retrieval quality evaluation:
  Score: 0.0234 | Threshold: 0.4
  Decision: ✗ INSUFFICIENT - triggering web search
[WEB SEARCH] Got 3 results from web
```

### 3. Test Full Pipeline

```bash
# Query 1: Should use local docs (good match)
./query.sh --debug --full "content you know is in your database"

# Query 2: Should trigger CRAG (no match)
./query.sh --debug --full "breaking news today"
```

Query 1 should show:
```
[CRAG] Decision: ✓ SUFFICIENT
```

Query 2 should show:
```
[CRAG] Decision: ✗ INSUFFICIENT - triggering web search
[WEB SEARCH] Got 3 results from web
```

## Alternative: Disable CRAG

If you don't want web search capability, disable CRAG:

### Permanently

Edit `config.env`:
```bash
CRAG_ENABLED=false
```

### Per Query

```bash
CRAG_ENABLED=false ./query.sh "your query"
```

### Mode Selection

```bash
# Ultrafast mode: CRAG disabled
./query.sh --ultrafast "query"

# Default mode: CRAG disabled by default
./query.sh "query"

# Full mode: CRAG enabled (requires SearXNG)
./query.sh --full "query"
```

**Note**: Without CRAG, queries for content not in your database will:
- Return irrelevant results
- Show LOW_CONFIDENCE correctly
- But not attempt to find better information from web

## Architecture

### CRAG Evaluation Flow

```python
def evaluate_retrieval_quality(query, chunks, threshold=0.4):
    """
    Calculate quality score based on:
    1. Keyword overlap (query words in chunk text)
    2. Retrieval scores (RRF or rerank scores)

    Formula:
      score = (overlap * 0.5) + (retrieval_score * 0.5)

    Returns: (is_sufficient, score)
    """

    # For query "Alticap" with Harry Potter results:
    # overlap = 0.0 (no "alticap" in Harry Potter text)
    # retrieval_score = 0.008 (vector similarity)
    # score = 0.0*0.5 + 0.008*0.5 = 0.004
    # 0.004 < 0.4 → INSUFFICIENT → Trigger web search
```

### Web Search Flow

```python
def crag_process(query, chunks, config):
    """CRAG main logic"""
    threshold = config.get("crag_threshold", 0.4)

    is_sufficient, score = evaluate_retrieval_quality(
        query, chunks, threshold
    )

    if is_sufficient:
        return chunks, False, []  # Good enough, no web search

    # Trigger web search
    from web_search import search_web
    web_results = search_web(query, max_results=3)

    return chunks, True, web_results
```

### SearXNG Integration

```python
def search_web(query, max_results=5, timeout=10):
    """Search via SearXNG"""
    searxng_url = "http://localhost:8085/search"

    try:
        resp = requests.get(
            searxng_url,
            params={"q": query, "format": "json"},
            timeout=timeout
        )

        if resp.status_code != 200:
            return []

        results = resp.json().get("results", [])
        return format_results(results[:max_results])

    except requests.exceptions.ConnectionError:
        # SearXNG not running → return empty
        # (Now shows error in DEBUG mode after patch)
        return []
```

## Common Questions

### Q: Why doesn't CRAG error visibly when SearXNG is down?

**A**: By design, CRAG fails gracefully. If web search fails, it falls back to local results only. This prevents the entire query from failing.

After applying the debug patch, you'll see the error in `--debug` mode.

### Q: Can I use a different search engine instead of SearXNG?

**A**: Yes, but you'd need to modify `lib/web_search.py`. SearXNG is recommended because:
- Privacy-respecting (self-hosted)
- No API keys needed
- Aggregates multiple search engines
- JSON API for easy integration

Alternative: Use DuckDuckGo API, Google Custom Search, or Bing API by modifying the web_search module.

### Q: Does CRAG cost money?

**A**: No, if using SearXNG (which uses free engines). If you modify to use commercial APIs (Google, Bing), those may have costs.

### Q: How much does CRAG slow down queries?

**A**: When triggered:
- SearXNG search: ~1-3 seconds
- Increases total query time by 20-30%

When not triggered (local results sufficient): No impact.

### Q: Can CRAG work without SearXNG?

**A**: Not currently. You need a web search backend. Options:
1. Install SearXNG (recommended)
2. Modify code to use another API
3. Disable CRAG entirely

## Summary

**Problem**: CRAG tries to trigger web search but fails silently because SearXNG is not running

**Symptoms**:
- No web search happening
- Irrelevant results for queries not in database
- No visible errors

**Solutions**:
1. **Install SearXNG** (5 minutes with Docker)
2. **Apply debug patch** to see CRAG activity
3. **Configure thresholds** for your use case

**Quick Fix**:
```bash
# Install SearXNG
docker run -d --name searxng -p 8085:8080 searxng/searxng:latest

# Apply debug patch
cd /root
./patch-crag-debug.sh

# Test
./query.sh --clear-cache
./query.sh --debug --full "test query not in database"
```

After this, CRAG will:
- Show quality evaluation in debug mode
- Trigger web search when needed
- Display errors if something fails
- Provide web-augmented answers for unknown queries

## Related Issues

This issue is related to:
- [TROUBLESHOOTING-BM25.md](./TROUBLESHOOTING-BM25.md) - BM25 index missing
- [README-v44.md](./README-v44.md) - Quality feedback loop

Both BM25 and CRAG need to be working for optimal results on queries like "Alticap".
