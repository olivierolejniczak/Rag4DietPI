# Fresh DietPI Installation Guide - RAG v44

## What We Discovered

From debugging your "Alticap" query issue, we found:

### Working Components ✅
- BM25 index building (194 documents)
- CRAG evaluation logic (correctly detects low quality: 0.0037 < 0.4)
- Quality scoring system
- Query wrapper script
- All Python modules present

### Issues Found ❌
1. **SearXNG bot detection** - Blocks API access with 403 Forbidden
2. **Docker/network configuration** - Port mapping issues on DietPI
3. **Missing "Alticap" in database** - Your DB has Harry Potter content only

### Root Cause
CRAG **works correctly** but can't complete web search because SearXNG blocks the requests. Even with proper headers, SearXNG's bot detection interferes.

---

## Fresh Installation Steps

### Prerequisites

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install required packages
apt-get install -y curl git python3 python3-pip docker.io

# Start Docker
systemctl enable docker
systemctl start docker
```

### Step 1: Clone Repository

```bash
cd ~
git clone https://github.com/olivierolejniczak/Rag4DietPI.git
cd Rag4DietPI
git checkout claude/debug-rag-query-F6HAr  # Use the fixed branch
```

### Step 2: Run Setup Scripts

```bash
# 1. Core setup (Qdrant, Ollama, config)
./setup-rag-core-v44.sh

# Wait for services to start
sleep 30

# 2. Ingestion setup (document processing)
./setup-rag-ingest-v44.sh

# 3. Query setup (RAG system)
./setup-rag-query-v44.sh
```

### Step 3: Fix SearXNG (Critical)

The default SearXNG has bot detection issues. Use this workaround:

```bash
# Option A: Run SearXNG with relaxed settings
docker run -d \
  --name searxng \
  -p 8085:8080 \
  -e SEARXNG_BASE_URL=http://localhost:8085/ \
  --restart unless-stopped \
  searxng/searxng:latest

sleep 20

# Apply our header patch to web_search.py
cd /root  # Or wherever query.sh was installed

cat > /tmp/patch_websearch.py << 'EOF'
with open('lib/web_search.py', 'r') as f:
    content = f.read()

old = '''        resp = requests.get(
            searxng_url,
            params=params,
            timeout=timeout
        )'''

new = '''        resp = requests.get(
            searxng_url,
            params=params,
            timeout=timeout,
            headers={
                'X-Forwarded-For': '127.0.0.1',
                'X-Real-IP': '127.0.0.1',
                'User-Agent': 'Mozilla/5.0 (RAGSystem/1.0)'
            }
        )'''

content = content.replace(old, new)

with open('lib/web_search.py', 'w') as f:
    f.write(content)

print("✓ Patched web_search.py with headers")
EOF

python3 /tmp/patch_websearch.py
```

**OR Option B: Disable CRAG if SearXNG won't work**

```bash
# In config.env
echo "CRAG_ENABLED=false" >> config.env

# Then queries won't try web search
# You'll get LOW_CONFIDENCE for unknown queries instead
```

### Step 4: Ingest Your Documents

```bash
# Put your documents in ./documents/
mkdir -p documents
cp /path/to/your/files/* documents/

# Run ingestion
./ingest.sh

# This will:
# - Process all documents
# - Create vector embeddings
# - Build BM25 index
# - Create vocabulary
```

### Step 5: Rebuild BM25 Index

The BM25 rebuild script from our fixes:

```bash
cd /root  # Or installation directory

# Copy rebuild script
cp ~/Rag4DietPI/rebuild-bm25.sh .
chmod +x rebuild-bm25.sh

# Install rank-bm25 if needed
pip3 install rank-bm25

# Run rebuild
./rebuild-bm25.sh
```

Expected output:
```
✓ Fetched XXX documents from Qdrant
✓ Prepared XXX documents for indexing
✓ BM25 index saved successfully
  Location: /root/cache/bm25_index.pkl
  Documents: XXX
  Size: X.XX MB
✓ Vocabulary saved: XXXX words
```

### Step 6: Verify Installation

```bash
# Check all services
./status.sh

# Should show:
# ✓ Ollama running
# ✓ Qdrant running
# ✓ SearXNG running (or ○ if disabled)

# Run verification
./verify.sh
```

### Step 7: Test Queries

```bash
# Test 1: Content in your database
./query.sh "question about your documents"

# Test 2: Content NOT in database (tests CRAG)
./query.sh --debug --full "some topic not in your docs"

# Should see CRAG trigger if SearXNG works:
# [CRAG] Score: 0.00XX | Threshold: 0.4
# [CRAG] Decision: ✗ INSUFFICIENT - triggering web search
# [WEB SEARCH] Got X results from web
```

---

## Configuration

### Essential Settings (config.env)

```bash
# LLM
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=qwen2.5:1.5b
EMBEDDING_MODEL=nomic-embed-text

# Vector DB
QDRANT_HOST=http://localhost:6333
COLLECTION_NAME=documents

# Web Search (if working)
SEARXNG_URL=http://localhost:8085/search
CRAG_ENABLED=true
CRAG_THRESHOLD=0.4

# Quality
CONFIDENCE_THRESHOLD_HIGH=0.7
GROUNDING_THRESHOLD=0.5
ABSTENTION_ENABLED=true
```

### If SearXNG Doesn't Work

```bash
# Disable CRAG in config.env
CRAG_ENABLED=false

# System will still work but:
# - No web fallback for unknown queries
# - You'll get LOW_CONFIDENCE instead
# - Quality scores still work
```

---

## Troubleshooting

### BM25 Returns 0 Results

```bash
# Check index exists
ls -lh cache/bm25_index.pkl

# If missing, rebuild
./rebuild-bm25.sh

# Verify in Python
python3 << 'EOF'
import pickle
with open("cache/bm25_index.pkl", "rb") as f:
    data = pickle.load(f)
print(f"Documents in index: {len(data['doc_ids'])}")
EOF
```

### SearXNG Not Working

```bash
# Check container
docker ps | grep searxng

# Check logs
docker logs searxng | tail -30

# Test manually
curl -H "X-Forwarded-For: 127.0.0.1" \
     "http://localhost:8085/search?q=test&format=json" | head -100

# If fails, disable CRAG:
echo "CRAG_ENABLED=false" >> config.env
```

### Low Grounding Scores

This is normal when querying for content not in your database.

Example from your "Alticap" query:
- Query: "Alticap" (not in DB)
- Retrieved: Harry Potter content (irrelevant)
- Grounding: 0.180 (very low, correct!)
- Decision: LOW_CONFIDENCE (correct!)

**This is the system working as designed.**

If CRAG is enabled and working:
- It should trigger web search
- Add web results to context
- Improve grounding score

---

## Key Files After Install

```
/root/  (or installation directory)
├── query.sh              # Main query interface
├── ingest.sh            # Document ingestion
├── rebuild-bm25.sh      # BM25 index rebuild
├── config.env           # Configuration
├── lib/                 # Python modules
│   ├── query_main.py
│   ├── hybrid_search.py
│   ├── post_retrieval.py  # Contains CRAG
│   ├── web_search.py      # SearXNG interface
│   ├── quality_ledger.py
│   └── ...
└── cache/               # Runtime data
    ├── bm25_index.pkl
    ├── vocabulary.json
    ├── quality_ledger.sqlite
    └── query_cache.json
```

---

## Expected Behavior

### Query in Database

```bash
./query.sh "Harry Potter Hogwarts"

# Output:
[HYBRID] Vector: 10, BM25: 8 results
[QUALITY SCORES]
  Grounding: 0.825  # High
✅ Confidence: CONFIDENT
```

### Query NOT in Database (with CRAG)

```bash
./query.sh --full "Alticap"

# Output:
[HYBRID] Vector: 10, BM25: 0 results
[CRAG] Score: 0.003 | Threshold: 0.4
[CRAG] Decision: ✗ INSUFFICIENT - triggering web search
[WEB SEARCH] Got 3 results from web
[QUALITY SCORES]
  Grounding: 0.754  # Better with web results
✅ Confidence: CONFIDENT
```

### Query NOT in Database (without CRAG)

```bash
./query.sh "Alticap"

# Output:
[HYBRID] Vector: 10, BM25: 0 results
[QUALITY SCORES]
  Grounding: 0.180  # Low
⚠️ Confidence: LOW_CONFIDENCE
# System abstains or warns
```

---

## What Changed from v43 to v44

1. **Quality Ledger** - SQLite database tracking all queries
2. **Scoring System** - Deterministic quality scores (no LLM)
3. **Decision Engine** - Confident/Low/Abstain decisions
4. **Abstention** - Can refuse to answer when uncertain
5. **CRAG** - Web search fallback (requires SearXNG)

All backward compatible - can disable new features.

---

## Summary

After fresh install, you should have:

✅ Working RAG system
✅ BM25 + Vector hybrid search
✅ Quality scoring and abstention
✅ CRAG logic (even if web search disabled)
✅ Query caching and memory
✅ Comprehensive debug output

With SearXNG working:
✅ Web search fallback for unknown queries
✅ Full CRAG pipeline

Without SearXNG:
⚠️ No web fallback (but system still works)
⚠️ Unknown queries get LOW_CONFIDENCE

---

## Post-Install Testing

```bash
# 1. Test known content
./query.sh --debug "topic in your documents"
# Should: High grounding, CONFIDENT

# 2. Test unknown content
./query.sh --debug --full "random topic not in docs"
# Should: Low grounding, attempt CRAG

# 3. Check quality ledger
./query.sh --ledger-recent
# Should: Show recent queries with scores

# 4. Check statistics
./query.sh --ledger-stats
# Should: Show confidence distribution
```

---

## Need Help?

After fresh install, if issues persist:

1. Run diagnostics:
   ```bash
   ./diagnose-bm25.sh
   ./check-searxng.sh
   ```

2. Check service status:
   ```bash
   ./status.sh
   ./verify.sh
   ```

3. Enable debug mode:
   ```bash
   ./query.sh --debug --full "your query"
   ```

4. Check logs:
   ```bash
   docker logs qdrant
   docker logs searxng
   ```

---

## Files to Copy from Repo

After fresh DietPI install, make sure to copy from the `claude/debug-rag-query-F6HAr` branch:

```bash
# Essential scripts
rebuild-bm25.sh          # BM25 index rebuild
diagnose-bm25.sh         # BM25 diagnostics
check-searxng.sh         # SearXNG diagnostics

# Documentation
TROUBLESHOOTING-BM25.md   # BM25 issues
TROUBLESHOOTING-CRAG.md   # CRAG/web search issues
FIX-ALTICAP-QUERY.md      # Complete troubleshooting
FRESH-INSTALL-GUIDE.md    # This file
```

Good luck with the fresh install! 🚀
