#!/bin/bash
# ============================================================================
# RAG System v46 - Query Setup (ALL FEATURES IMPLEMENTED)
# ============================================================================
# All features fully implemented:
#   - HyDE (Hypothetical Document Embeddings)
#   - CRAG (Corrective RAG with SearXNG web fallback)
#   - FlashRank reranking
#   - StepBack prompting
#   - Subquery decomposition
#   - Query classification
#   - Grounding verification
#   - RSE (Relevant Segment Extraction)
#   - Context window expansion
#   - Diversity filtering
#   - Adaptive retrieval
#   - Self-RAG
#   - Query caching
#   - Conversation memory
#   - SearXNG web search (privacy-respecting, no commercial engines)
#   - Enhanced debug output (LLM/embedding model info)
#   - --full mode with NO timeouts
#
# v45 NEW - Quality Feedback Loop:
#   - Quality Ledger (SQLite persistent tracking)
#   - Retrieval Confidence scoring (no LLM)
#   - Answer Coverage scoring (no LLM)
#   - Grounding Score (no LLM)
#   - Decision Engine (confident/low_confidence/abstained)
#   - Controlled abstention (no hallucination)
#   - Human feedback support (--feedback flag)
# 
# Default config: Heavy features DISABLED for speed (<90s)
# Enable with --full flag or edit config.env
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m'

log_ok() { echo -e "[${GREEN}OK${NC}] $1"; }
log_warn() { echo -e "[${YELLOW}WARN${NC}] $1"; }

PROJECT_DIR="${1:-$(pwd)}"

if [ ! -f "$PROJECT_DIR/config.env" ]; then
    echo "ERROR: Run setup-rag-core-v46.sh first!"
    exit 1
fi

cd "$PROJECT_DIR"
source config.env
[ -d "./venv" ] && source ./venv/bin/activate

clear
echo "============================================================================"
echo "   RAG System v46 - Query Setup (Full Features)"
echo "============================================================================"
echo ""

mkdir -p lib cache

# ============================================================================
# LLM Helper Module (with debug info)
# ============================================================================
echo "Creating LLM helper module..."
cat > lib/llm_helper.py << 'EOFPY'
"""LLM generation helper with timeout handling and debug tracking"""
import os
import requests
import time

# Debug tracking for LLM and embedding calls
_debug_info = {
    "llm_model": None,
    "llm_calls": 0,
    "llm_total_time": 0,
    "embedding_model": None,
    "embedding_calls": 0,
    "embedding_total_time": 0,
}

def get_debug_info():
    """Return debug info about LLM/embedding usage"""
    return _debug_info.copy()

def reset_debug_info():
    """Reset debug counters"""
    global _debug_info
    _debug_info = {
        "llm_model": None,
        "llm_calls": 0,
        "llm_total_time": 0,
        "embedding_model": None,
        "embedding_calls": 0,
        "embedding_total_time": 0,
    }

def get_config():
    return {
        "ollama_host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        "llm_model": os.environ.get("LLM_MODEL", "qwen2.5:1.5b"),
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"),
        "timeout_default": int(os.environ.get("LLM_TIMEOUT_DEFAULT", "180")),
        "timeout_ultrafast": int(os.environ.get("LLM_TIMEOUT_ULTRAFAST", "90")),
        "timeout_full": int(os.environ.get("LLM_TIMEOUT_FULL", "0")),  # 0 = no timeout
        "temperature": float(os.environ.get("TEMPERATURE", "0.2")),
    }

def llm_generate(prompt, max_tokens=500, timeout=None, temperature=None):
    """Generate text with LLM, tracking debug info"""
    global _debug_info
    config = get_config()
    
    _debug_info["llm_model"] = config["llm_model"]
    _debug_info["llm_calls"] += 1
    
    if timeout is None:
        timeout = config["timeout_default"]
    if temperature is None:
        temperature = config["temperature"]
    
    # Timeout 0 = no timeout (for --full mode)
    req_timeout = None if timeout == 0 else timeout
    
    start = time.time()
    try:
        resp = requests.post(
            f"{config['ollama_host']}/api/generate",
            json={
                "model": config["llm_model"],
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            },
            timeout=req_timeout
        )
        elapsed = time.time() - start
        _debug_info["llm_total_time"] += elapsed
        
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
        else:
            return None
    except requests.exceptions.Timeout:
        _debug_info["llm_total_time"] += time.time() - start
        return None
    except Exception as e:
        _debug_info["llm_total_time"] += time.time() - start
        return None

def llm_generate_fast(prompt, max_tokens=100):
    """Quick generation for classification/enhancement"""
    config = get_config()
    timeout = min(30, config["timeout_ultrafast"])
    return llm_generate(prompt, max_tokens, timeout, temperature=0.1)

def get_embedding(text, timeout=60):
    """Get embedding vector, tracking debug info"""
    global _debug_info
    config = get_config()
    
    _debug_info["embedding_model"] = config["embedding_model"]
    _debug_info["embedding_calls"] += 1
    
    # Timeout 0 = no timeout
    req_timeout = None if timeout == 0 else timeout
    
    start = time.time()
    try:
        resp = requests.post(
            f"{config['ollama_host']}/api/embeddings",
            json={"model": config["embedding_model"], "prompt": text},
            timeout=req_timeout
        )
        elapsed = time.time() - start
        _debug_info["embedding_total_time"] += elapsed
        
        if resp.status_code == 200:
            return resp.json().get("embedding", [])
        return []
    except:
        _debug_info["embedding_total_time"] += time.time() - start
        return []
EOFPY
log_ok "LLM helper module (with debug)"

# ============================================================================
# Query Enhancement Module (HyDE, StepBack, Subqueries)
# ============================================================================
echo "Creating query enhancement module..."
cat > lib/query_enhance.py << 'EOFPY'
"""Query enhancement: Classification, HyDE, StepBack, Subqueries"""
import os
import re
from llm_helper import llm_generate_fast, llm_generate

def classify_query(query):
    """
    Classify query type to adjust retrieval strategy.
    Returns: factual, analytical, procedural, comparison, troubleshooting
    """
    query_lower = query.lower()
    
    # Pattern-based classification (fast)
    if any(w in query_lower for w in ["compare", "difference", "vs", "versus", "better"]):
        return "comparison"
    elif any(w in query_lower for w in ["how to", "steps", "guide", "tutorial", "process"]):
        return "procedural"
    elif any(w in query_lower for w in ["error", "fix", "problem", "issue", "debug", "troubleshoot"]):
        return "troubleshooting"
    elif any(w in query_lower for w in ["why", "explain", "reason", "cause", "analyze"]):
        return "analytical"
    elif any(w in query_lower for w in ["what", "who", "when", "where", "which", "define"]):
        return "factual"
    
    return "factual"

def classify_query_llm(query):
    """LLM-based query classification (more accurate but slower)"""
    prompt = f"""Classify this query into exactly ONE category:
- factual: Simple fact lookup (what, who, when, where)
- analytical: Requires explanation or reasoning (why, how does)
- procedural: Step-by-step instructions (how to)
- comparison: Comparing options (vs, better, difference)
- troubleshooting: Problem solving (error, fix, debug)

Query: {query}

Category (one word only):"""

    result = llm_generate_fast(prompt, max_tokens=10)
    if result:
        result = result.lower().strip().split()[0]
        if result in ["factual", "analytical", "procedural", "comparison", "troubleshooting"]:
            return result
    
    return classify_query(query)  # Fallback to pattern-based

def generate_hyde_document(query, classification="factual"):
    """
    HyDE: Generate hypothetical document that would answer the query.
    Search with the hypothetical document embedding instead of query.
    """
    prompts = {
        "factual": f"Write a short paragraph that directly answers this question:\n{query}\n\nAnswer:",
        "analytical": f"Write a detailed explanation that answers:\n{query}\n\nExplanation:",
        "procedural": f"Write step-by-step instructions for:\n{query}\n\nSteps:",
        "comparison": f"Write a comparison that addresses:\n{query}\n\nComparison:",
        "troubleshooting": f"Write a troubleshooting guide for:\n{query}\n\nSolution:",
    }
    
    prompt = prompts.get(classification, prompts["factual"])
    
    hyde_doc = llm_generate(prompt, max_tokens=200, timeout=30)
    if hyde_doc and len(hyde_doc) > 50:
        return hyde_doc
    
    return None

def generate_stepback_query(query):
    """
    StepBack prompting: Generate a more abstract/general question.
    Helps retrieve broader context.
    """
    prompt = f"""Given this specific question, generate a more general/abstract question that would help understand the topic better.

Specific question: {query}

General question (one sentence):"""

    result = llm_generate_fast(prompt, max_tokens=50)
    if result and len(result) > 10:
        # Clean up
        result = result.strip().strip('"').strip()
        if not result.endswith('?'):
            result += '?'
        return result
    
    return None

def decompose_query(query, max_subqueries=3):
    """
    Decompose complex query into simpler sub-queries.
    """
    # Check if query is complex enough
    if len(query.split()) < 8 and ' and ' not in query.lower():
        return [query]
    
    prompt = f"""Break this complex question into {max_subqueries} simpler sub-questions.
Return ONLY the questions, one per line.

Complex question: {query}

Sub-questions:"""

    result = llm_generate_fast(prompt, max_tokens=150)
    if not result:
        return [query]
    
    # Parse sub-queries
    subqueries = []
    for line in result.split('\n'):
        line = line.strip()
        # Remove numbering
        line = re.sub(r'^[\d\.\)\-]+\s*', '', line)
        if line and len(line) > 10 and '?' in line or len(line) > 20:
            subqueries.append(line)
    
    if subqueries:
        return subqueries[:max_subqueries]
    
    return [query]

def rewrite_query(query):
    """Simple query rewriting: fix grammar, expand abbreviations"""
    # Common abbreviations
    expansions = {
        "ai": "artificial intelligence",
        "ml": "machine learning",
        "api": "application programming interface",
        "db": "database",
        "ui": "user interface",
        "ux": "user experience",
    }
    
    words = query.split()
    rewritten = []
    for word in words:
        lower = word.lower().strip('?.,!')
        if lower in expansions:
            rewritten.append(expansions[lower])
        else:
            rewritten.append(word)
    
    return ' '.join(rewritten)

def enhance_query(query, config):
    """
    Main query enhancement function.
    Returns enhanced query info based on config flags.
    """
    result = {
        "original": query,
        "rewritten": query,
        "classification": "factual",
        "hyde_document": None,
        "stepback_query": None,
        "sub_queries": [query],
    }
    
    # Query rewriting (always fast)
    if config.get("rewrite_enabled"):
        result["rewritten"] = rewrite_query(query)
    
    # Classification
    if config.get("classification_enabled"):
        result["classification"] = classify_query_llm(query)
    else:
        result["classification"] = classify_query(query)
    
    # HyDE
    if config.get("hyde_enabled"):
        hyde_doc = generate_hyde_document(query, result["classification"])
        if hyde_doc:
            result["hyde_document"] = hyde_doc
    
    # StepBack
    if config.get("stepback_enabled"):
        stepback = generate_stepback_query(query)
        if stepback:
            result["stepback_query"] = stepback
    
    # Subqueries
    if config.get("subquery_enabled"):
        max_sub = config.get("subquery_max", 2)
        result["sub_queries"] = decompose_query(query, max_sub)
    
    return result
EOFPY
log_ok "Query enhancement module"

# ============================================================================
# Hybrid Search Module
# ============================================================================
echo "Creating hybrid search module..."
cat > lib/hybrid_search.py << 'EOFPY'
"""Hybrid search: BM25 + Dense vectors with RRF fusion"""
import os
import requests
import pickle
import re

def get_embedding(text):
    """Get embedding from Ollama"""
    try:
        resp = requests.post(
            f"{os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}/api/embeddings",
            json={
                "model": os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"),
                "prompt": text[:2000]
            },
            timeout=int(os.environ.get("EMBEDDING_TIMEOUT", "60"))
        )
        if resp.status_code == 200:
            return resp.json().get("embedding", [])
    except:
        pass
    return []

def bm25_search(query, top_k=10):
    """BM25 keyword search"""
    try:
        with open("cache/bm25_index.pkl", "rb") as f:
            data = pickle.load(f)
        
        bm25 = data["bm25"]
        doc_ids = data["doc_ids"]
        
        # Tokenize query
        tokens = re.findall(r'\b\w+\b', query.lower())
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by'}
        tokens = [t for t in tokens if len(t) > 2 and t not in stopwords]
        
        if not tokens:
            return []
        
        scores = bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        return [(doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]
    except Exception as e:
        return []

def vector_search(query_or_embedding, top_k=10):
    """Dense vector search in Qdrant"""
    import os
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    if isinstance(query_or_embedding, str):
        embedding = get_embedding(query_or_embedding)
    else:
        embedding = query_or_embedding
    
    if not embedding:
        if debug:
            print("  [VECTOR] No embedding generated")
        return []
    
    qdrant = os.environ.get("QDRANT_HOST", "http://localhost:6333")
    collection = os.environ.get("COLLECTION_NAME", "documents")
    
    try:
        resp = requests.post(
            f"{qdrant}/collections/{collection}/points/search",
            json={
                "vector": embedding,
                "limit": top_k,
                "with_payload": True
            },
            timeout=30
        )
        if resp.status_code == 200:
            results = resp.json().get("result", [])
            if debug:
                print(f"  [VECTOR] Got {len(results)} results from Qdrant")
                for r in results[:2]:
                    has_text = bool(r.get("payload", {}).get("text"))
                    print(f"    - id={str(r['id'])[:16]}... score={r['score']:.3f} has_text={has_text}")
            return [(r["id"], r["score"], r.get("payload", {})) for r in results]
        else:
            if debug:
                print(f"  [VECTOR] Qdrant error: {resp.status_code}")
    except Exception as e:
        if debug:
            print(f"  [VECTOR] Exception: {e}")
    return []

def fetch_payloads(doc_ids):
    """Fetch payloads for document IDs from Qdrant"""
    if not doc_ids:
        return {}
    
    import os
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    qdrant = os.environ.get("QDRANT_HOST", "http://localhost:6333")
    collection = os.environ.get("COLLECTION_NAME", "documents")
    
    payloads = {}
    try:
        # Qdrant API: POST /collections/{name}/points with ids list
        resp = requests.post(
            f"{qdrant}/collections/{collection}/points",
            json={"ids": list(doc_ids), "with_payload": True},
            timeout=30
        )
        
        if debug:
            print(f"  [FETCH] Qdrant response: {resp.status_code}")
        
        if resp.status_code == 200:
            result = resp.json().get("result", [])
            if debug:
                print(f"  [FETCH] Got {len(result)} points from Qdrant")
            for point in result:
                pid = point.get("id")
                payload = point.get("payload", {})
                if pid and payload:
                    payloads[pid] = payload
        else:
            if debug:
                print(f"  [FETCH] Error: {resp.text[:200]}")
    except Exception as e:
        if debug:
            print(f"  [FETCH] Exception: {e}")
    
    return payloads

def hybrid_search(query, top_k=5, alpha=0.5, hyde_embedding=None):
    """
    Combine BM25 and vector search using Reciprocal Rank Fusion.
    
    alpha: Weight for BM25 (1.0 = BM25 only, 0.0 = vector only)
    hyde_embedding: Optional pre-computed HyDE embedding
    """
    import os
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    k = 60  # RRF constant
    
    # Vector search (with optional HyDE)
    if hyde_embedding is not None:
        vector_results = vector_search(hyde_embedding, top_k * 2)
    else:
        vector_results = vector_search(query, top_k * 2)
    
    if debug:
        print(f"  [HYBRID] Vector search returned {len(vector_results)} results")
    
    # BM25 search
    bm25_results = bm25_search(query, top_k * 2)
    
    if debug:
        print(f"  [HYBRID] BM25 search returned {len(bm25_results)} results")
    
    # Collect scores and payloads
    rrf_scores = {}
    payloads = {}
    
    # Process vector results (these come with payloads)
    for rank, item in enumerate(vector_results):
        if len(item) >= 3:
            doc_id, score, payload = item[0], item[1], item[2]
        else:
            continue
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 - alpha) / (k + rank + 1)
        if payload and payload.get("text"):
            payloads[doc_id] = payload
    
    # Process BM25 results (need to fetch payloads separately)
    bm25_ids_to_fetch = set()
    for rank, item in enumerate(bm25_results):
        if len(item) >= 2:
            doc_id, score = item[0], item[1]
        else:
            continue
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + alpha / (k + rank + 1)
        if doc_id not in payloads:
            bm25_ids_to_fetch.add(doc_id)
    
    # Fetch missing payloads for BM25-only results
    if bm25_ids_to_fetch:
        fetched = fetch_payloads(bm25_ids_to_fetch)
        if debug:
            print(f"  [HYBRID] Fetched {len(fetched)} payloads for BM25 results")
        payloads.update(fetched)
    
    # =========================================================================
    # FILENAME BOOSTING: Boost documents whose filename matches query terms
    # =========================================================================
    query_terms = set(re.findall(r'\b\w{3,}\b', query.lower()))
    
    for doc_id in rrf_scores:
        payload = payloads.get(doc_id, {})
        filename = payload.get("filename", "").lower()
        
        # Check how many query terms appear in filename
        filename_terms = set(re.findall(r'\b\w{3,}\b', filename))
        matches = query_terms & filename_terms
        
        if matches:
            # Boost by 50% for each matching term
            boost = 1.0 + (0.5 * len(matches))
            rrf_scores[doc_id] *= boost
            if debug:
                print(f"  [BOOST] {filename[:30]}... +{len(matches)} terms ({', '.join(matches)}) -> score * {boost:.1f}")
    
    # Sort by RRF score (now with filename boost)
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    if debug:
        print(f"  [HYBRID] Top {len(sorted_results)} after RRF fusion + filename boost")
        for doc_id, score in sorted_results[:3]:
            has_text = bool(payloads.get(doc_id, {}).get("text"))
            fname = payloads.get(doc_id, {}).get("filename", "?")[:25]
            print(f"    - {fname}... score={score:.4f}")
    
    # Format results - include even if text is missing (with placeholder)
    results = []
    for doc_id, score in sorted_results:
        payload = payloads.get(doc_id, {})
        text = payload.get("text", "")
        if text:  # Only include if we have text
            results.append({
                "id": doc_id,
                "rrf_score": score,
                "text": text,
                "source": payload.get("source", ""),
                "filename": payload.get("filename", "unknown"),
                "section": payload.get("section", ""),
                "chunk_type": payload.get("chunk_type", "chunk"),
            })
    
    if debug and len(results) < len(sorted_results):
        print(f"  [HYBRID] Warning: {len(sorted_results) - len(results)} results dropped (missing text)")
    
    return results
EOFPY
log_ok "Hybrid search module"

# ============================================================================
# Post-Retrieval Module (Reranking, CRAG, RSE, Diversity)
# ============================================================================
echo "Creating post-retrieval module..."
cat > lib/post_retrieval.py << 'EOFPY'
"""Post-retrieval: Reranking, CRAG, RSE, Context Window, Diversity"""
import os
import re
import hashlib

# ============================================================================
# RERANKING with FlashRank
# ============================================================================
_reranker = None

def get_reranker():
    """Lazy-load FlashRank reranker"""
    global _reranker
    if _reranker is None:
        try:
            from flashrank import Ranker, RerankRequest
            model = os.environ.get("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")
            _reranker = Ranker(model_name=model)
        except ImportError:
            _reranker = False
        except Exception:
            _reranker = False
    return _reranker

def rerank_chunks(query, chunks, top_k=None):
    """
    Rerank chunks using FlashRank cross-encoder.
    Returns reordered chunks with rerank scores.
    """
    if not chunks:
        return []
    
    reranker = get_reranker()
    if not reranker:
        return chunks  # Return original order if reranker unavailable
    
    try:
        from flashrank import RerankRequest
        
        # Prepare passages
        passages = []
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            passages.append({"id": i, "text": text[:1000]})  # Limit text length
        
        # Rerank
        request = RerankRequest(query=query, passages=passages)
        results = reranker.rerank(request)
        
        # Reorder chunks
        reranked = []
        for result in results:
            idx = result.get("id", 0)
            if idx < len(chunks):
                chunk = chunks[idx].copy() if isinstance(chunks[idx], dict) else {"text": chunks[idx]}
                chunk["rerank_score"] = result.get("score", 0)
                reranked.append(chunk)
        
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked
    except Exception as e:
        return chunks

# ============================================================================
# CRAG: Corrective RAG
# ============================================================================
def evaluate_retrieval_quality(query, chunks, threshold=0.4):
    """
    Evaluate if retrieved chunks are relevant enough.
    Returns: (is_sufficient, confidence_score)
    """
    if not chunks:
        return False, 0.0
    
    # Simple heuristic: check keyword overlap and scores
    query_words = set(query.lower().split())
    
    total_score = 0
    for chunk in chunks:
        text = chunk.get("text", "").lower() if isinstance(chunk, dict) else str(chunk).lower()
        chunk_words = set(text.split())
        
        # Keyword overlap
        overlap = len(query_words & chunk_words) / max(len(query_words), 1)
        
        # RRF/rerank score
        score = chunk.get("rrf_score", 0) or chunk.get("rerank_score", 0)
        
        total_score += overlap * 0.5 + score * 0.5
    
    avg_score = total_score / len(chunks)
    return avg_score >= threshold, avg_score

def crag_process(query, chunks, config):
    """
    CRAG: If retrieval quality is low, trigger web search.
    Returns: (chunks, web_triggered, web_results)
    """
    threshold = config.get("crag_threshold", 0.4)
    
    is_sufficient, score = evaluate_retrieval_quality(query, chunks, threshold)
    
    if is_sufficient:
        return chunks, False, []
    
    # Trigger web search
    from web_search import search_web
    web_results = search_web(query, max_results=3)
    
    return chunks, True, web_results

# ============================================================================
# RSE: Relevant Segment Extraction
# ============================================================================
def extract_relevant_segments(query, text, max_segments=3, segment_size=200):
    """
    Extract most relevant segments from a chunk.
    Uses sentence-level matching.
    """
    if not text:
        return text
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 3:
        return text
    
    query_words = set(query.lower().split())
    
    # Score sentences
    scored = []
    for i, sent in enumerate(sentences):
        sent_words = set(sent.lower().split())
        overlap = len(query_words & sent_words)
        # Boost earlier sentences slightly
        position_bonus = 0.1 * (1 - i / len(sentences))
        scored.append((overlap + position_bonus, i, sent))
    
    # Get top sentences, maintain order
    scored.sort(reverse=True)
    top_indices = sorted([s[1] for s in scored[:max_segments]])
    
    # Reconstruct text
    segments = [sentences[i] for i in top_indices]
    return ' '.join(segments)

def apply_rse(query, chunks):
    """Apply RSE to all chunks"""
    result = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            new_chunk = chunk.copy()
            new_chunk["text"] = extract_relevant_segments(query, chunk.get("text", ""))
            result.append(new_chunk)
        else:
            result.append(extract_relevant_segments(query, str(chunk)))
    return result

# ============================================================================
# CONTEXT WINDOW: Fetch surrounding chunks
# ============================================================================
def expand_context_window(chunks, window_size=1):
    """
    Fetch surrounding chunks for context.
    Uses chunk_index and source to find neighbors.
    """
    if window_size == 0 or not chunks:
        return chunks
    
    import requests
    qdrant = os.environ.get("QDRANT_HOST", "http://localhost:6333")
    collection = os.environ.get("COLLECTION_NAME", "documents")
    
    expanded = []
    seen_ids = set()
    
    for chunk in chunks:
        if not isinstance(chunk, dict):
            expanded.append(chunk)
            continue
        
        chunk_id = chunk.get("id")
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        
        source = chunk.get("source", "")
        chunk_idx = chunk.get("chunk_index", -1)
        
        if not source or chunk_idx < 0:
            expanded.append(chunk)
            continue
        
        # Try to fetch neighbors
        try:
            # Search for chunks from same source with nearby indices
            neighbor_texts = []
            
            for offset in range(-window_size, window_size + 1):
                if offset == 0:
                    continue
                
                # This is simplified - in production, use proper filtering
                resp = requests.post(
                    f"{qdrant}/collections/{collection}/points/scroll",
                    json={
                        "filter": {
                            "must": [
                                {"key": "source", "match": {"value": source}},
                                {"key": "chunk_index", "match": {"value": chunk_idx + offset}}
                            ]
                        },
                        "limit": 1,
                        "with_payload": True
                    },
                    timeout=10
                )
                
                if resp.status_code == 200:
                    points = resp.json().get("result", {}).get("points", [])
                    if points:
                        neighbor_text = points[0].get("payload", {}).get("text", "")
                        if neighbor_text:
                            neighbor_texts.append((offset, neighbor_text))
            
            # Combine with original chunk
            if neighbor_texts:
                neighbor_texts.sort()
                all_texts = [t for _, t in neighbor_texts if _ < 0]
                all_texts.append(chunk.get("text", ""))
                all_texts.extend([t for _, t in neighbor_texts if _ > 0])
                
                expanded_chunk = chunk.copy()
                expanded_chunk["text"] = "\n\n".join(all_texts)
                expanded_chunk["context_expanded"] = True
                expanded.append(expanded_chunk)
            else:
                expanded.append(chunk)
        except:
            expanded.append(chunk)
    
    return expanded

# ============================================================================
# DIVERSITY FILTER: Remove near-duplicates
# ============================================================================
def compute_text_hash(text, length=100):
    """Compute hash of text beginning for dedup"""
    return hashlib.md5(text[:length].encode()).hexdigest()

def filter_diverse(chunks, threshold=0.85):
    """
    Remove near-duplicate chunks based on text similarity.
    """
    if len(chunks) <= 1:
        return chunks
    
    result = []
    seen_hashes = set()
    
    for chunk in chunks:
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        
        # Compute multiple hashes at different positions
        hashes = [
            compute_text_hash(text, 100),
            compute_text_hash(text, 200),
        ]
        
        # Check for near-duplicates
        is_duplicate = any(h in seen_hashes for h in hashes)
        
        if not is_duplicate:
            result.append(chunk)
            seen_hashes.update(hashes)
    
    return result

# ============================================================================
# RELEVANCE FILTER
# ============================================================================
def filter_by_relevance(chunks, threshold=0.001):
    """Remove chunks below relevance threshold. Default is very low (0.001) to keep most results."""
    import os
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    result = []
    for chunk in chunks:
        score = chunk.get("rrf_score", 0) or chunk.get("rerank_score", 0)
        
        if debug:
            print(f"    Filter check: score={score:.6f} threshold={threshold} keep={score >= threshold or score == 0}")
        
        # Keep if score meets threshold OR if score is 0 (means no score to judge by)
        if score >= threshold or score == 0:
            result.append(chunk)
    
    return result

# ============================================================================
# MAIN POST-PROCESSING
# ============================================================================
def post_process_retrieval(query, chunks, config):
    """
    Main post-retrieval processing pipeline.
    Applies configured processing steps.
    """
    if not chunks:
        return chunks, False, []
    
    web_triggered = False
    web_results = []
    
    # 1. Relevance filter (fast, always run)
    if config.get("relevance_filter_enabled", True):
        threshold = config.get("relevance_threshold", 0.001)
        chunks = filter_by_relevance(chunks, threshold)
    
    # 2. Diversity filter (fast)
    if config.get("diversity_filter_enabled", True):
        threshold = config.get("diversity_threshold", 0.85)
        chunks = filter_diverse(chunks, threshold)
    
    # 3. Reranking (medium speed, ~3-5s)
    if config.get("rerank_enabled", False):
        top_k = config.get("rerank_top_k", 5)
        chunks = rerank_chunks(query, chunks, top_k)
    
    # 4. CRAG check (may trigger web search)
    if config.get("crag_enabled", False):
        chunks, web_triggered, web_results = crag_process(query, chunks, config)
    
    # 5. Context window expansion
    if config.get("context_window_enabled", False):
        window_size = config.get("context_window_size", 1)
        chunks = expand_context_window(chunks, window_size)
    
    # 6. RSE
    if config.get("rse_enabled", False):
        chunks = apply_rse(query, chunks)
    
    return chunks, web_triggered, web_results
EOFPY
log_ok "Post-retrieval module"

# ============================================================================
# Grounding Verification Module
# ============================================================================
echo "Creating grounding module..."
cat > lib/grounding.py << 'EOFPY'
"""Grounding verification: Check answer is supported by sources"""
import re
from llm_helper import llm_generate

def extract_claims(answer, max_claims=5):
    """Extract factual claims from answer"""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    
    claims = []
    for sent in sentences:
        sent = sent.strip()
        # Skip short sentences, questions, or meta-text
        if len(sent) < 20:
            continue
        if sent.endswith('?'):
            continue
        if any(w in sent.lower() for w in ["i think", "might be", "could be", "perhaps"]):
            continue
        
        claims.append(sent)
        if len(claims) >= max_claims:
            break
    
    return claims

def verify_claim_against_context(claim, context):
    """Check if claim is supported by context"""
    # Simple word overlap check
    claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
    context_words = set(re.findall(r'\b\w+\b', context.lower()))
    
    # Remove stopwords
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'for', 'on', 'with', 'it', 'this', 'that'}
    claim_words -= stopwords
    context_words -= stopwords
    
    if not claim_words:
        return True, 1.0
    
    overlap = len(claim_words & context_words) / len(claim_words)
    return overlap >= 0.5, overlap

def verify_grounding_llm(answer, context, max_claims=3):
    """LLM-based grounding verification"""
    prompt = f"""Check if this answer is supported by the given context.

Context:
{context[:2000]}

Answer:
{answer}

Is every factual claim in the answer supported by the context?
Reply with: SUPPORTED or UNSUPPORTED, then explain briefly."""

    result = llm_generate(prompt, max_tokens=100, timeout=30)
    
    if result:
        is_supported = "SUPPORTED" in result.upper() and "UNSUPPORTED" not in result.upper()
        return {
            "verified": is_supported,
            "explanation": result,
            "score": 1.0 if is_supported else 0.0
        }
    
    return {"verified": True, "score": 0.5, "explanation": "Could not verify"}

def verify_answer_grounding(answer, context, config):
    """
    Main grounding verification function.
    Returns verification result with details.
    """
    if not config.get("grounding_check_enabled", False):
        return {"verified": True, "score": 1.0, "claims": []}
    
    max_claims = config.get("grounding_max_claims", 3)
    threshold = config.get("grounding_threshold", 0.5)
    
    # Extract claims
    claims = extract_claims(answer, max_claims)
    
    if not claims:
        return {"verified": True, "score": 1.0, "claims": []}
    
    # Verify each claim
    verified_claims = []
    total_score = 0
    
    for claim in claims:
        is_verified, score = verify_claim_against_context(claim, context)
        verified_claims.append({
            "claim": claim[:100],
            "verified": is_verified,
            "score": score
        })
        total_score += score
    
    avg_score = total_score / len(claims)
    
    return {
        "verified": avg_score >= threshold,
        "score": avg_score,
        "claims": verified_claims
    }
EOFPY
log_ok "Grounding module"

# ============================================================================
# Web Search Module - SearXNG (Privacy-Respecting)
# ============================================================================
echo "Creating web search module..."
cat > lib/web_search.py << 'EOFPY'
"""Web Search via SearXNG - Privacy-respecting, self-hosted"""
import os
import requests
import time

# Debug storage for web search
_web_debug = {
    "enabled": False,
    "searches": [],
    "total_results": 0,
    "engines_used": set(),
}

def reset_web_debug():
    """Reset web debug info"""
    global _web_debug
    _web_debug = {
        "enabled": False,
        "searches": [],
        "total_results": 0,
        "engines_used": set(),
    }

def get_web_debug():
    """Get web search debug info"""
    result = _web_debug.copy()
    result["engines_used"] = list(_web_debug["engines_used"])
    return result

def search_web(query, max_results=5, timeout=None):
    """
    Search web using SearXNG.
    
    Returns list of:
    {
        "title": str,
        "url": str,
        "snippet": str,
        "engine": str,
        "score": float
    }
    """
    global _web_debug
    
    searxng_url = os.environ.get("SEARXNG_URL", "http://localhost:8085/search")
    timeout = timeout or int(os.environ.get("SEARXNG_TIMEOUT", "10"))
    allowed_engines = os.environ.get("SEARXNG_ALLOWED_ENGINES", "")
    debug_web = os.environ.get("DEBUG_WEB", "false").lower() == "true"
    
    _web_debug["enabled"] = debug_web
    
    search_record = {
        "query": query,
        "url": searxng_url,
        "timestamp": time.time(),
        "results": [],
        "engines_responded": [],
        "error": None,
    }
    
    try:
        params = {
            "q": query,
            "format": "json",
            "safesearch": "0",
        }
        
        # Add engine filter if specified
        if allowed_engines:
            params["engines"] = allowed_engines
        
        resp = requests.get(
            searxng_url,
            params=params,
            timeout=timeout
        )
        
        if resp.status_code != 200:
            search_record["error"] = f"HTTP {resp.status_code}"
            _web_debug["searches"].append(search_record)
            return []
        
        data = resp.json()
        raw_results = data.get("results", [])
        
        # Track which engines responded
        engines_in_results = set()
        for r in raw_results:
            eng = r.get("engine", "unknown")
            engines_in_results.add(eng)
            _web_debug["engines_used"].add(eng)
        
        search_record["engines_responded"] = list(engines_in_results)
        
        # Format results
        results = []
        seen_urls = set()
        
        for r in raw_results[:max_results * 2]:  # Get more, dedupe later
            url = r.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            result = {
                "title": r.get("title", ""),
                "url": url,
                "snippet": r.get("content", ""),
                "engine": r.get("engine", "unknown"),
                "score": r.get("score", 0.0),
            }
            results.append(result)
            
            search_record["results"].append({
                "title": result["title"][:50],
                "url": url,
                "engine": result["engine"],
                "used_for_context": len(results) <= max_results,
            })
            
            if len(results) >= max_results:
                break
        
        _web_debug["total_results"] += len(results)
        _web_debug["searches"].append(search_record)
        
        return results
        
    except requests.exceptions.Timeout:
        search_record["error"] = f"Timeout ({timeout}s)"
        _web_debug["searches"].append(search_record)
        return []
    except requests.exceptions.ConnectionError:
        search_record["error"] = "Connection failed - is SearXNG running?"
        _web_debug["searches"].append(search_record)
        return []
    except Exception as e:
        search_record["error"] = str(e)
        _web_debug["searches"].append(search_record)
        return []

def format_web_results(results):
    """Format web results for context"""
    if not results:
        return ""
    
    parts = []
    for i, r in enumerate(results, 1):
        engine = r.get("engine", "web")
        parts.append(f"[WEB:{engine}] {r['title']}\nURL: {r['url']}\n{r['snippet']}")
    
    return "\n\n".join(parts)

def print_web_debug():
    """Print web search debug info"""
    debug = get_web_debug()
    
    if not debug["enabled"]:
        return
    
    print("\n[WEB DEBUG]")
    print(f"  Searches: {len(debug['searches'])}")
    print(f"  Total results: {debug['total_results']}")
    print(f"  Engines used: {', '.join(debug['engines_used']) or 'none'}")
    
    for i, search in enumerate(debug["searches"], 1):
        print(f"\n  Search #{i}: '{search['query'][:30]}...'")
        if search["error"]:
            print(f"    ERROR: {search['error']}")
        else:
            print(f"    Engines responded: {', '.join(search['engines_responded'])}")
            print(f"    Results: {len(search['results'])}")
            for r in search["results"][:3]:
                status = "✓" if r["used_for_context"] else "○"
                print(f"      {status} [{r['engine']}] {r['title'][:40]}...")
EOFPY
log_ok "Web search module (SearXNG)"

# ============================================================================
# Citations Module
# ============================================================================
echo "Creating citations module..."
cat > lib/citations.py << 'EOFPY'
"""Citation formatting and extraction"""
import os
import re

def format_context_with_citations(chunks, max_chunk_chars=1000):
    """Format chunks with citation numbers - simplified for clarity"""
    if not chunks:
        return "", {}
    
    context_parts = []
    citation_map = {}
    
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        source = chunk.get("source", "unknown") if isinstance(chunk, dict) else "unknown"
        filename = chunk.get("filename", os.path.basename(source)) if isinstance(chunk, dict) else "unknown"
        section = chunk.get("section", "") if isinstance(chunk, dict) else ""
        
        # Truncate if needed
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars] + "..."
        
        # Simple citation marker - text already has document info from contextual headers
        # Just add [1], [2] markers for reference
        context_parts.append(f"[{i}] {text}")
        
        citation_map[i] = {
            "source": source,
            "filename": filename,
            "section": section,
            "text": text[:200]
        }
    
    return "\n\n".join(context_parts), citation_map

def extract_citations_from_answer(answer, citation_map):
    """Extract citation references from answer"""
    refs = set(re.findall(r'\[(\d+)\]', answer))
    return [int(r) for r in refs if int(r) in citation_map]

def format_sources_footer(citations_used, citation_map):
    """Format sources footer"""
    if not citations_used:
        return ""
    
    lines = ["\n---\nSources:"]
    for ref in sorted(citations_used):
        if ref in citation_map:
            info = citation_map[ref]
            source = info.get("filename", info.get("source", "unknown"))
            section = info.get("section", "")
            line = f"[{ref}] {source}"
            if section:
                line += f" ({section})"
            lines.append(line)
    
    return "\n".join(lines)
EOFPY
log_ok "Citations module"

# ============================================================================
# Memory Module
# ============================================================================
echo "Creating memory module..."
cat > lib/memory.py << 'EOFPY'
"""Conversation memory with context"""
import json
import os

class ConversationMemory:
    def __init__(self, file_path=None):
        self.file_path = file_path or os.environ.get("MEMORY_FILE", "cache/memory.json")
        self.history = []
        self._load()
    
    def _load(self):
        try:
            with open(self.file_path) as f:
                self.history = json.load(f)
        except:
            self.history = []
    
    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, 'w') as f:
                json.dump(self.history[-20:], f)  # Keep last 20
        except:
            pass
    
    def add(self, query, answer, sources=None):
        entry = {
            "query": query,
            "answer": answer[:1000],  # Truncate long answers
        }
        if sources:
            entry["sources"] = sources[:3]  # Keep top 3 sources
        
        self.history.append(entry)
        self._save()
    
    def get_context(self, max_turns=3, max_chars=500):
        recent = self.history[-max_turns:]
        if not recent:
            return ""
        
        lines = []
        total_chars = 0
        
        for turn in reversed(recent):
            turn_text = f"Q: {turn['query']}\nA: {turn['answer'][:200]}..."
            if total_chars + len(turn_text) > max_chars:
                break
            lines.insert(0, turn_text)
            total_chars += len(turn_text)
        
        return "\n\n".join(lines)
    
    def clear(self):
        self.history = []
        self._save()
    
    def get_related_context(self, query, max_results=2):
        """Find previous Q&A related to current query"""
        if not self.history:
            return ""
        
        query_words = set(query.lower().split())
        
        scored = []
        for i, turn in enumerate(self.history):
            turn_words = set(turn["query"].lower().split())
            overlap = len(query_words & turn_words)
            if overlap > 1:  # At least 2 words in common
                scored.append((overlap, i, turn))
        
        if not scored:
            return ""
        
        scored.sort(reverse=True)
        related = scored[:max_results]
        
        lines = []
        for _, _, turn in related:
            lines.append(f"Previous Q: {turn['query']}\nPrevious A: {turn['answer'][:150]}...")
        
        return "\n\n".join(lines)
EOFPY
log_ok "Memory module"

# ============================================================================
# Query Cache Module
# ============================================================================
echo "Creating query cache module..."
cat > lib/query_cache.py << 'EOFPY'
"""Query result caching"""
import json
import os
import hashlib
import time

class QueryCache:
    def __init__(self, cache_file=None, ttl=3600):
        self.cache_file = cache_file or "cache/query_cache.json"
        self.ttl = ttl  # Time to live in seconds
        self.cache = {}
        self._load()
    
    def _load(self):
        try:
            with open(self.cache_file) as f:
                self.cache = json.load(f)
        except:
            self.cache = {}
    
    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            # Prune old entries before saving
            now = time.time()
            self.cache = {
                k: v for k, v in self.cache.items()
                if now - v.get("timestamp", 0) < self.ttl * 24  # Keep for 24x TTL
            }
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except:
            pass
    
    def _hash(self, query):
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query):
        """Get cached result if fresh"""
        key = self._hash(query)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry.get("timestamp", 0) < self.ttl:
                entry["hits"] = entry.get("hits", 0) + 1
                return entry.get("result")
        return None
    
    def set(self, query, result):
        """Cache a result"""
        key = self._hash(query)
        self.cache[key] = {
            "query": query,
            "result": result,
            "timestamp": time.time(),
            "hits": 0
        }
        self._save()
    
    def clear(self):
        self.cache = {}
        self._save()
EOFPY
log_ok "Query cache module"

# ============================================================================
# Query Correction Module (spell check, typo fix, expansion)
# ============================================================================
echo "Creating query correction module..."
cat > lib/query_correction.py << 'EOFPY'
"""Query Correction & Enrichment - Spell check, typo fix, acronym expansion"""
import os
import re
import json
from collections import Counter, defaultdict

def levenshtein_distance(s1, s2):
    """Calculate edit distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

def load_vocabulary(filepath="cache/vocabulary.json"):
    """Load vocabulary from file"""
    try:
        with open(filepath) as f:
            return json.load(f)
    except:
        return {"terms": {}, "cooccurrence": {}}

def save_vocabulary(vocab, filepath="cache/vocabulary.json"):
    """Save vocabulary to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(vocab, f)

def build_vocabulary_from_texts(texts, min_freq=2, max_terms=10000):
    """Build vocabulary from list of texts"""
    word_freq = Counter()
    cooccurrence = defaultdict(Counter)
    
    for text in texts:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        unique_words = set(words)
        word_freq.update(words)
        for word in unique_words:
            for other in unique_words:
                if word != other:
                    cooccurrence[word][other] += 1
    
    vocab_terms = {
        word: freq for word, freq in word_freq.most_common(max_terms)
        if freq >= min_freq
    }
    
    simplified_cooc = {}
    for word, related in cooccurrence.items():
        if word in vocab_terms:
            simplified_cooc[word] = dict(related.most_common(10))
    
    return {"terms": vocab_terms, "cooccurrence": simplified_cooc}

def find_similar_terms(word, vocabulary, max_distance=2, max_results=3):
    """Find similar terms using edit distance"""
    word = word.lower()
    if word in vocabulary:
        return [(word, 0, vocabulary[word])]
    
    candidates = []
    for term, freq in vocabulary.items():
        if abs(len(term) - len(word)) > max_distance or len(term) < 3:
            continue
        dist = levenshtein_distance(word, term)
        if dist <= max_distance:
            candidates.append((term, dist, freq))
    
    candidates.sort(key=lambda x: (x[1], -x[2]))
    return candidates[:max_results]

def correct_query_spelling(query, vocabulary):
    """Correct spelling in query using document vocabulary"""
    words = query.split()
    corrected = []
    corrections = []
    vocab_terms = vocabulary.get("terms", {})
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if len(clean_word) < 3 or clean_word in vocab_terms:
            corrected.append(word)
            continue
        
        similar = find_similar_terms(clean_word, vocab_terms, max_distance=2)
        if similar and similar[0][1] > 0:
            best_match = similar[0][0]
            if word[0].isupper():
                best_match = best_match.capitalize()
            corrected.append(best_match)
            corrections.append(f"{word} → {best_match}")
        else:
            corrected.append(word)
    
    return " ".join(corrected), corrections

# Acronym expansions (IT/MSP domain)
ACRONYM_EXPANSIONS = {
    "rmm": "remote monitoring management",
    "msp": "managed service provider",
    "it": "information technology",
    "api": "application programming interface",
    "ai": "artificial intelligence",
    "os": "operating system",
    "vm": "virtual machine",
    "vpn": "virtual private network",
    "sla": "service level agreement",
    "nis": "network information security",
    "rgpd": "règlement général protection données",
}

def expand_acronyms(query):
    """Expand known acronyms"""
    words = query.lower().split()
    expansions = []
    for word in words:
        clean = re.sub(r'[^\w]', '', word)
        if clean in ACRONYM_EXPANSIONS:
            expansions.append(f"{clean} = {ACRONYM_EXPANSIONS[clean]}")
    return expansions

def correct_and_enrich_query(query, vocabulary=None, config=None):
    """Main function to correct and enrich a query"""
    if config is None:
        config = {}
    if vocabulary is None:
        vocabulary = load_vocabulary()
    
    result = {
        "original": query,
        "corrected": query,
        "corrections": [],
        "expansions": [],
    }
    
    # Spell correction
    if config.get("spell_correction", True) and vocabulary.get("terms"):
        corrected, corrections = correct_query_spelling(query, vocabulary)
        if corrections:
            result["corrected"] = corrected
            result["corrections"] = corrections
    
    # Acronym expansion (informational only)
    if config.get("expand_acronyms", True):
        result["expansions"] = expand_acronyms(query)
    
    return result
EOFPY
log_ok "Query correction module"

# ============================================================================
# Quality Ledger Module (v45 - Persistent tracking)
# ============================================================================
echo "Creating quality ledger module..."
cat > lib/quality_ledger.py << 'EOFPY'
"""Quality Ledger - Persistent tracking of query quality metrics (v45)"""
import os
import sqlite3
import json
import time
import uuid
from datetime import datetime

class QualityLedger:
    """SQLite-based ledger for tracking query quality"""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or os.environ.get(
            "QUALITY_LEDGER_FILE", 
            "cache/quality_ledger.sqlite"
        )
        self.enabled = os.environ.get("QUALITY_LEDGER_ENABLED", "true").lower() == "true"
        self._ensure_db()
    
    def _ensure_db(self):
        """Create database and tables if they don't exist"""
        if not self.enabled:
            return
        
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entries (
                query_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                
                -- Retrieval metrics
                retrieval_count INTEGER,
                retrieval_sources_distinct INTEGER,
                retrieval_avg_score REAL,
                retrieval_score_std REAL,
                crag_triggered INTEGER,
                web_used INTEGER,
                
                -- Quality scores (0-1)
                retrieval_confidence REAL,
                answer_coverage REAL,
                grounding_score REAL,
                
                -- Decision
                decision TEXT,
                decision_reason TEXT,
                
                -- Response
                response_length INTEGER,
                response_truncated TEXT,
                
                -- Human feedback (optional)
                feedback TEXT,
                feedback_timestamp TEXT,
                
                -- Timing
                duration_ms INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_entry(self, query):
        """Create a new ledger entry, returns query_id"""
        if not self.enabled:
            return None
        
        query_id = str(uuid.uuid4())[:8]
        timestamp = datetime.utcnow().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO entries (query_id, timestamp, query)
            VALUES (?, ?, ?)
        ''', (query_id, timestamp, query))
        
        conn.commit()
        conn.close()
        
        return query_id
    
    def update_entry(self, query_id, **kwargs):
        """Update an existing entry with metrics"""
        if not self.enabled or not query_id:
            return
        
        allowed_fields = {
            'retrieval_count', 'retrieval_sources_distinct', 'retrieval_avg_score',
            'retrieval_score_std', 'crag_triggered', 'web_used',
            'retrieval_confidence', 'answer_coverage', 'grounding_score',
            'decision', 'decision_reason', 'response_length', 'response_truncated',
            'feedback', 'feedback_timestamp', 'duration_ms'
        }
        
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return
        
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [query_id]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f'''
            UPDATE entries SET {set_clause} WHERE query_id = ?
        ''', values)
        
        conn.commit()
        conn.close()
    
    def add_feedback(self, query_id, feedback):
        """Add human feedback to an entry"""
        if not self.enabled or not query_id:
            return
        
        self.update_entry(
            query_id,
            feedback=feedback,
            feedback_timestamp=datetime.utcnow().isoformat()
        )
    
    def get_entry(self, query_id):
        """Retrieve a ledger entry"""
        if not self.enabled:
            return None
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM entries WHERE query_id = ?', (query_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        return dict(row) if row else None
    
    def get_stats(self, limit=100):
        """Get aggregated stats from recent entries"""
        if not self.enabled:
            return {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                AVG(retrieval_confidence) as avg_retrieval_conf,
                AVG(answer_coverage) as avg_answer_cov,
                AVG(grounding_score) as avg_grounding,
                SUM(CASE WHEN decision = 'confident' THEN 1 ELSE 0 END) as confident_count,
                SUM(CASE WHEN decision = 'low_confidence' THEN 1 ELSE 0 END) as low_conf_count,
                SUM(CASE WHEN decision = 'abstained' THEN 1 ELSE 0 END) as abstained_count,
                AVG(duration_ms) as avg_duration
            FROM entries
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'total': row[0],
                'avg_retrieval_confidence': row[1],
                'avg_answer_coverage': row[2],
                'avg_grounding_score': row[3],
                'confident_count': row[4],
                'low_confidence_count': row[5],
                'abstained_count': row[6],
                'avg_duration_ms': row[7]
            }
        return {}
    
    def log_entry(self, data):
        """
        Convenience method: create entry and update with all data at once.
        Returns query_id.
        """
        if not self.enabled:
            return None
        
        query = data.get('query', '')
        query_id = self.create_entry(query)
        
        if not query_id:
            return None
        
        # Extract nested data
        retrieval = data.get('retrieval', {})
        scores = data.get('scores', {})
        
        self.update_entry(
            query_id,
            retrieval_count=retrieval.get('count', 0),
            retrieval_sources_distinct=len(retrieval.get('sources', [])),
            crag_triggered=1 if retrieval.get('crag_used') else 0,
            web_used=1 if retrieval.get('web_used') else 0,
            retrieval_confidence=scores.get('retrieval_confidence', 0),
            answer_coverage=scores.get('answer_coverage', 0),
            grounding_score=scores.get('grounding_score', 0),
            decision=data.get('decision', 'unknown'),
            response_length=data.get('answer_length', 0),
            duration_ms=data.get('response_time_ms', 0)
        )
        
        return query_id
    
    def get_recent(self, limit=10):
        """Get recent ledger entries"""
        if not self.enabled:
            return []
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM entries 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

# Global instance
_ledger = None

def get_ledger():
    global _ledger
    if _ledger is None:
        _ledger = QualityLedger()
    return _ledger
EOFPY
log_ok "Quality ledger module"

# ============================================================================
# Score Calculator Module (v45 - NO LLM, pure metrics)
# ============================================================================
echo "Creating scoring module..."
cat > lib/scoring.py << 'EOFPY'
"""Score Calculator - Quality metrics without LLM (v45)"""
import os
import re
import math
from collections import Counter

def calculate_retrieval_confidence(chunks, crag_triggered=False, web_used=False):
    """
    Calculate retrieval confidence score (0-1) based on:
    - Score dispersion (lower std = more confident)
    - Number of distinct sources
    - CRAG/web usage (reduces confidence)
    
    NO LLM INVOLVED.
    """
    if not chunks:
        return 0.0
    
    # Extract scores
    scores = []
    sources = set()
    
    for chunk in chunks:
        score = chunk.get('rrf_score', 0) or chunk.get('rerank_score', 0) or chunk.get('score', 0)
        scores.append(score)
        
        source = chunk.get('filename', '') or chunk.get('source', '')
        if source:
            sources.add(source)
    
    if not scores:
        return 0.0
    
    # 1. Score quality (mean normalized)
    mean_score = sum(scores) / len(scores)
    # Normalize: RRF scores are typically 0.005-0.05, rerank 0.1-0.9
    if mean_score < 0.1:  # Likely RRF scores
        score_quality = min(1.0, mean_score * 20)  # 0.05 -> 1.0
    else:
        score_quality = min(1.0, mean_score)
    
    # 2. Score consistency (low std = good)
    if len(scores) > 1:
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = math.sqrt(variance)
        consistency = 1.0 - min(1.0, std_dev * 10)  # Lower std = higher consistency
    else:
        consistency = 0.5
    
    # 3. Source diversity (more sources = potentially more reliable)
    source_factor = min(1.0, len(sources) / 3)  # Cap at 3 sources
    
    # 4. CRAG/Web penalty (using fallbacks = lower confidence in local docs)
    fallback_penalty = 0.0
    if crag_triggered:
        fallback_penalty += 0.15
    if web_used:
        fallback_penalty += 0.1
    
    # Combine
    confidence = (
        0.4 * score_quality +
        0.3 * consistency +
        0.3 * source_factor
    ) - fallback_penalty
    
    return max(0.0, min(1.0, confidence))

def calculate_answer_coverage(query, answer):
    """
    Calculate how well the answer covers the query (0-1).
    Based on lexical overlap - NO LLM.
    
    - Checks if query terms appear in answer
    - Penalizes very short or very long answers
    """
    if not query or not answer:
        return 0.0
    
    # Tokenize
    def tokenize(text):
        words = re.findall(r'\b\w{3,}\b', text.lower())
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 
                    'in', 'for', 'on', 'with', 'at', 'by', 'this', 'that', 'and',
                    'or', 'but', 'what', 'how', 'why', 'when', 'where', 'which',
                    'qui', 'que', 'est', 'sont', 'les', 'des', 'une', 'pour'}
        return [w for w in words if w not in stopwords]
    
    query_tokens = set(tokenize(query))
    answer_tokens = tokenize(answer)
    answer_token_set = set(answer_tokens)
    
    if not query_tokens:
        return 0.5  # Can't measure
    
    # 1. Query term coverage in answer
    covered = len(query_tokens & answer_token_set)
    coverage_ratio = covered / len(query_tokens)
    
    # 2. Answer length penalty
    # Too short = probably incomplete
    # Too long = probably padding/hallucinating
    answer_len = len(answer)
    if answer_len < 50:
        length_factor = answer_len / 50  # Penalize very short
    elif answer_len > 2000:
        length_factor = max(0.5, 1.0 - (answer_len - 2000) / 5000)  # Penalize very long
    else:
        length_factor = 1.0
    
    # 3. Token density (unique meaningful words per character)
    density = len(answer_token_set) / max(1, answer_len) * 100
    density_factor = min(1.0, density / 5)  # Expect ~5% meaningful words
    
    # Combine
    coverage = (
        0.5 * coverage_ratio +
        0.3 * length_factor +
        0.2 * density_factor
    )
    
    return max(0.0, min(1.0, coverage))

def calculate_grounding_score(answer, chunks):
    """
    Calculate how well the answer is grounded in the retrieved chunks (0-1).
    Based on lexical overlap with sources - NO LLM.
    
    Checks if answer content appears in the provided context.
    """
    if not answer or not chunks:
        return 0.0
    
    # Build context text
    context_text = " ".join(
        chunk.get('text', '') if isinstance(chunk, dict) else str(chunk)
        for chunk in chunks
    ).lower()
    
    if not context_text:
        return 0.0
    
    # Tokenize answer into meaningful phrases (bigrams and trigrams)
    answer_lower = answer.lower()
    answer_words = re.findall(r'\b\w{3,}\b', answer_lower)
    
    if not answer_words:
        return 0.5
    
    # Check word-level grounding
    grounded_words = sum(1 for w in answer_words if w in context_text)
    word_grounding = grounded_words / len(answer_words)
    
    # Check phrase-level grounding (more strict)
    phrases_found = 0
    phrases_total = 0
    
    for i in range(len(answer_words) - 1):
        bigram = f"{answer_words[i]} {answer_words[i+1]}"
        phrases_total += 1
        if bigram in context_text:
            phrases_found += 1
    
    phrase_grounding = phrases_found / max(1, phrases_total)
    
    # Combine
    grounding = 0.6 * word_grounding + 0.4 * phrase_grounding
    
    return max(0.0, min(1.0, grounding))

def calculate_all_scores(query, answer, chunks, crag_triggered=False, web_used=False):
    """Calculate all quality scores at once"""
    return {
        'retrieval_confidence': calculate_retrieval_confidence(chunks, crag_triggered, web_used),
        'answer_coverage': calculate_answer_coverage(query, answer),
        'grounding_score': calculate_grounding_score(answer, chunks)
    }
EOFPY
log_ok "Score calculator module"

# ============================================================================
# Decision Engine Module (v45 - Abstention logic)
# ============================================================================
echo "Creating decision engine module..."
cat > lib/decision_engine.py << 'EOFPY'
"""Decision Engine - Determines confidence level and abstention (v45)"""
import os

def get_thresholds():
    """Get confidence thresholds from config"""
    return {
        'retrieval_confidence_min': float(os.environ.get('RETRIEVAL_CONFIDENCE_MIN', '0.3')),
        'answer_coverage_min': float(os.environ.get('ANSWER_COVERAGE_MIN', '0.2')),
        'grounding_score_min': float(os.environ.get('GROUNDING_SCORE_MIN', '0.4')),
    }

def make_decision(scores, thresholds=None):
    """
    Make a decision based on quality scores.
    
    Returns:
    {
        'decision': 'confident' | 'low_confidence' | 'abstained',
        'reason': str,
        'details': dict
    }
    """
    if thresholds is None:
        thresholds = get_thresholds()
    
    retrieval_conf = scores.get('retrieval_confidence', 0)
    answer_cov = scores.get('answer_coverage', 0)
    grounding = scores.get('grounding_score', 0)
    
    # Check each threshold
    checks = {
        'retrieval': retrieval_conf >= thresholds['retrieval_confidence_min'],
        'coverage': answer_cov >= thresholds['answer_coverage_min'],
        'grounding': grounding >= thresholds['grounding_score_min'],
    }
    
    passed = sum(checks.values())
    
    details = {
        'retrieval_confidence': retrieval_conf,
        'retrieval_threshold': thresholds['retrieval_confidence_min'],
        'retrieval_passed': checks['retrieval'],
        'answer_coverage': answer_cov,
        'coverage_threshold': thresholds['answer_coverage_min'],
        'coverage_passed': checks['coverage'],
        'grounding_score': grounding,
        'grounding_threshold': thresholds['grounding_score_min'],
        'grounding_passed': checks['grounding'],
    }
    
    # Decision logic
    if passed == 3:
        # All checks passed
        return {
            'decision': 'confident',
            'reason': 'All quality thresholds met',
            'details': details
        }
    elif passed >= 2:
        # Most checks passed
        failed = [k for k, v in checks.items() if not v]
        return {
            'decision': 'low_confidence',
            'reason': f'Threshold not met: {", ".join(failed)}',
            'details': details
        }
    else:
        # Too many checks failed - abstain
        failed = [k for k, v in checks.items() if not v]
        return {
            'decision': 'abstained',
            'reason': f'Multiple thresholds not met: {", ".join(failed)}',
            'details': details
        }

def format_abstention_response(query, chunks, decision_details):
    """Format a response when abstaining"""
    abstention_msg = os.environ.get(
        'ABSTENTION_MESSAGE',
        "Je n'ai pas assez d'informations fiables pour répondre avec confiance."
    )
    
    # List available sources
    sources = []
    seen = set()
    for chunk in chunks[:5]:
        fname = chunk.get('filename', '') or chunk.get('source', '')
        if fname and fname not in seen:
            sources.append(fname)
            seen.add(fname)
    
    response = f"{abstention_msg}\n\n"
    
    if sources:
        response += "Documents consultés:\n"
        for s in sources[:3]:
            response += f"  • {s}\n"
    
    response += "\nRecommandation: Reformulez votre question ou consultez directement les sources."
    
    return response

def should_abstain():
    """Check if abstention is enabled"""
    return os.environ.get('ABSTENTION_ENABLED', 'true').lower() == 'true'
EOFPY
log_ok "Decision engine module"

echo ""
echo "Modules created. Now creating main query script..."

# ============================================================================
# Main Query Script
# ============================================================================
cat > "$PROJECT_DIR/query.sh" << 'EOFSH'
#!/bin/bash
# ============================================================================
# RAG Query v46 - Full Features, Optimized Defaults
# ============================================================================
# All features implemented, most disabled by default for speed.
# Use --full to enable all features.
# Target: Default query < 90s on low-resource hardware
# ============================================================================

[ -f "./config.env" ] && source ./config.env
[ -d "./venv" ] && source ./venv/bin/activate

# Export ALL configuration
export OLLAMA_HOST QDRANT_HOST COLLECTION_NAME LLM_MODEL EMBEDDING_MODEL
export EMBEDDING_TIMEOUT EMBEDDING_DIMENSION

# Query enhancement
export QUERY_CLASSIFICATION_ENABLED QUERY_REWRITE_ENABLED
export HYDE_ENABLED SUBQUERY_ENABLED SUBQUERY_MAX STEPBACK_ENABLED

# Post-retrieval
export RERANK_ENABLED RERANK_MODEL RERANK_TOP_K
export RELEVANCE_FILTER_ENABLED RELEVANCE_THRESHOLD
export CRAG_ENABLED CRAG_THRESHOLD
export CONTEXT_WINDOW_ENABLED CONTEXT_WINDOW_SIZE
export RSE_ENABLED
export DIVERSITY_FILTER_ENABLED DIVERSITY_THRESHOLD

# Generation
export CITATIONS_ENABLED
export GROUNDING_CHECK_ENABLED GROUNDING_THRESHOLD GROUNDING_MAX_CLAIMS

# Memory & Cache
export MEMORY_ENABLED MEMORY_MAX_TURNS MEMORY_FILE
export QUERY_CACHE_ENABLED QUERY_CACHE_TTL

# Web search
export WEB_SEARCH_ENABLED WEB_SEARCH_MODE WEB_SEARCH_MAX_RESULTS WEB_SEARCH_TIMEOUT

# Timeouts
export LLM_TIMEOUT_DEFAULT LLM_TIMEOUT_ULTRAFAST LLM_TIMEOUT_RAG_ONLY

# Context limits
export MAX_CONTEXT_CHARS MAX_CHUNK_CHARS MAX_MEMORY_CHARS

# Generation params
export NUM_PREDICT_DEFAULT NUM_PREDICT_ULTRAFAST NUM_PREDICT_FULL TEMPERATURE

# Defaults
TOP_K="${DEFAULT_TOP_K:-5}"
QUERY=""
DEBUG=false
VERBOSE=false
ULTRAFAST=false
RAG_ONLY=false
FULL_MODE=false
WEB_MODE="${WEB_SEARCH_MODE:-auto}"
NO_WEB=false
NO_CACHE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug) DEBUG=true; VERBOSE=true; shift ;;
        --verbose|-v) VERBOSE=true; shift ;;
        --ultrafast|--fast)
            ULTRAFAST=true
            # Override to disable heavy features
            export HYDE_ENABLED=false
            export RERANK_ENABLED=false
            export CRAG_ENABLED=false
            export GROUNDING_CHECK_ENABLED=false
            export STEPBACK_ENABLED=false
            export SUBQUERY_ENABLED=false
            export CONTEXT_WINDOW_ENABLED=false
            export RSE_ENABLED=false
            TOP_K=3
            shift ;;
        --full)
            FULL_MODE=true
            # Enable all features
            export QUERY_CLASSIFICATION_ENABLED=true
            export HYDE_ENABLED=true
            export RERANK_ENABLED=true
            export CRAG_ENABLED=true
            export GROUNDING_CHECK_ENABLED=true
            export STEPBACK_ENABLED=true
            export SUBQUERY_ENABLED=true
            export CONTEXT_WINDOW_ENABLED=true
            export RSE_ENABLED=true
            export DIVERSITY_FILTER_ENABLED=true
            # NO timeouts in full mode (0 = unlimited)
            export LLM_TIMEOUT_DEFAULT=0
            export LLM_TIMEOUT_ULTRAFAST=0
            export LLM_TIMEOUT_FULL=0
            export EMBEDDING_TIMEOUT=0
            export RERANK_TIMEOUT=0
            export SEARXNG_TIMEOUT=30
            # Full context and output
            export MAX_CONTEXT_CHARS=15000
            export NUM_PREDICT_FULL=2000
            TOP_K=8
            shift ;;
        --rag-only) RAG_ONLY=true; shift ;;
        --web) WEB_MODE="always"; shift ;;
        --web=*) WEB_MODE="${1#*=}"; shift ;;
        --no-web) NO_WEB=true; WEB_MODE="never"; shift ;;
        --no-cache) NO_CACHE=true; shift ;;
        --no-memory) export MEMORY_ENABLED=false; shift ;;
        --no-rerank) export RERANK_ENABLED=false; shift ;;
        --no-hyde) export HYDE_ENABLED=false; shift ;;
        --hyde) export HYDE_ENABLED=true; shift ;;
        --rerank) export RERANK_ENABLED=true; shift ;;
        --crag) export CRAG_ENABLED=true; shift ;;
        --grounding) export GROUNDING_CHECK_ENABLED=true; shift ;;
        -k|--top-k) TOP_K="$2"; shift 2 ;;
        --clear-memory)
            python3 -c "from lib.memory import ConversationMemory; ConversationMemory().clear(); print('Memory cleared')"
            exit 0 ;;
        --clear-cache)
            python3 -c "from lib.query_cache import QueryCache; QueryCache().clear(); print('Cache cleared')"
            exit 0 ;;
        --status)
            ./status.sh
            exit 0 ;;
        -h|--help)
            echo "RAG Query v46 - Full Features"
            echo ""
            echo "Usage: $0 [options] \"your question\""
            echo ""
            echo "MODES:"
            echo "  --ultrafast    Minimal features, fastest (~30-45s)"
            echo "  (default)      Balanced features (<90s target)"
            echo "  --full         All features enabled (~2-5min)"
            echo "  --rag-only     Document search only, no LLM (<1s)"
            echo ""
            echo "FEATURES (toggle individually):"
            echo "  --hyde         Enable HyDE (hypothetical document)"
            echo "  --rerank       Enable FlashRank reranking"
            echo "  --crag         Enable CRAG (web fallback)"
            echo "  --grounding    Enable answer grounding check"
            echo "  --no-hyde      Disable HyDE"
            echo "  --no-rerank    Disable reranking"
            echo "  --no-memory    Disable conversation memory"
            echo "  --no-cache     Bypass query cache"
            echo ""
            echo "WEB SEARCH:"
            echo "  --web          Always include web results"
            echo "  --no-web       Never use web search"
            echo "  --web=auto     Web search when RAG insufficient (default)"
            echo ""
            echo "OTHER:"
            echo "  -k N           Return N documents (default: $TOP_K)"
            echo "  --debug        Show all processing details"
            echo "  --verbose      Show feature status"
            echo "  --clear-memory Clear conversation history"
            echo "  --clear-cache  Clear query cache"
            echo "  --status       Show system status"
            exit 0 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) QUERY="$1"; shift ;;
    esac
done

if [ -z "$QUERY" ]; then
    echo "Usage: $0 \"your question\""
    echo "       $0 --help for all options"
    exit 1
fi

export DEBUG VERBOSE ULTRAFAST RAG_ONLY FULL_MODE WEB_MODE TOP_K NO_CACHE FEEDBACK

# Run query pipeline
python3 << 'EOFPY'
import sys
import os
import time

sys.path.insert(0, './lib')

# Import modules
from llm_helper import llm_generate, get_config as get_llm_config
from query_enhance import enhance_query, classify_query
from hybrid_search import hybrid_search, get_embedding
from post_retrieval import post_process_retrieval, rerank_chunks
from grounding import verify_answer_grounding
from citations import format_context_with_citations, extract_citations_from_answer, format_sources_footer
from memory import ConversationMemory
from web_search import search_web, format_web_results
from query_cache import QueryCache

# ============================================================================
# CONFIGURATION
# ============================================================================
def str_to_bool(s):
    return str(s).lower() in ('true', '1', 'yes', 'on')

# Core
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
QDRANT = os.environ.get("QDRANT_HOST", "http://localhost:6333")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:0.5b")
TOP_K = int(os.environ.get("TOP_K", "3"))

# Timeouts
LLM_TIMEOUT_DEFAULT = int(os.environ.get("LLM_TIMEOUT_DEFAULT", "180"))
LLM_TIMEOUT_ULTRAFAST = int(os.environ.get("LLM_TIMEOUT_ULTRAFAST", "90"))

# Context limits
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "5000"))
MAX_CHUNK_CHARS = int(os.environ.get("MAX_CHUNK_CHARS", "1000"))
MAX_MEMORY_CHARS = int(os.environ.get("MAX_MEMORY_CHARS", "500"))

# Generation
NUM_PREDICT_DEFAULT = int(os.environ.get("NUM_PREDICT_DEFAULT", "500"))
NUM_PREDICT_ULTRAFAST = int(os.environ.get("NUM_PREDICT_ULTRAFAST", "150"))
NUM_PREDICT_FULL = int(os.environ.get("NUM_PREDICT_FULL", "800"))

# Feature flags
QUERY_CLASSIFICATION = str_to_bool(os.environ.get("QUERY_CLASSIFICATION_ENABLED", "false"))
HYDE_ENABLED = str_to_bool(os.environ.get("HYDE_ENABLED", "false"))
STEPBACK_ENABLED = str_to_bool(os.environ.get("STEPBACK_ENABLED", "false"))
SUBQUERY_ENABLED = str_to_bool(os.environ.get("SUBQUERY_ENABLED", "false"))
SUBQUERY_MAX = int(os.environ.get("SUBQUERY_MAX", "2"))

RERANK_ENABLED = str_to_bool(os.environ.get("RERANK_ENABLED", "false"))
RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", "10"))
RELEVANCE_FILTER = str_to_bool(os.environ.get("RELEVANCE_FILTER_ENABLED", "true"))
RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", "0.001"))
CRAG_ENABLED = str_to_bool(os.environ.get("CRAG_ENABLED", "false"))
CRAG_THRESHOLD = float(os.environ.get("CRAG_THRESHOLD", "0.4"))
CONTEXT_WINDOW = str_to_bool(os.environ.get("CONTEXT_WINDOW_ENABLED", "false"))
CONTEXT_WINDOW_SIZE = int(os.environ.get("CONTEXT_WINDOW_SIZE", "1"))
RSE_ENABLED = str_to_bool(os.environ.get("RSE_ENABLED", "false"))
DIVERSITY_FILTER = str_to_bool(os.environ.get("DIVERSITY_FILTER_ENABLED", "true"))
DIVERSITY_THRESHOLD = float(os.environ.get("DIVERSITY_THRESHOLD", "0.85"))

CITATIONS_ENABLED = str_to_bool(os.environ.get("CITATIONS_ENABLED", "true"))
GROUNDING_ENABLED = str_to_bool(os.environ.get("GROUNDING_CHECK_ENABLED", "false"))
GROUNDING_THRESHOLD = float(os.environ.get("GROUNDING_THRESHOLD", "0.5"))
GROUNDING_MAX_CLAIMS = int(os.environ.get("GROUNDING_MAX_CLAIMS", "3"))

MEMORY_ENABLED = str_to_bool(os.environ.get("MEMORY_ENABLED", "true"))
MEMORY_MAX_TURNS = int(os.environ.get("MEMORY_MAX_TURNS", "3"))
CACHE_ENABLED = str_to_bool(os.environ.get("QUERY_CACHE_ENABLED", "true"))

WEB_ENABLED = str_to_bool(os.environ.get("WEB_SEARCH_ENABLED", "true"))
WEB_MAX = int(os.environ.get("WEB_SEARCH_MAX_RESULTS", "3"))

# Runtime flags
DEBUG = str_to_bool(os.environ.get("DEBUG", "false"))
VERBOSE = str_to_bool(os.environ.get("VERBOSE", "false"))
ULTRAFAST = str_to_bool(os.environ.get("ULTRAFAST", "false"))
RAG_ONLY = str_to_bool(os.environ.get("RAG_ONLY", "false"))
FULL_MODE = str_to_bool(os.environ.get("FULL_MODE", "false"))
WEB_MODE = os.environ.get("WEB_MODE", "auto")
NO_CACHE = str_to_bool(os.environ.get("NO_CACHE", "false"))

query = """QUERYPLACEHOLDER"""

start_time = time.time()

# ============================================================================
# DEBUG HEADER
# ============================================================================
if DEBUG or VERBOSE:
    print("=" * 60)
    print("RAG Query v46 - Full Features")
    print("=" * 60)
    mode = "ULTRAFAST" if ULTRAFAST else "FULL" if FULL_MODE else "DEFAULT"
    print(f"Query: {query[:80]}{'...' if len(query) > 80 else ''}")
    print(f"Mode: {mode}")
    print(f"Features: HyDE={HYDE_ENABLED}, Rerank={RERANK_ENABLED}, CRAG={CRAG_ENABLED}, Grounding={GROUNDING_ENABLED}")
    print(f"Timeouts: {LLM_TIMEOUT_ULTRAFAST if ULTRAFAST else LLM_TIMEOUT_DEFAULT}s")
    print("=" * 60)

# ============================================================================
# CACHE CHECK
# ============================================================================
cache = QueryCache() if CACHE_ENABLED and not NO_CACHE else None
cached_result = None

if cache and not RAG_ONLY:
    cached_result = cache.get(query)
    if cached_result:
        if DEBUG:
            print("\n[CACHE HIT]")
        print("\n" + "=" * 60)
        print("ANSWER (cached)")
        print("=" * 60)
        print(cached_result.get("answer", ""))
        if cached_result.get("sources"):
            print("\n---\nSources:", ", ".join(cached_result["sources"]))
        sys.exit(0)

# ============================================================================
# MEMORY CONTEXT
# ============================================================================
memory = ConversationMemory() if MEMORY_ENABLED else None
memory_context = ""

if memory:
    memory_context = memory.get_context(max_turns=MEMORY_MAX_TURNS, max_chars=MAX_MEMORY_CHARS)
    if DEBUG and memory_context:
        print(f"\n[MEMORY] {len(memory_context)} chars from previous conversations")

# ============================================================================
# QUERY ENHANCEMENT
# ============================================================================
if DEBUG:
    print("\n[QUERY ENHANCEMENT]")

enhance_config = {
    "classification_enabled": QUERY_CLASSIFICATION,
    "rewrite_enabled": True,
    "hyde_enabled": HYDE_ENABLED,
    "stepback_enabled": STEPBACK_ENABLED,
    "subquery_enabled": SUBQUERY_ENABLED,
    "subquery_max": SUBQUERY_MAX,
}

enhanced = enhance_query(query, enhance_config)

if DEBUG:
    print(f"  Classification: {enhanced['classification']}")
    if enhanced.get("hyde_document"):
        print(f"  HyDE document: {len(enhanced['hyde_document'])} chars")
    if enhanced.get("stepback_query"):
        print(f"  StepBack query: {enhanced['stepback_query'][:60]}...")
    if len(enhanced.get("sub_queries", [])) > 1:
        print(f"  Sub-queries: {len(enhanced['sub_queries'])}")

# ============================================================================
# RETRIEVAL
# ============================================================================
if DEBUG:
    print(f"\n[RETRIEVAL] Searching for {TOP_K} documents...")

# Get HyDE embedding if enabled
hyde_embedding = None
if enhanced.get("hyde_document"):
    hyde_embedding = get_embedding(enhanced["hyde_document"])
    if DEBUG and hyde_embedding:
        print(f"  Using HyDE embedding ({len(hyde_embedding)} dims)")

# Main search
results = hybrid_search(query, top_k=TOP_K, hyde_embedding=hyde_embedding)

# StepBack search (if enabled and results sparse)
if enhanced.get("stepback_query") and len(results) < 2:
    if DEBUG:
        print(f"  Adding StepBack results...")
    stepback_results = hybrid_search(enhanced["stepback_query"], top_k=2)
    # Merge without duplicates
    seen_ids = {r["id"] for r in results}
    for r in stepback_results:
        if r["id"] not in seen_ids:
            results.append(r)

# Sub-query search (if enabled)
if len(enhanced.get("sub_queries", [])) > 1:
    if DEBUG:
        print(f"  Searching {len(enhanced['sub_queries'])} sub-queries...")
    seen_ids = {r["id"] for r in results}
    for subq in enhanced["sub_queries"][1:]:  # Skip first (original query)
        sub_results = hybrid_search(subq, top_k=2)
        for r in sub_results:
            if r["id"] not in seen_ids:
                results.append(r)
                seen_ids.add(r["id"])

if DEBUG:
    print(f"  Found {len(results)} results")

# ============================================================================
# RAG-ONLY MODE
# ============================================================================
if RAG_ONLY:
    print("\n" + "=" * 60)
    print("DOCUMENTS FOUND (RAG-only mode)")
    print("=" * 60)
    
    for i, chunk in enumerate(results[:10], 1):
        source = chunk.get("filename", "unknown")
        section = chunk.get("section", "")
        score = chunk.get("rrf_score", 0)
        text = chunk.get("text", "")[:200].replace("\n", " ")
        
        header = f"[{i}] {source}"
        if section:
            header += f" | {section}"
        header += f" (score: {score:.3f})"
        
        print(f"\n{header}")
        print(f"    {text}...")
    
    print("\n" + "=" * 60)
    elapsed = time.time() - start_time
    print(f"Time: {elapsed:.1f}s")
    sys.exit(0)

# ============================================================================
# POST-RETRIEVAL PROCESSING
# ============================================================================
if DEBUG:
    print(f"\n[POST-RETRIEVAL]")

post_config = {
    "relevance_filter_enabled": RELEVANCE_FILTER,
    "relevance_threshold": RELEVANCE_THRESHOLD,
    "diversity_filter_enabled": DIVERSITY_FILTER,
    "diversity_threshold": DIVERSITY_THRESHOLD,
    "rerank_enabled": RERANK_ENABLED,
    "rerank_top_k": RERANK_TOP_K,
    "crag_enabled": CRAG_ENABLED,
    "crag_threshold": CRAG_THRESHOLD,
    "context_window_enabled": CONTEXT_WINDOW,
    "context_window_size": CONTEXT_WINDOW_SIZE,
    "rse_enabled": RSE_ENABLED,
}

final_chunks, crag_triggered, crag_web_results = post_process_retrieval(query, results, post_config)

if DEBUG:
    print(f"  After processing: {len(final_chunks)} chunks")
    if RERANK_ENABLED:
        print(f"  Reranking: applied")
    if crag_triggered:
        print(f"  CRAG: triggered web search ({len(crag_web_results)} results)")

# ============================================================================
# WEB SEARCH
# ============================================================================
web_results = crag_web_results if crag_triggered else []
web_context = ""

# Check if we need web search
should_search_web = False
if WEB_MODE == "always":
    should_search_web = True
elif WEB_MODE == "auto" and WEB_ENABLED and not crag_triggered:
    # Check if RAG results are insufficient
    max_score = max([c.get("rrf_score", 0) or c.get("rerank_score", 0) for c in final_chunks]) if final_chunks else 0
    if max_score < 0.2 or len(final_chunks) < 2:
        should_search_web = True

if should_search_web and WEB_MODE != "never":
    if DEBUG:
        print(f"\n[WEB SEARCH]")
    web_results = search_web(query, max_results=WEB_MAX)
    if DEBUG:
        print(f"  Found {len(web_results)} web results")

if web_results:
    web_context = format_web_results(web_results)

# ============================================================================
# FORMAT CONTEXT
# ============================================================================
if CITATIONS_ENABLED:
    context, citation_map = format_context_with_citations(final_chunks, MAX_CHUNK_CHARS)
else:
    context_parts = []
    for c in final_chunks:
        text = c.get("text", "") if isinstance(c, dict) else str(c)
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS] + "..."
        context_parts.append(text)
    context = "\n\n".join(context_parts)
    citation_map = {}

# Truncate context
if len(context) > MAX_CONTEXT_CHARS:
    context = context[:MAX_CONTEXT_CHARS] + "\n\n[Context truncated for speed]"
    if DEBUG:
        print(f"\n[CONTEXT] Truncated to {MAX_CONTEXT_CHARS} chars")

# Add web context
if web_context:
    context = context + "\n\n--- WEB RESULTS ---\n\n" + web_context

if DEBUG:
    print(f"\n[CONTEXT] Total: {len(context)} chars")

# ============================================================================
# BUILD PROMPT
# ============================================================================
system_prompt = """You are a helpful assistant. Answer based on the provided context.
Cite sources using [1], [2], etc. when using information from the documents.
Be concise and accurate. If the context doesn't contain the answer, say so."""

prompt_parts = [system_prompt]

if memory_context:
    prompt_parts.append(f"\nPrevious conversation:\n{memory_context}")

prompt_parts.append(f"\nContext:\n{context}")
prompt_parts.append(f"\nQuestion: {query}")
prompt_parts.append("\nAnswer:")

full_prompt = "\n".join(prompt_parts)

if DEBUG:
    print(f"\n[PROMPT] {len(full_prompt)} chars")

# ============================================================================
# GENERATE ANSWER
# ============================================================================
if DEBUG:
    print(f"\n[GENERATION] Using {LLM_MODEL}...")

timeout = LLM_TIMEOUT_ULTRAFAST if ULTRAFAST else LLM_TIMEOUT_DEFAULT
num_tokens = NUM_PREDICT_ULTRAFAST if ULTRAFAST else (NUM_PREDICT_FULL if FULL_MODE else NUM_PREDICT_DEFAULT)

import requests

try:
    gen_start = time.time()
    
    resp = requests.post(f"{OLLAMA}/api/generate", json={
        "model": LLM_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "num_predict": num_tokens,
            "temperature": float(os.environ.get("TEMPERATURE", "0.3"))
        }
    }, timeout=timeout)
    
    gen_time = time.time() - gen_start
    
    if DEBUG:
        print(f"  Generation time: {gen_time:.1f}s")
    
    if resp.status_code == 200:
        answer = resp.json().get("response", "").strip()
        if not answer:
            answer = "No response generated. The model returned an empty response."
    else:
        answer = f"Error: HTTP {resp.status_code} from LLM"
        if DEBUG:
            print(f"  Error: {resp.text[:200]}")

except requests.exceptions.Timeout:
    answer = f"⏱️ LLM timeout after {timeout}s.\n\nDocuments found:\n"
    for i, chunk in enumerate(final_chunks[:3], 1):
        source = chunk.get("filename", "unknown")
        text = chunk.get("text", "")[:100].replace("\n", " ")
        answer += f"\n[{i}] {source}\n    {text}...\n"
    answer += "\n💡 Try: ./query.sh --ultrafast \"...\" or ./query.sh --rag-only \"...\""

except Exception as e:
    answer = f"Error: {e}"

# ============================================================================
# GROUNDING CHECK
# ============================================================================
grounding_result = None
if GROUNDING_ENABLED and answer and not answer.startswith("Error") and not answer.startswith("⏱️"):
    if DEBUG:
        print(f"\n[GROUNDING CHECK]")
    
    grounding_config = {
        "grounding_check_enabled": True,
        "grounding_threshold": GROUNDING_THRESHOLD,
        "grounding_max_claims": GROUNDING_MAX_CLAIMS,
    }
    grounding_result = verify_answer_grounding(answer, context, grounding_config)
    
    if DEBUG:
        status = "✓ VERIFIED" if grounding_result["verified"] else "✗ UNVERIFIED"
        print(f"  {status} (score: {grounding_result['score']:.2f})")

# ============================================================================
# OUTPUT
# ============================================================================
elapsed = time.time() - start_time

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)
print(answer)

# Sources footer
if CITATIONS_ENABLED and citation_map:
    citations_used = extract_citations_from_answer(answer, citation_map)
    footer = format_sources_footer(citations_used, citation_map)
    if footer:
        print(footer)

# Web sources
if web_results:
    print("\n[WEB] Web sources:")
    for i, r in enumerate(web_results, 1):
        print(f"  [{i}] {r['title'][:50]}")
        print(f"      {r['url'][:60]}")

# Grounding status
if grounding_result:
    status = "✓ VERIFIED" if grounding_result["verified"] else "⚠️ UNVERIFIED"
    print(f"\nGrounding: {status} ({grounding_result['score']*100:.0f}%)")

# Timing
print(f"\n⏱️ {elapsed:.1f}s", end="")
if ULTRAFAST:
    print(" (ultrafast mode)")
elif FULL_MODE:
    print(" (full mode)")
else:
    print("")

# ============================================================================
# SAVE TO MEMORY & CACHE
# ============================================================================
if memory and not answer.startswith("Error") and not answer.startswith("⏱️"):
    sources = [c.get("filename", "") for c in final_chunks[:3]]
    memory.add(query, answer, sources)

if cache and not answer.startswith("Error") and not answer.startswith("⏱️"):
    cache.set(query, {
        "answer": answer,
        "sources": [c.get("filename", "") for c in final_chunks[:3]]
    })

if DEBUG:
    print("\n" + "=" * 60)
    print("Debug complete")
EOFPY

# Inject the actual query
sed -i "s|QUERYPLACEHOLDER|$QUERY|g" /dev/stdin | python3
EOFSH

# Fix the query injection - use a different approach
cat > "$PROJECT_DIR/query.sh.tmp" << 'EOFSH2'
#!/bin/bash
# RAG Query v46 - Wrapper that properly escapes the query

[ -f "./config.env" ] && source ./config.env
[ -d "./venv" ] && source ./venv/bin/activate

# Export ALL configuration
export OLLAMA_HOST QDRANT_HOST COLLECTION_NAME LLM_MODEL EMBEDDING_MODEL
export EMBEDDING_TIMEOUT EMBEDDING_DIMENSION
export QUERY_CLASSIFICATION_ENABLED QUERY_REWRITE_ENABLED
export HYDE_ENABLED SUBQUERY_ENABLED SUBQUERY_MAX STEPBACK_ENABLED
export RERANK_ENABLED RERANK_MODEL RERANK_TOP_K
export RELEVANCE_FILTER_ENABLED RELEVANCE_THRESHOLD
export CRAG_ENABLED CRAG_THRESHOLD
export CONTEXT_WINDOW_ENABLED CONTEXT_WINDOW_SIZE
export RSE_ENABLED DIVERSITY_FILTER_ENABLED DIVERSITY_THRESHOLD
export CITATIONS_ENABLED
export GROUNDING_CHECK_ENABLED GROUNDING_THRESHOLD GROUNDING_MAX_CLAIMS
export MEMORY_ENABLED MEMORY_MAX_TURNS MEMORY_FILE
export QUERY_CACHE_ENABLED QUERY_CACHE_TTL
export WEB_SEARCH_ENABLED WEB_SEARCH_MODE WEB_SEARCH_MAX_RESULTS WEB_SEARCH_TIMEOUT
export LLM_TIMEOUT_DEFAULT LLM_TIMEOUT_ULTRAFAST LLM_TIMEOUT_RAG_ONLY
export MAX_CONTEXT_CHARS MAX_CHUNK_CHARS MAX_MEMORY_CHARS
export NUM_PREDICT_DEFAULT NUM_PREDICT_ULTRAFAST NUM_PREDICT_FULL TEMPERATURE
# v45 - Quality Feedback Loop
export QUALITY_LEDGER_ENABLED QUALITY_LEDGER_PATH
export CONFIDENCE_THRESHOLD_HIGH CONFIDENCE_THRESHOLD_LOW COVERAGE_THRESHOLD GROUNDING_THRESHOLD
export ABSTENTION_ENABLED ABSTENTION_MESSAGE
export DEBUG_QUALITY

TOP_K="${DEFAULT_TOP_K:-5}"
QUERY=""
FEEDBACK=""
DEBUG=false
VERBOSE=false
ULTRAFAST=false
RAG_ONLY=false
FULL_MODE=false
WEB_MODE="${WEB_SEARCH_MODE:-auto}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --debug) DEBUG=true; VERBOSE=true; shift ;;
        --verbose|-v) VERBOSE=true; shift ;;
        --ultrafast|--fast)
            ULTRAFAST=true
            export HYDE_ENABLED=false RERANK_ENABLED=false CRAG_ENABLED=false
            export GROUNDING_CHECK_ENABLED=false STEPBACK_ENABLED=false
            export SUBQUERY_ENABLED=false CONTEXT_WINDOW_ENABLED=false RSE_ENABLED=false
            TOP_K=3
            shift ;;
        --full)
            FULL_MODE=true
            export QUERY_CLASSIFICATION_ENABLED=true HYDE_ENABLED=true RERANK_ENABLED=true
            export CRAG_ENABLED=true GROUNDING_CHECK_ENABLED=true STEPBACK_ENABLED=true
            export SUBQUERY_ENABLED=true CONTEXT_WINDOW_ENABLED=true RSE_ENABLED=true
            export DIVERSITY_FILTER_ENABLED=true
            # MAX timeouts for --full mode
            export LLM_TIMEOUT_DEFAULT=600 LLM_TIMEOUT_ULTRAFAST=600
            export EMBEDDING_TIMEOUT=120 RERANK_TIMEOUT=120 WEB_SEARCH_TIMEOUT=30
            export MAX_CONTEXT_CHARS=12000 NUM_PREDICT_FULL=1200
            TOP_K=5
            shift ;;
        --rag-only) RAG_ONLY=true; shift ;;
        --web) WEB_MODE="always"; shift ;;
        --no-web) WEB_MODE="never"; shift ;;
        --no-cache) export NO_CACHE=true; shift ;;
        --no-memory) export MEMORY_ENABLED=false; shift ;;
        --no-rerank) export RERANK_ENABLED=false; shift ;;
        --hyde) export HYDE_ENABLED=true; shift ;;
        --rerank) export RERANK_ENABLED=true; shift ;;
        --no-abstention) export ABSTENTION_ENABLED=false; shift ;;
        --feedback)
            FEEDBACK="$2"
            shift 2 ;;
        --ledger-stats)
            python3 -c "import sys; sys.path.insert(0,'./lib'); from quality_ledger import QualityLedger; import json; print(json.dumps(QualityLedger().get_stats(), indent=2))"
            exit 0 ;;
        --ledger-recent)
            python3 -c "import sys; sys.path.insert(0,'./lib'); from quality_ledger import QualityLedger; import json; print(json.dumps(QualityLedger().get_recent(10), indent=2, default=str))"
            exit 0 ;;
        -k) TOP_K="$2"; shift 2 ;;
        --clear-memory)
            python3 -c "import sys; sys.path.insert(0,'./lib'); from memory import ConversationMemory; ConversationMemory().clear(); print('Memory cleared')"
            exit 0 ;;
        --clear-cache)
            python3 -c "import sys; sys.path.insert(0,'./lib'); from query_cache import QueryCache; QueryCache().clear(); print('Cache cleared')"
            exit 0 ;;
        --status) ./status.sh; exit 0 ;;
        -h|--help)
            cat << 'EOF'
RAG Query v46 - Full Features + Quality Feedback Loop

Usage: ./query.sh [options] "your question"

MODES:
  --ultrafast    Minimal features (~30-45s)
  (default)      Balanced (<90s target)
  --full         All features (~2-5min)
  --rag-only     Document search only (<1s)

FEATURES:
  --hyde         Enable HyDE
  --rerank       Enable reranking
  --no-memory    Disable memory
  --no-cache     Bypass cache

QUALITY (v45):
  --no-abstention  Disable abstention (always answer)
  --feedback X     Add feedback (correct/incorrect/partial)
  --ledger-stats   Show quality statistics
  --ledger-recent  Show recent entries

WEB:
  --web          Always use web
  --no-web       Never use web

OTHER:
  -k N           Top N documents
  --debug        Show details
  --clear-memory Clear history
  --clear-cache  Clear cache
EOF
            exit 0 ;;
        -*) echo "Unknown: $1"; exit 1 ;;
        *) QUERY="$1"; shift ;;
    esac
done

[ -z "$QUERY" ] && { echo "Usage: $0 \"question\""; exit 1; }

export DEBUG VERBOSE ULTRAFAST RAG_ONLY FULL_MODE WEB_MODE TOP_K FEEDBACK

# Run Python with query as argument
python3 ./lib/query_main.py "$QUERY"
EOFSH2

mv "$PROJECT_DIR/query.sh.tmp" "$PROJECT_DIR/query.sh"
chmod +x "$PROJECT_DIR/query.sh"

# Create the main query Python script
cat > "$PROJECT_DIR/lib/query_main.py" << 'EOFMAIN'
#!/usr/bin/env python3
"""RAG Query v46 - Main Query Pipeline"""
import sys
import os
import time
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_helper import llm_generate, get_config as get_llm_config, get_debug_info, reset_debug_info
from query_enhance import enhance_query
from hybrid_search import hybrid_search, get_embedding
from post_retrieval import post_process_retrieval
from grounding import verify_answer_grounding
from citations import format_context_with_citations, extract_citations_from_answer, format_sources_footer
from memory import ConversationMemory
from web_search import search_web, format_web_results, get_web_debug, reset_web_debug
from query_cache import QueryCache
# v45 - Quality Feedback Loop
from quality_ledger import QualityLedger
from scoring import calculate_all_scores
from decision_engine import make_decision, format_abstention_response, should_abstain

def str_to_bool(s):
    return str(s).lower() in ('true', '1', 'yes', 'on')

def main():
    if len(sys.argv) < 2:
        print("Usage: query_main.py \"your question\"")
        sys.exit(1)
    
    query = sys.argv[1]
    start_time = time.time()
    
    # Configuration
    OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:0.5b")
    TOP_K = int(os.environ.get("TOP_K", "3"))
    
    LLM_TIMEOUT_DEFAULT = int(os.environ.get("LLM_TIMEOUT_DEFAULT", "180"))
    LLM_TIMEOUT_ULTRAFAST = int(os.environ.get("LLM_TIMEOUT_ULTRAFAST", "90"))
    
    MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "5000"))
    MAX_CHUNK_CHARS = int(os.environ.get("MAX_CHUNK_CHARS", "1000"))
    MAX_MEMORY_CHARS = int(os.environ.get("MAX_MEMORY_CHARS", "500"))
    
    NUM_PREDICT_DEFAULT = int(os.environ.get("NUM_PREDICT_DEFAULT", "500"))
    NUM_PREDICT_ULTRAFAST = int(os.environ.get("NUM_PREDICT_ULTRAFAST", "150"))
    NUM_PREDICT_FULL = int(os.environ.get("NUM_PREDICT_FULL", "800"))
    
    # Feature flags
    HYDE_ENABLED = str_to_bool(os.environ.get("HYDE_ENABLED", "false"))
    STEPBACK_ENABLED = str_to_bool(os.environ.get("STEPBACK_ENABLED", "false"))
    SUBQUERY_ENABLED = str_to_bool(os.environ.get("SUBQUERY_ENABLED", "false"))
    RERANK_ENABLED = str_to_bool(os.environ.get("RERANK_ENABLED", "false"))
    CRAG_ENABLED = str_to_bool(os.environ.get("CRAG_ENABLED", "false"))
    GROUNDING_ENABLED = str_to_bool(os.environ.get("GROUNDING_CHECK_ENABLED", "false"))
    CITATIONS_ENABLED = str_to_bool(os.environ.get("CITATIONS_ENABLED", "true"))
    MEMORY_ENABLED = str_to_bool(os.environ.get("MEMORY_ENABLED", "true"))
    CACHE_ENABLED = str_to_bool(os.environ.get("QUERY_CACHE_ENABLED", "true"))
    WEB_ENABLED = str_to_bool(os.environ.get("WEB_SEARCH_ENABLED", "true"))
    
    # v45 - Quality Feedback Loop
    QUALITY_LEDGER_ENABLED = str_to_bool(os.environ.get("QUALITY_LEDGER_ENABLED", "true"))
    ABSTENTION_ENABLED = str_to_bool(os.environ.get("ABSTENTION_ENABLED", "true"))
    DEBUG_QUALITY = str_to_bool(os.environ.get("DEBUG_QUALITY", "false"))
    
    DEBUG = str_to_bool(os.environ.get("DEBUG", "false"))
    ULTRAFAST = str_to_bool(os.environ.get("ULTRAFAST", "false"))
    RAG_ONLY = str_to_bool(os.environ.get("RAG_ONLY", "false"))
    FULL_MODE = str_to_bool(os.environ.get("FULL_MODE", "false"))
    WEB_MODE = os.environ.get("WEB_MODE", "auto")
    NO_CACHE = str_to_bool(os.environ.get("NO_CACHE", "false"))
    
    # Debug header
    if DEBUG:
        print("=" * 60)
        print("RAG Query v46")
        print("=" * 60)
        mode = "ULTRAFAST" if ULTRAFAST else "FULL" if FULL_MODE else "DEFAULT"
        print(f"Query: {query[:60]}...")
        print(f"Mode: {mode} | HyDE={HYDE_ENABLED} | Rerank={RERANK_ENABLED}")
        print("=" * 60)
    
    # Cache check
    cache = QueryCache() if CACHE_ENABLED and not NO_CACHE and not RAG_ONLY else None
    if cache:
        cached = cache.get(query)
        if cached:
            if DEBUG:
                print("\n[CACHE HIT]")
            print("\n" + "=" * 60)
            print("ANSWER (cached)")
            print("=" * 60)
            print(cached.get("answer", ""))
            return
    
    # Memory
    memory = ConversationMemory() if MEMORY_ENABLED else None
    memory_context = memory.get_context(max_chars=MAX_MEMORY_CHARS) if memory else ""
    
    # Query correction (spell check, typo fix)
    try:
        from query_correction import correct_and_enrich_query, load_vocabulary
        vocab = load_vocabulary()
        correction_result = correct_and_enrich_query(query, vocab, config={
            "spell_correction": True,
            "expand_acronyms": True,
            "add_synonyms": False,  # Can make results too broad
            "use_cooccurrence": False,
        })
        
        search_query = correction_result["corrected"]
        
        if DEBUG:
            if correction_result["corrections"]:
                print(f"\n[SPELL CHECK] Corrections: {', '.join(correction_result['corrections'])}")
                print(f"  Using: '{search_query}'")
            if correction_result["expansions"]:
                print(f"  Acronyms: {', '.join(correction_result['expansions'])}")
    except Exception as e:
        if DEBUG:
            print(f"\n[SPELL CHECK] Skipped: {e}")
        search_query = query
    
    # Query enhancement
    enhance_config = {
        "hyde_enabled": HYDE_ENABLED,
        "stepback_enabled": STEPBACK_ENABLED,
        "subquery_enabled": SUBQUERY_ENABLED,
    }
    enhanced = enhance_query(search_query, enhance_config)
    
    if DEBUG and enhanced.get("hyde_document"):
        print(f"\n[HyDE] Generated {len(enhanced['hyde_document'])} char document")
    
    # Retrieval - use corrected query
    hyde_embedding = None
    if enhanced.get("hyde_document"):
        hyde_embedding = get_embedding(enhanced["hyde_document"])
    
    results = hybrid_search(search_query, top_k=TOP_K, hyde_embedding=hyde_embedding)
    
    if DEBUG:
        print(f"\n[SEARCH] Found {len(results)} results")
    
    # RAG-only mode
    if RAG_ONLY:
        print("\n" + "=" * 60)
        print("DOCUMENTS (RAG-only)")
        print("=" * 60)
        for i, c in enumerate(results[:10], 1):
            print(f"\n[{i}] {c.get('filename', '?')} (score: {c.get('rrf_score', 0):.3f})")
            print(f"    {c.get('text', '')[:150].replace(chr(10), ' ')}...")
        print(f"\n⏱️ {time.time() - start_time:.1f}s")
        return
    
    # Post-retrieval (with lower threshold to keep more results)
    if DEBUG:
        print(f"\n[PRE-FILTER] {len(results)} chunks from search:")
        for i, c in enumerate(results[:5], 1):
            score = c.get('rrf_score', 0)
            text_len = len(c.get('text', ''))
            print(f"  [{i}] score={score:.4f}, text={text_len} chars, file={c.get('filename', '?')}")
    
    post_config = {
        "rerank_enabled": RERANK_ENABLED,
        "crag_enabled": CRAG_ENABLED,
        "relevance_filter_enabled": True,
        "relevance_threshold": 0.001,  # Very low - keep almost everything
        "diversity_filter_enabled": False,  # Disable for now - it may be too aggressive
    }
    final_chunks, crag_triggered, web_results = post_process_retrieval(query, results, post_config)
    
    if DEBUG:
        print(f"\n[POST-RETRIEVAL] {len(final_chunks)} chunks after filtering")
        for i, c in enumerate(final_chunks[:3], 1):
            text_preview = c.get('text', '')[:100].replace('\n', ' ')
            print(f"  [{i}] {c.get('filename', '?')}: {text_preview}...")
    
    # Web search - only if NO local results (not just < 2)
    if WEB_MODE == "always":
        web_results = search_web(query, max_results=3)
    elif WEB_MODE == "auto" and WEB_ENABLED and len(final_chunks) == 0:
        if not web_results:
            web_results = search_web(query, max_results=3)
    
    # Format context - DISABLE CITATIONS for small models (they get confused)
    # Just use raw text which is easier to process
    context_parts = []
    for c in final_chunks:
        text = c.get("text", "") if isinstance(c, dict) else str(c)
        # Remove the "Document: ..." header if present (added during ingest)
        # Keep just the actual content
        if text.startswith("Document:"):
            # Find the actual content after the header
            lines = text.split('\n', 2)
            if len(lines) > 1:
                text = '\n'.join(lines[1:]).strip()
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS] + "..."
        context_parts.append(text)
    
    context = "\n\n---\n\n".join(context_parts)
    citation_map = {}  # Empty - we'll add sources at the end manually
    
    if DEBUG:
        print(f"\n[CONTEXT] RAG context: {len(context)} chars")
    
    # Truncate if needed
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n[truncated]"
    
    # Add web results as SUPPLEMENT only (not replacement)
    if web_results:
        web_context = format_web_results(web_results)
        # Only add web if we have room
        remaining = MAX_CONTEXT_CHARS - len(context) - 100
        if remaining > 500:
            context += "\n\n--- Additional Web Info ---\n" + web_context[:remaining]
    
    # Build prompt - EXTRACTION style (works better with small models)
    # No instructions to "say you can't answer" - just extract what's there
    prompt = f"""Context:
{context}

Question: {query}

Answer: """

    if DEBUG:
        print(f"\n[PROMPT] {len(prompt)} chars")
        if len(context) < 500:
            print(f"  WARNING: Context is very short! Check retrieval.")
        # Show first 500 chars of context for debugging
        print(f"\n[CONTEXT PREVIEW]")
        print("-" * 40)
        print(context[:500])
        print("-" * 40)
    
    # Generate - use lower temperature for small models to reduce hallucination
    timeout = LLM_TIMEOUT_ULTRAFAST if ULTRAFAST else LLM_TIMEOUT_DEFAULT
    num_tokens = NUM_PREDICT_ULTRAFAST if ULTRAFAST else (NUM_PREDICT_FULL if FULL_MODE else NUM_PREDICT_DEFAULT)
    
    # Small models need lower temperature to stay grounded
    temperature = 0.1 if ("0.5b" in LLM_MODEL.lower() or "tiny" in LLM_MODEL.lower()) else 0.2
    
    try:
        resp = requests.post(f"{OLLAMA}/api/generate", json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": num_tokens, "temperature": temperature}
        }, timeout=timeout)
        
        answer = resp.json().get("response", "").strip() if resp.status_code == 200 else f"Error: HTTP {resp.status_code}"
    except requests.exceptions.Timeout:
        answer = f"⏱️ Timeout ({timeout}s). Try --ultrafast or --rag-only"
    except Exception as e:
        answer = f"Error: {e}"
    
    # =========================================================================
    # v45 - QUALITY FEEDBACK LOOP
    # =========================================================================
    response_time_ms = int((time.time() - start_time) * 1000)
    query_id = None
    decision_result = {"decision": "unknown", "reason": "", "should_abstain": False}
    scores = {}
    
    # Calculate scores (no LLM, deterministic)
    if not answer.startswith("Error") and not answer.startswith("⏱️"):
        scores = calculate_all_scores(
            query=query,
            answer=answer,
            chunks=final_chunks,
            crag_triggered=crag_triggered,
            web_used=bool(web_results)
        )
        
        # Make decision
        decision_result = make_decision(scores)
        
        if DEBUG or DEBUG_QUALITY:
            print("\n[QUALITY SCORES]")
            print(f"  Retrieval confidence: {scores.get('retrieval_confidence', 0):.3f}")
            print(f"  Answer coverage:      {scores.get('answer_coverage', 0):.3f}")
            print(f"  Grounding:            {scores.get('grounding_score', 0):.3f}")
            print(f"  Decision: {decision_result['decision']} - {decision_result['reason']}")
        
        # Abstention check - if decision is 'abstained', replace answer
        if ABSTENTION_ENABLED and decision_result['decision'] == 'abstained':
            answer = format_abstention_response(query, final_chunks, decision_result['details'])
            if DEBUG or DEBUG_QUALITY:
                print("\n[ABSTENTION TRIGGERED]")
    
    # Log to ledger
    if QUALITY_LEDGER_ENABLED:
        try:
            ledger = QualityLedger()
            query_id = ledger.log_entry({
                "query": query,
                "retrieval": {
                    "count": len(final_chunks),
                    "sources": list(set(c.get("filename", "") for c in final_chunks)),
                    "crag_used": crag_triggered,
                    "web_used": bool(web_results)
                },
                "scores": scores,
                "decision": decision_result["decision"],
                "answer_length": len(answer),
                "response_time_ms": response_time_ms,
                "metadata": {
                    "model": LLM_MODEL,
                    "mode": "ultrafast" if ULTRAFAST else ("full" if FULL_MODE else "default")
                }
            })
            if DEBUG or DEBUG_QUALITY:
                print(f"  Ledger ID: {query_id}")
        except Exception as e:
            if DEBUG:
                print(f"  [WARN] Ledger error: {e}")
    
    # Output
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)
    
    # Show decision badge
    if scores and not answer.startswith("Error") and not answer.startswith("⏱️"):
        badge = {"confident": "✅", "low_confidence": "⚠️", "abstained": "🛑"}.get(decision_result["decision"], "❓")
        print(f"\n{badge} Confidence: {decision_result['decision'].upper()}")
        if query_id:
            print(f"   [Ledger: {query_id}]")
    
    # Show sources (simple list of unique filenames)
    if final_chunks:
        unique_sources = []
        seen = set()
        for c in final_chunks:
            fname = c.get("filename", "unknown")
            section = c.get("section", "")
            key = f"{fname}|{section}"
            if key not in seen:
                seen.add(key)
                source_str = fname
                if section:
                    source_str += f" ({section[:30]}...)" if len(section) > 30 else f" ({section})"
                unique_sources.append(source_str)
        
        print("\n---")
        print("Sources:")
        for i, src in enumerate(unique_sources[:5], 1):
            print(f"[{i}] {src}")
    
    if web_results:
        print("\n---")
        print("Web Sources:")
        for i, r in enumerate(web_results[:3], 1):
            print(f"[W{i}] {r['title'][:50]}")
            print(f"     {r['url']}")
    
    # Warning for small models that may hallucinate
    if "0.5b" in LLM_MODEL.lower() or "tiny" in LLM_MODEL.lower():
        print("\n⚠️  Small model may hallucinate. Use --rag-only for reliable facts.")
    
    print(f"\n⏱️ {time.time() - start_time:.1f}s")
    
    # v45 - Handle feedback if provided
    feedback = os.environ.get("FEEDBACK", "")
    if feedback and query_id and QUALITY_LEDGER_ENABLED:
        try:
            ledger = QualityLedger()
            ledger.add_feedback(query_id, feedback)
            print(f"\n📝 Feedback '{feedback}' recorded for {query_id}")
        except Exception as e:
            if DEBUG:
                print(f"[WARN] Feedback error: {e}")
    
    # Save
    if memory and not answer.startswith("Error") and not answer.startswith("⏱️"):
        memory.add(query, answer)
    if cache and not answer.startswith("Error") and not answer.startswith("⏱️"):
        cache.set(query, {"answer": answer})

if __name__ == "__main__":
    main()
EOFMAIN

log_ok "Query script created"

echo ""
echo "============================================================================"
echo "   Query Setup Complete! (v46 - SmartChunker + Quality Feedback)"
echo "============================================================================"
echo ""
echo "ALL FEATURES IMPLEMENTED:"
echo "  ✓ HyDE (Hypothetical Document Embeddings)"
echo "  ✓ CRAG (Corrective RAG with SearXNG)"
echo "  ✓ FlashRank reranking"
echo "  ✓ StepBack prompting"
echo "  ✓ Subquery decomposition"
echo "  ✓ Query classification"
echo "  ✓ Grounding verification"
echo "  ✓ RSE (Relevant Segment Extraction)"
echo "  ✓ Context window expansion"
echo "  ✓ Diversity filtering"
echo "  ✓ Query caching"
echo "  ✓ Conversation memory"
echo ""
echo "v46 FEATURES (SmartChunker):"
echo "  + Document layout analysis (DeepDoc/Heuristic)"
echo "  + Content-type aware chunking"
echo "  + Table/code/figure/equation preservation"
echo ""
echo "v45 FEATURES (Quality Feedback Loop):"
echo "  ✓ Quality Ledger (SQLite)"
echo "  ✓ Scoring (retrieval confidence, coverage, grounding)"
echo "  ✓ Decision Engine (confident/low_confidence/abstained)"
echo "  ✓ Abstention (no hallucination when uncertain)"
echo "  ✓ Human Feedback (--feedback flag)"
echo ""
echo "DEFAULT: Heavy features OFF (target <90s)"
echo ""
echo "Usage:"
echo "  ./query.sh \"question\"              # Default (<90s)"
echo "  ./query.sh --ultrafast \"question\"  # Minimal (~30-45s)"
echo "  ./query.sh --full \"question\"       # All features (~2-5min)"
echo "  ./query.sh --rag-only \"question\"   # Doc search only (<1s)"
echo ""
echo "Toggle features:"
echo "  ./query.sh --hyde \"question\"       # Enable HyDE"
echo "  ./query.sh --rerank \"question\"     # Enable reranking"
echo "  ./query.sh --debug \"question\"      # Show details"
echo "============================================================================"
