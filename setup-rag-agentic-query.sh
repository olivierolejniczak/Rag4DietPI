#!/bin/bash
# setup-rag-agentic-query.sh
# RAG System - Experimental Agentic-Loop Query Setup (see plan: vast-toasting-stearns)
#
# Sibling generator to setup-rag-query.sh. Writes lib/agentic_deep.py and
# agentic-query.sh only -- never touches setup-rag-query.sh, query.sh, or any
# other existing generated file. Installs pydantic-ai as an OPTIONAL
# dependency (mirrors the RAGAS-optional pattern in setup-rag-core.sh): if
# the install fails, agentic-query.sh still works, it just always falls back
# to the fixed pipeline (lib/agentic_deep.py detects the missing import and
# reports [agentic_deep] fallback: ... on stderr).
# Plain ASCII output
set -e

log_ok()   { echo "[OK] $1"; }
log_err()  { echo "[ERROR] $1" >&2; }
log_info() { echo "[INFO] $1"; }

PROJECT_DIR="${1:-$(pwd)}"
PROJECT_DIR="$(cd "$PROJECT_DIR" && pwd)"  # canonicalize: systemd unit files need an absolute WorkingDirectory

echo "============================================"
echo " RAG System - Agentic Query Setup (experimental)"
echo "============================================"
echo ""

mkdir -p "$PROJECT_DIR"/lib
cd "$PROJECT_DIR"

[ -f "./config.env" ] && { set -a; source ./config.env; set +a; }

# ----------------------------------------------------------------------------
# Optional dependency: pydantic-ai (winner of the Phase 0 toolkit bake-off).
# Non-fatal: failure just means run_agentic_deep() always returns None and
# agentic-query.sh falls through to the same fixed pipeline query.sh uses.
# ----------------------------------------------------------------------------
echo "Installing pydantic-ai (optional, agentic loop)..."
PIP_FLAGS="--break-system-packages --root-user-action=ignore"
pip3 install --help 2>&1 | grep -q "break-system-packages" || PIP_FLAGS=""
pip3 install --quiet pydantic-ai $PIP_FLAGS 2>/dev/null \
    && log_ok "pydantic-ai installed" \
    || log_info "pydantic-ai not installed - agentic loop will always fall back to the fixed pipeline"

log_info "Creating lib/agentic_deep.py..."
cat > "$PROJECT_DIR/lib/agentic_deep.py" << 'EOFPY'
"""Experimental ReAct-style agentic loop for the deep query tier.

Uses PydanticAI (chosen after a bake-off against smolagents ToolCallingAgent/
CodeAgent and tiny-agent-framework - see the agentic-experiment plan) to let
the model itself decide whether to call retrieve_documents / web_search or
answer directly, instead of always running the fixed classify -> decompose ->
HyDE -> retrieve -> rerank -> CRAG -> reflect sequence in query_main.py.

This module NEVER duplicates retrieval/web-search logic: both tools are thin
wrappers around the exact same primitives query_main.py already uses
(hybrid_search / hybrid_search_all_collections, web_search.search_web).

Any failure (missing dependency, backend error, usage-limit exceeded with no
final answer) is reported on stderr as `[agentic_deep] fallback: <reason>`
and run_agentic_deep() returns None so the caller can fall through to the
existing fixed pipeline (lib/query_main.py::main) unchanged.
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_helper import get_config, reset_debug_info, get_debug_info
import llm_helper as _llm_helper_module
from hybrid_search import hybrid_search, hybrid_search_all_collections
from collection_utils import collection_for_source
from web_search import search_web
from query_enhancement import classify_query


def _int_env(key, default):
    try:
        return int(os.environ.get(key, ""))
    except (TypeError, ValueError):
        return int(default)


def _gate_categories():
    """Parse AGENTIC_GATE_CATEGORIES (comma-separated, default 'factual').

    'all' disables the gate entirely (agentic loop runs for every category).
    Comparison run 2026-08-07 (tests/results/comparison-20260807-145846.md)
    showed the loop matches/beats the fixed pipeline on factual questions
    while being *faster*, but blows the >2x latency budget on comparative/
    multi-hop/procedural questions despite better answers there too - so the
    default gate routes only factual questions to the loop.
    """
    raw = os.environ.get("AGENTIC_GATE_CATEGORIES", "factual").lower().strip()
    if raw == "all":
        return None
    return {c.strip() for c in raw.split(",") if c.strip()}


SYSTEM_PROMPT = (
    "You are a research assistant answering questions about a fixed corpus "
    "of books. You do NOT have the corpus memorized and cannot see it "
    "unless you call retrieve_documents - never say you lack access to "
    "'the text' or 'the corpus' or ask the user to paste it; call "
    "retrieve_documents instead. Any question that refers to specific "
    "facts, quotes, names, chapters, or details 'in the text/corpus/book' "
    "REQUIRES at least one retrieve_documents call before you answer. If a "
    "question requires combining facts from multiple parts of the corpus, "
    "call retrieve_documents more than once with different, more specific "
    "queries. Use the web_search tool only if retrieval shows the corpus "
    "does not contain the answer and the question is about current/"
    "external information. Only skip retrieve_documents when the question "
    "is generic world knowledge with no reference to a specific text (e.g. "
    "'what is a metaphor'). Always answer in the same language as the "
    "question."
)


def run_agentic_deep(query: str, max_steps: int | None = None):
    """Run the agentic deep-tier loop for `query`.

    Returns a dict {"answer": str, "chunks": list, "path": "agentic-loop"}
    on success, or None on any failure (see module docstring) - callers must
    fall back to lib/query_main.py::main() when None is returned.
    """
    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        from pydantic_ai.usage import UsageLimits
    except ImportError as e:
        print(f"[agentic_deep] fallback: pydantic-ai not installed ({e})", file=sys.stderr)
        return None

    if max_steps is None:
        max_steps = _int_env("AGENTIC_MAX_STEPS", 4)

    reset_debug_info()

    allowed = _gate_categories()
    if allowed is not None:
        category = classify_query(query)
        if category not in allowed:
            print(
                f"[agentic_deep] fallback: category '{category}' not in "
                f"AGENTIC_GATE_CATEGORIES {sorted(allowed)}",
                file=sys.stderr,
            )
            return None

    llm_config = get_config()
    collected_chunks = []
    base_collection = os.environ.get("COLLECTION_NAME", "documents")
    source = os.environ.get("SOURCE", "")
    web_search_enabled = os.environ.get("WEB_SEARCH_ENABLED", "true").lower() == "true"

    try:
        model = OpenAIChatModel(
            llm_config["llm_model"],
            provider=OpenAIProvider(base_url=llm_config["api_base"], api_key="not-needed"),
        )
        agent = Agent(model, system_prompt=SYSTEM_PROMPT)

        @agent.tool_plain
        def retrieve_documents(query: str, top_k: int = 5) -> str:
            """Search the document corpus for chunks relevant to `query`."""
            if source:
                chunks = hybrid_search(query, top_k=top_k, collection=collection_for_source(source, base_collection))
            else:
                chunks = hybrid_search_all_collections(query, top_k=top_k, base_collection=base_collection)
            collected_chunks.extend(chunks)
            if not chunks:
                return "No relevant documents found."
            parts = []
            for c in chunks:
                filename = c.get("filename", "unknown")
                text = (c.get("text") or "")[:800]
                parts.append(f"[{filename}] {text}")
            return "\n\n".join(parts)

        if web_search_enabled:
            @agent.tool_plain
            def web_search(query: str, max_results: int = 5) -> str:
                """Search the web when the corpus does not contain the answer."""
                results = search_web(query, max_results=max_results)
                for r in results:
                    collected_chunks.append({
                        "filename": r.get("source", r.get("url", "web")),
                        "text": r.get("content", ""),
                        "chunk_type": "web",
                        "source": r.get("url", ""),
                    })
                if not results:
                    return "No web results found."
                return "\n\n".join(f"[{r.get('title', 'web')}] {r.get('content', '')[:500]}" for r in results)

        t0 = time.time()
        result = agent.run_sync(query, usage_limits=UsageLimits(request_limit=max_steps))
        elapsed = time.time() - t0

        usage = result.usage
        _llm_helper_module._debug_info["llm_model"] = llm_config["llm_model"]
        _llm_helper_module._debug_info["llm_calls"] += getattr(usage, "requests", 1)
        _llm_helper_module._debug_info["llm_total_time"] += elapsed

        answer = str(result.output).strip()
        if not answer:
            print("[agentic_deep] fallback: empty answer from agent", file=sys.stderr)
            return None

        print(f"[agentic_deep] path: agentic-loop ({usage})", file=sys.stderr)
        return {"answer": answer, "chunks": collected_chunks, "path": "agentic-loop"}

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(f"[agentic_deep] fallback: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _print_result(query, result, elapsed):
    debug_info = get_debug_info()
    print("============================================")
    print(f"Query: {query}")
    print(f"Path: {result['path']}")
    print("")
    print("=== Sources ===")
    for i, c in enumerate(result["chunks"], 1):
        filename = c.get("filename", "unknown")
        if c.get("chunk_type") == "web":
            print(f"  [{i}] WEB: {c.get('source', filename)}")
        else:
            print(f"  [{i}] {filename}")
    print("\n=== Debug Info ===")
    print(f"LLM: {debug_info['llm_model']} ({debug_info['llm_calls']} calls, {debug_info['llm_total_time']:.1f}s)")
    print(f"Total time: {elapsed:.1f}s")
    print("\n=== Answer ===")
    print(result["answer"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agentic_deep.py 'your question'")
        sys.exit(1)
    query = sys.argv[1]
    t0 = time.time()
    result = run_agentic_deep(query)
    elapsed = time.time() - t0
    if result is None:
        print("[agentic_deep] fallback: falling through to fixed pipeline (query_main.main)", file=sys.stderr)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from query_main import main as query_main_main
        query_main_main(query)
    else:
        _print_result(query, result, elapsed)
EOFPY
chmod 644 "$PROJECT_DIR/lib/agentic_deep.py"
log_ok "lib/agentic_deep.py"

log_info "Creating agentic-query.sh..."
cat > "$PROJECT_DIR/agentic-query.sh" << 'EOFSH'
#!/bin/bash
# Experimental agentic-loop variant of query.sh (see plan: vast-toasting-stearns).
# Flag parsing/env-export/dispatch is a verbatim clone of query.sh so that
# --rag-only/--web-only/--ultrafast/default all behave identically. Only
# --full (deep tier) diverges: when AGENTIC_DEEP_ENABLED=true, it first tries
# lib/agentic_deep.py's ReAct-style loop and falls back to the same fixed
# query_main.py pipeline query.sh uses when the loop fails or is disabled.
# Modes: default, --rag-only, --web-only, --ultrafast, --full

cd "$(dirname "$0")"
set -a; source ./config.env 2>/dev/null || true; set +a

export OLLAMA_HOST LLM_MODEL TEMPERATURE QDRANT_HOST QDRANT_GRPC_PORT COLLECTION_NAME
export ANSWER_LANG
export SPARSE_EMBED_ENABLED HYBRID_SEARCH_MODE FASTEMBED_MODEL
export SEARXNG_URL WEB_SEARCH_ENABLED
export CRAG_ENABLED CRAG_THRESHOLD
export RERANK_ENABLED RELEVANCE_THRESHOLD
export SPELLCHECK_ENABLED QUERY_NORMALIZE_ENABLED SPELLCHECK_WHITELIST_FILE DEBUG VERBOSE
export ADAPTIVE_ENABLED MAX_TIER ESCALATE_CONFIDENCE_MIN
export SOURCE
export AGENTIC_DEEP_ENABLED="${AGENTIC_DEEP_ENABLED:-false}"
export AGENTIC_MAX_STEPS="${AGENTIC_MAX_STEPS:-4}"
export AGENTIC_GATE_CATEGORIES="${AGENTIC_GATE_CATEGORIES:-factual}"

# Default settings
MODE="default"
DEBUG_FLAG=""
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --rag-only)
            MODE="rag-only"
            export HYDE_ENABLED=false CRAG_ENABLED=false RERANK_ENABLED=false
            export MEMORY_ENABLED=false QUERY_CACHE_ENABLED=true
            export ADAPTIVE_ENABLED=false
            shift ;;
        --web-only)
            MODE="web-only"
            export ADAPTIVE_ENABLED=false
            shift ;;
        --ultrafast)
            MODE="ultrafast"
            export HYDE_ENABLED=false CRAG_ENABLED=false RERANK_ENABLED=false
            export MULTIPASS_ENABLED=false STEPBACK_ENABLED=false
            export ADAPTIVE_ENABLED=false
            export NUM_PREDICT=${NUM_PREDICT_ULTRAFAST:-400}
            export LLM_TIMEOUT=${LLM_TIMEOUT_ULTRAFAST:-90}
            shift ;;
        --full)
            MODE="full"
            export HYDE_ENABLED=true CRAG_ENABLED=true RERANK_ENABLED=true
            export MULTIPASS_ENABLED=true CITATIONS_ENABLED=true
            export HYPOTHETICAL_TITLE_ENABLED=true QUERY_REWRITE_ENABLED=true
            export ADAPTIVE_ENABLED=false
            export NUM_PREDICT=${NUM_PREDICT_FULL:-1200}
            export LLM_TIMEOUT=${LLM_TIMEOUT_FULL:-0}
            shift ;;
        --debug) DEBUG_FLAG="--debug"; export DEBUG=true VERBOSE=true; shift ;;
        --multipass) export MULTIPASS_ENABLED=true; shift ;;
        --citations) export CITATIONS_ENABLED=true; shift ;;
        --no-adaptive) export ADAPTIVE_ENABLED=false; shift ;;
        --max-tier)
            shift
            if ! [[ "${1:-}" =~ ^[0-3]$ ]]; then
                echo "Error: --max-tier requires a value 0-3 (got '${1:-}')" >&2
                exit 2
            fi
            export MAX_TIER="$1"
            shift ;;
        --source)
            shift
            if [ -z "${1:-}" ]; then
                echo "Error: --source requires a folder name (e.g. --source MAISONS)" >&2
                exit 2
            fi
            export SOURCE="$1"
            shift ;;
        --list-sources)
            python3 -c "
import sys; sys.path.insert(0, './lib')
from collection_utils import list_source_names
from qdrant_client_helper import get_client
import os
client, _mode = get_client()
base = os.environ.get('COLLECTION_NAME', 'documents')
sources = list_source_names(client, base) if client else []
if sources:
    print('Available --source values:')
    for s in sources:
        print(f'  {s}')
else:
    print('No per-folder collections found yet. Ingest documents first with: ./ingest.sh')
"
            exit 0 ;;
        --list-folders)
            LIST_FOLDERS=1; shift
            if [[ "${1:-}" =~ ^[0-9]+$ ]]; then FOLDERS_DEPTH="$1"; shift; fi ;;
        --no-memory) export MEMORY_ENABLED=false; shift ;;
        --no-cache) export QUERY_CACHE_ENABLED=false; shift ;;
        --clear-cache)
            python3 -c "import sys; sys.path.insert(0,'./lib'); from query_cache import QueryCache; QueryCache().clear(); print('Cache cleared.')"
            exit 0 ;;
        --whitelist-add)
            shift
            python3 -c "import sys; sys.path.insert(0,'./lib'); from spellcheck import add_to_whitelist; r=add_to_whitelist('$1'); print(f'Added: $1' if r else f'Already exists: $1')"
            exit 0 ;;
        --whitelist-show)
            echo "=== Spellcheck Whitelist ==="
            echo "Default terms: $(python3 -c "import sys; sys.path.insert(0,'./lib'); from spellcheck import DEFAULT_WHITELIST; print(len(DEFAULT_WHITELIST))")"
            if [ -f "./cache/spellcheck_whitelist.txt" ]; then
                echo "Custom terms:"
                cat ./cache/spellcheck_whitelist.txt | grep -v '^#' | sort
            else
                echo "No custom whitelist file yet."
            fi
            exit 0 ;;
        --whitelist-auto)
            echo "Auto-populating whitelist from indexed documents..."
            python3 -c "import sys; sys.path.insert(0,'./lib'); from spellcheck import populate_whitelist_from_collection; n=populate_whitelist_from_collection(); print(f'Added {n} new terms')"
            exit 0 ;;
        --agentic) export AGENTIC_DEEP_ENABLED=true; shift ;;
        --no-agentic) export AGENTIC_DEEP_ENABLED=false; shift ;;
        --max-steps)
            shift
            if ! [[ "${1:-}" =~ ^[0-9]+$ ]]; then
                echo "Error: --max-steps requires a positive integer (got '${1:-}')" >&2
                exit 2
            fi
            export AGENTIC_MAX_STEPS="$1"
            shift ;;
        --gate-categories)
            shift
            if [ -z "${1:-}" ]; then
                echo "Error: --gate-categories requires a value (e.g. 'factual' or 'all')" >&2
                exit 2
            fi
            export AGENTIC_GATE_CATEGORIES="$1"
            shift ;;
        --help|-h)
            echo "Usage: ./agentic-query.sh [options] 'question'"
            echo ""
            echo "Experimental clone of query.sh. Modes: (default), --rag-only,"
            echo "--web-only, --ultrafast, --full."
            echo ""
            echo "Agentic-loop options (deep/--full tier only):"
            echo "  --agentic      Force-enable the ReAct-style agentic loop"
            echo "  --no-agentic   Force-disable it (use the fixed pipeline)"
            echo "  --max-steps N  Max tool-call steps before the loop must answer (default 4)"
            echo "  --gate-categories LIST"
            echo "                 Comma-separated categories routed to the loop (default 'factual');"
            echo "                 other categories fall back to the fixed pipeline. 'all' disables the gate."
            echo ""
            echo "All other options match query.sh --help."
            exit 0 ;;
        --*)
            echo "Error: unknown option '$1'" >&2
            echo "Try: ./agentic-query.sh --help" >&2
            exit 2 ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift ;;
    esac
done

QUERY=$(echo "$EXTRA_ARGS" | sed 's/^ *//')

# --list-folders: deferred so --source is captured wherever it appears.
if [ -n "${LIST_FOLDERS:-}" ]; then
    python3 -c "
import sys, os; sys.path.insert(0, './lib')
from collection_utils import collection_for_source, list_ingest_collections, list_folders
from qdrant_client_helper import get_client
base = os.environ.get('COLLECTION_NAME', 'documents')
docs = os.environ.get('DOCUMENTS_DIR', '/root/documents')
depth = ${FOLDERS_DEPTH:-0} or None
src = os.environ.get('SOURCE', '')
if src:
    cols = [collection_for_source(src, base)]
else:
    client, _m = get_client()
    cols = list_ingest_collections(client, base) if client else [base]
found = False
for col in cols:
    rows = [(d, n) for d, n in list_folders(col, docs, depth=depth) if d]
    if not rows:
        continue
    found = True
    print(f'# {col}')
    for d, n in rows:
        print(f'  {n:4d}  {d}')
if not found:
    print('No folders found. Check --source or ingest documents first with: ./ingest.sh')
"
    exit 0
fi

case $MODE in
    ultrafast) export QUERY_MODE_ACTIVE=quick ;;
    full)      export QUERY_MODE_ACTIVE=deep ;;
    *)         export QUERY_MODE_ACTIVE=default
               export ADAPTIVE_ENABLED=${ADAPTIVE_ENABLED:-true} ;;
esac

if [ -z "$QUERY" ]; then
    echo "Usage: ./agentic-query.sh [options] 'question'"
    echo "Try: ./agentic-query.sh --help"
    exit 1
fi

echo "============================================"
echo " RAG Query (agentic experiment) [$MODE]"
echo "============================================"

case $MODE in
    web-only)
        echo "Mode: Web-Only (bypass RAG)"
        echo ""
        python3 ./lib/web_only_query.py "$QUERY"
        ;;
    rag-only)
        echo "Mode: RAG-Only (no LLM)"
        echo ""
        RAG_ONLY=true python3 ./lib/query_main.py "$QUERY"
        ;;
    full)
        if [ "$AGENTIC_DEEP_ENABLED" = "true" ]; then
            echo "Mode: $MODE (agentic loop, max $AGENTIC_MAX_STEPS steps; falls back to fixed pipeline on failure)"
            echo ""
            python3 ./lib/agentic_deep.py "$QUERY"
        else
            echo "Mode: $MODE (fixed pipeline; AGENTIC_DEEP_ENABLED=false)"
            echo ""
            python3 ./lib/query_entry.py "$QUERY"
        fi
        ;;
    *)
        echo "Mode: $MODE"
        echo ""
        python3 ./lib/query_entry.py "$QUERY"
        ;;
esac
EOFSH
chmod +x "$PROJECT_DIR/agentic-query.sh"
log_ok "agentic-query.sh"

# ============================================================================
# Verification
# ============================================================================
echo ""
echo "=== Verification ==="
[ -f "$PROJECT_DIR/lib/agentic_deep.py" ] && log_ok "lib/agentic_deep.py" || log_err "lib/agentic_deep.py"
[ -x "$PROJECT_DIR/agentic-query.sh" ] && log_ok "agentic-query.sh (executable)" || log_err "agentic-query.sh"
python3 -c "import ast; ast.parse(open('$PROJECT_DIR/lib/agentic_deep.py').read())" 2>/dev/null \
    && log_ok "lib/agentic_deep.py (valid Python syntax)" || log_err "lib/agentic_deep.py has a syntax error"
python3 -c "import pydantic_ai" 2>/dev/null && log_ok "pydantic-ai importable" || log_info "pydantic-ai not importable (agentic loop will fall back to fixed pipeline)"

echo ""
echo "============================================"
echo " Agentic Query Setup Complete (experimental)"
echo "============================================"
echo ""
echo "This is an experimental, parallel query path. It does not change the"
echo "behavior of query.sh in any way."
echo ""
echo "Usage:"
echo "  AGENTIC_DEEP_ENABLED=true ./agentic-query.sh --full 'question'"
echo "  ./agentic-query.sh --full --agentic 'question'    # force-enable per-call"
echo "  ./agentic-query.sh --full --no-agentic 'question' # force fixed pipeline"
echo ""
echo "Config (config.env):"
echo "  AGENTIC_DEEP_ENABLED=false   # default: off, --full uses the fixed pipeline"
echo "  AGENTIC_MAX_STEPS=4          # max tool-call steps before the loop must answer"
echo "  AGENTIC_GATE_CATEGORIES=factual  # categories routed to the loop; 'all' disables the gate"
