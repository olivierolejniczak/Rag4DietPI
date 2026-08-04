#!/bin/bash
# Opt-in BEIR retrieval benchmark (standalone, does not touch the RAG runtime).
# Evaluates this system's own retrieval stack -- FastEmbed dense + SPLADE sparse
# + Qdrant RRF fusion + FlashRank rerank -- on public BEIR datasets and reports
# nDCG@10 / Recall@100 / MAP / latency, comparable to the BEIR leaderboard.
#
# Requires Qdrant running (./status.sh). No new heavy deps: reuses fastembed,
# flashrank and qdrant-client already installed for the runtime.
#
# Examples:
#   ./run-beir.sh                              # scifact, all three methods
#   ./run-beir.sh --dataset nfcorpus
#   ./run-beir.sh --dataset fiqa --max-corpus 20000 --methods hybrid,hybrid_rerank

cd "$(dirname "$0")"
set -a; source ./config.env 2>/dev/null || true; set +a

# Same retrieval-relevant knobs the query path exports, so the benchmark hits
# the exact models/collection wiring the deployed system uses.
export QDRANT_HOST QDRANT_GRPC_PORT
export FASTEMBED_MODEL EMBEDDING_DIMENSION
export SPARSE_EMBED_ENABLED SPARSE_EMBED_MODEL
export DENSE_VECTOR_NAME SPARSE_VECTOR_NAME
export RERANK_MODEL

# -u / PYTHONUNBUFFERED: stdout is fully buffered when redirected to a file, so
# without this a mid-run kill (e.g. OOM) loses all progress output and the log
# looks empty. Unbuffered keeps progress and errors visible as they happen.
export PYTHONUNBUFFERED=1
exec python3 -u ./benchmark/beir_eval.py "$@"
