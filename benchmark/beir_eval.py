#!/usr/bin/env python3
"""Opt-in BEIR retrieval benchmark for the RAG system's retrieval stack.

Evaluates THIS system's retrieval components — FastEmbed dense embeddings,
SPLADE sparse embeddings, Qdrant native RRF fusion, and the FlashRank
cross-encoder reranker — on standard public BEIR datasets, so the numbers are
directly comparable to the BEIR leaderboard.

It is deliberately dependency-light: it downloads the public BEIR zip and
computes nDCG/Recall itself, reusing only lib helpers already installed for the
RAG runtime (fastembed, flashrank, qdrant-client). No `beir`/torch needed.

Design notes / honest caveats (report these alongside results):
  * Queries are embedded exactly as the deployed pipeline embeds them (no
    special bge query-instruction prefix), so this measures the SHIPPED
    behaviour, not the theoretical best of bge-base.
  * The reranker (ms-marco-MiniLM) is trained on MS MARCO, so BEIR is mildly
    in-domain for it — expected to help most on MS-MARCO-like datasets.
  * hybrid+rerank only re-orders the hybrid candidate pool, so Recall@100 for
    it equals hybrid; the rerank effect shows in nDCG@10.

Usage:
  python3 beir_eval.py --dataset scifact
  python3 beir_eval.py --dataset fiqa --max-corpus 20000 --methods hybrid,hybrid_rerank
"""
import argparse
import csv
import io
import json
import math
import os
import sys
import time
import zipfile

# --- reuse the RAG system's own lib helpers ---------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "lib"))

from embedding_helper import get_embeddings_batch, get_embedding  # noqa: E402
from sparse_embedding_helper import (  # noqa: E402
    get_sparse_embeddings_batch,
    get_sparse_embedding,
    is_sparse_embed_available,
)
from qdrant_client_helper import get_client  # noqa: E402
from qdrant_hybrid_helper import (  # noqa: E402
    delete_collection,
    ensure_hybrid_collection,
    upload_hybrid_points,
)
from post_retrieval import rerank_chunks  # noqa: E402

import requests  # noqa: E402

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DENSE_NAME = os.environ.get("DENSE_VECTOR_NAME", "dense")
SPARSE_NAME = os.environ.get("SPARSE_VECTOR_NAME", "sparse")


# ----------------------------------------------------------------------------
# Dataset loading (public BEIR zip -> corpus/queries/qrels)
# ----------------------------------------------------------------------------
def download_dataset(name, datasets_dir):
    """Download + unzip a BEIR dataset if not already present. Returns its dir."""
    ds_dir = os.path.join(datasets_dir, name)
    if os.path.isdir(ds_dir) and os.path.exists(os.path.join(ds_dir, "corpus.jsonl")):
        return ds_dir
    os.makedirs(datasets_dir, exist_ok=True)
    url = BEIR_URL.format(name=name)
    print(f"[BEIR] downloading {url}")
    resp = requests.get(url, timeout=300, stream=True)
    resp.raise_for_status()
    buf = io.BytesIO(resp.content)
    with zipfile.ZipFile(buf) as zf:
        zf.extractall(datasets_dir)
    if not os.path.exists(os.path.join(ds_dir, "corpus.jsonl")):
        raise RuntimeError(f"unexpected archive layout for {name}")
    return ds_dir


def load_corpus(ds_dir):
    """Return {doc_id: text} — title and body joined as BEIR convention."""
    corpus = {}
    with open(os.path.join(ds_dir, "corpus.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            title = (d.get("title") or "").strip()
            text = (d.get("text") or "").strip()
            corpus[d["_id"]] = (title + " " + text).strip() if title else text
    return corpus


def load_queries(ds_dir):
    q = {}
    with open(os.path.join(ds_dir, "queries.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            q[d["_id"]] = d["text"]
    return q


def load_qrels(ds_dir, split):
    """Return {query_id: {doc_id: relevance}} from qrels/<split>.tsv."""
    qrels = {}
    path = os.path.join(ds_dir, "qrels", f"{split}.tsv")
    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)  # header: query-id  corpus-id  score
        for row in reader:
            if len(row) < 3:
                continue
            qid, did, score = row[0], row[1], int(row[2])
            qrels.setdefault(qid, {})[did] = score
    return qrels


# ----------------------------------------------------------------------------
# Indexing
# ----------------------------------------------------------------------------
def index_corpus(collection, corpus, batch_size=64):
    """Embed (dense + sparse) and upsert the corpus into a fresh collection.

    Returns (id_to_doc mapping, elapsed_seconds, n_indexed)."""
    delete_collection(collection)
    if not ensure_hybrid_collection(collection):
        raise RuntimeError(f"could not create collection {collection}")

    doc_ids = list(corpus.keys())
    id_to_doc = {}
    start = time.time()
    n = 0
    for i in range(0, len(doc_ids), batch_size):
        batch_ids = doc_ids[i:i + batch_size]
        texts = [corpus[d] for d in batch_ids]
        dense = get_embeddings_batch(texts, batch_size=batch_size)
        sparse = get_sparse_embeddings_batch(texts, batch_size=batch_size)
        points = []
        for j, did in enumerate(batch_ids):
            pid = i + j
            id_to_doc[pid] = did
            points.append({
                "id": pid,
                "dense_vector": dense[j],
                "sparse_vector": sparse[j],  # may be None -> dense-only for that doc
                "payload": {"doc_id": did},
            })
        if not upload_hybrid_points(collection, points):
            raise RuntimeError("point upload failed")
        n += len(points)
        print(f"\r[INDEX] {n}/{len(doc_ids)} docs", end="", flush=True)
    print()
    return id_to_doc, time.time() - start, n


# ----------------------------------------------------------------------------
# Retrieval (per method), returning {doc_id: score} for one query
# ----------------------------------------------------------------------------
def _dense_search(client, collection, qvec, limit):
    from qdrant_client import models
    res = client.query_points(
        collection_name=collection, query=qvec, using=DENSE_NAME,
        limit=limit, with_payload=["doc_id"],
    ).points
    return [(p.payload["doc_id"], p.score) for p in res]


def _hybrid_search(client, collection, qvec, qsparse, limit, prefetch):
    from qdrant_client import models
    pre = [models.Prefetch(query=qvec, using=DENSE_NAME, limit=prefetch)]
    if qsparse:
        pre.append(models.Prefetch(
            query=models.SparseVector(indices=qsparse["indices"], values=qsparse["values"]),
            using=SPARSE_NAME, limit=prefetch,
        ))
    res = client.query_points(
        collection_name=collection, prefetch=pre,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit, with_payload=["doc_id"],
    ).points
    return [(p.payload["doc_id"], p.score) for p in res]


def _rerank(query, ranked, corpus, top_k):
    """Rerank a (doc_id, score) candidate list with the FlashRank cross-encoder."""
    chunks = [{"doc_id": did, "text": corpus[did]} for did, _ in ranked]
    reranked = rerank_chunks(query, chunks, top_k=len(chunks))
    out = []
    for i, ch in enumerate(reranked):
        out.append((ch["doc_id"], ch.get("rerank_score", -i)))
    return out[:top_k] if top_k else out


# ----------------------------------------------------------------------------
# Metrics (standard TREC-style nDCG@k, Recall@k, MAP)
# ----------------------------------------------------------------------------
def dcg(rels):
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_ids, rel, k):
    gains = [rel.get(d, 0) for d in ranked_ids[:k]]
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids, rel, k):
    relevant = {d for d, r in rel.items() if r > 0}
    if not relevant:
        return 0.0
    hit = sum(1 for d in ranked_ids[:k] if d in relevant)
    return hit / len(relevant)


def average_precision(ranked_ids, rel):
    relevant = {d for d, r in rel.items() if r > 0}
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for i, d in enumerate(ranked_ids):
        if d in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / len(relevant)


def aggregate(per_query):
    """Mean each metric across queries."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(q[k] for q in per_query) / len(per_query) for k in keys}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="BEIR retrieval benchmark for the RAG stack")
    ap.add_argument("--dataset", default="scifact",
                    help="BEIR dataset name (scifact, fiqa, nfcorpus, arguana, trec-covid, ...)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--methods", default="dense,hybrid,hybrid_rerank",
                    help="comma list of: dense, hybrid, hybrid_rerank")
    ap.add_argument("--top-k", type=int, default=100, help="retrieval depth (Recall@k)")
    ap.add_argument("--ndcg-k", type=int, default=10)
    ap.add_argument("--rerank-candidates", type=int, default=100,
                    help="hybrid candidates fed to the reranker")
    ap.add_argument("--prefetch", type=int, default=100,
                    help="per-branch prefetch depth before RRF fusion")
    ap.add_argument("--corpus-batch", type=int, default=64,
                    help="corpus embedding batch size (lower = less peak RAM)")
    ap.add_argument("--max-corpus", type=int, default=0, help="cap corpus size (0 = all)")
    ap.add_argument("--max-queries", type=int, default=0, help="cap #queries (0 = all)")
    ap.add_argument("--datasets-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets"))
    ap.add_argument("--keep-collection", action="store_true", help="don't drop the temp collection at the end")
    ap.add_argument("--json-out", default="", help="optional path to write results JSON")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    collection = f"beir_{args.dataset}".replace("-", "_")

    ds_dir = download_dataset(args.dataset, args.datasets_dir)
    corpus = load_corpus(ds_dir)
    queries = load_queries(ds_dir)
    qrels = load_qrels(ds_dir, args.split)
    # keep only queries that have judgments in this split
    queries = {q: t for q, t in queries.items() if q in qrels}
    if args.max_queries:
        queries = dict(list(queries.items())[:args.max_queries])

    # If capping the corpus, always keep judged docs so metrics stay meaningful
    if args.max_corpus and args.max_corpus < len(corpus):
        judged = {d for q in queries for d in qrels.get(q, {})}
        keep = list(judged) + [d for d in corpus if d not in judged]
        keep = set(keep[:max(args.max_corpus, len(judged))])
        corpus = {d: t for d, t in corpus.items() if d in keep}

    print(f"[BEIR] {args.dataset}/{args.split}: {len(corpus)} docs, {len(queries)} queries, "
          f"sparse={'on' if is_sparse_embed_available() else 'OFF'}")

    client, mode = get_client()
    if client is None:
        print("[ERROR] Qdrant unreachable. Start services with ./status.sh", file=sys.stderr)
        return 1

    _id_to_doc, index_secs, n_indexed = index_corpus(collection, corpus, batch_size=args.corpus_batch)
    print(f"[INDEX] {n_indexed} docs in {index_secs:.1f}s "
          f"({n_indexed / index_secs:.0f} docs/s)")

    # Per-method accumulators
    results = {m: [] for m in methods}
    latency = {m: 0.0 for m in methods}

    qids = list(queries.keys())
    for n, qid in enumerate(qids, 1):
        query = queries[qid]
        rel = qrels[qid]
        qvec = get_embedding(query)
        qsparse = get_sparse_embedding(query) if is_sparse_embed_available() else None

        for m in methods:
            t0 = time.time()
            if m == "dense":
                ranked = _dense_search(client, collection, qvec, args.top_k)
            elif m == "hybrid":
                ranked = _hybrid_search(client, collection, qvec, qsparse, args.top_k, args.prefetch)
            elif m == "hybrid_rerank":
                cand = _hybrid_search(client, collection, qvec, qsparse,
                                      max(args.rerank_candidates, args.top_k), args.prefetch)
                reranked = _rerank(query, cand[:args.rerank_candidates], corpus, top_k=0)
                # keep reranked head, then append the untouched tail so Recall@k is fair
                seen = {d for d, _ in reranked}
                ranked = reranked + [(d, s) for d, s in cand if d not in seen]
            else:
                continue
            latency[m] += time.time() - t0
            ids = [d for d, _ in ranked]
            results[m].append({
                f"nDCG@{args.ndcg_k}": ndcg_at_k(ids, rel, args.ndcg_k),
                f"Recall@{args.top_k}": recall_at_k(ids, rel, args.top_k),
                "MAP": average_precision(ids, rel),
            })
        print(f"\r[EVAL] {n}/{len(qids)} queries", end="", flush=True)
    print()

    if not args.keep_collection:
        delete_collection(collection)

    # ---- report ----
    summary = {}
    print("\n" + "=" * 72)
    print(f" BEIR: {args.dataset}/{args.split}  |  bge-base-en-v1.5 + SPLADE + RRF"
          f"{' + rerank' if 'hybrid_rerank' in methods else ''}")
    print(f" corpus={len(corpus)}  queries={len(qids)}  index={index_secs:.0f}s"
          f"  qdrant_mode={mode}")
    print("=" * 72)
    header = f"{'method':<16}{'nDCG@'+str(args.ndcg_k):>12}{'Recall@'+str(args.top_k):>14}{'MAP':>10}{'ms/query':>12}"
    print(header)
    print("-" * len(header))
    for m in methods:
        agg = aggregate(results[m])
        ms = latency[m] / max(len(qids), 1) * 1000
        summary[m] = {**agg, "ms_per_query": ms}
        print(f"{m:<16}{agg[f'nDCG@{args.ndcg_k}']:>12.4f}"
              f"{agg[f'Recall@{args.top_k}']:>14.4f}{agg['MAP']:>10.4f}{ms:>12.1f}")
    print("=" * 72)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({
                "dataset": args.dataset, "split": args.split,
                "corpus_size": len(corpus), "num_queries": len(qids),
                "index_seconds": index_secs, "qdrant_mode": mode,
                "embedding_model": os.environ.get("FASTEMBED_MODEL", "BAAI/bge-base-en-v1.5"),
                "sparse_model": os.environ.get("SPARSE_EMBED_MODEL", "prithivida/Splade_PP_en_v1"),
                "rerank_model": os.environ.get("RERANK_MODEL", "ms-marco-MiniLM-L-12-v2"),
                "results": summary,
            }, fh, indent=2)
        print(f"[BEIR] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
