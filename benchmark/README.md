# BEIR retrieval benchmark (opt-in)

Standalone benchmark for **this system's retrieval stack**. It is completely
separate from the RAG runtime and the setup generators — nothing here is
imported by `ingest`/`query`, and running it does not modify any runtime
collection.

It measures the components that decide what the LLM sees:

- **dense** — FastEmbed `bge-base-en-v1.5` only
- **hybrid** — dense + SPLADE sparse fused with Qdrant native RRF
- **hybrid_rerank** — hybrid candidates re-ordered by the FlashRank
  cross-encoder (`ms-marco-MiniLM-L-12-v2`)

on standard public [BEIR](https://github.com/beir-cellar/beir) datasets, so the
`nDCG@10` / `Recall@100` numbers are directly comparable to the BEIR
leaderboard.

## Why it exists

Retrieval quality is otherwise only observable anecdotally (a query looks good
or bad). This gives a repeatable, judge-free score — no LLM in the loop — to
answer "did that retrieval change actually help?" before/after tuning fusion,
swapping an embedding model, or toggling the reranker.

## Requirements

No new heavy dependencies. It reuses what the runtime already installs
(`fastembed`, `flashrank`, `qdrant-client`) plus `requests`. In particular it
does **not** pull in the `beir`/`torch` stack — BEIR datasets are downloaded as
the public zip and the metrics (nDCG/Recall/MAP) are computed locally.

Qdrant must be running (`./status.sh`). A temporary collection `beir_<dataset>`
is created and dropped automatically (`--keep-collection` to retain it).

## Usage

```bash
./run-beir.sh                                   # scifact, all three methods
./run-beir.sh --dataset nfcorpus                # small, ~3.6k docs — fast
./run-beir.sh --dataset fiqa --max-corpus 20000 --methods hybrid,hybrid_rerank
./run-beir.sh --dataset scifact --json-out benchmark/scifact.json
```

Good CPU-friendly starters (small corpora): `nfcorpus` (~3.6k docs),
`scifact` (~5.2k), `arguana` (~8.7k). `fiqa` (~57k) and `trec-covid` (~171k)
are large — cap with `--max-corpus` for a smoke test (judged docs are always
kept so metrics stay valid).

### Throughput on a CPU box (plan accordingly)

Indexing is dominated by **SPLADE sparse embedding** (dense bge-base is cheap by
comparison). On this 6-core box the corpus indexes at roughly **0.5–1 doc/s**,
so even "small" datasets are not interactive:

| corpus | approx. index time (this box) |
|--------|-------------------------------|
| `--max-corpus 1000` | ~15–30 min |
| `nfcorpus` (3.6k)   | ~1–2 h |
| `scifact` (5.2k)    | ~1.5–3 h |
| `fiqa` (57k)        | overnight — always cap or run detached |

Practical recipe:

- **Interactive proof / iterating on config** → cap it:
  `./run-beir.sh --dataset nfcorpus --max-corpus 1000 --max-queries 100`
  (validates the whole pipeline in minutes; Recall/nDCG are inflated by the
  smaller distractor pool, so treat these as a smoke, not a real score).
- **Real, quotable numbers** → run the full corpus **detached** so an SSH drop
  doesn't kill it:
  `tmux new -s beir` then `./run-beir.sh --dataset nfcorpus --json-out benchmark/nfcorpus.json`,
  or `nohup ./run-beir.sh --dataset nfcorpus > beir.log 2>&1 &` and `tail -f beir.log`.

### Memory

Loading dense + sparse (+ reranker, if requested) alongside the index can spike
RAM. If a run dies with exit 137 (SIGKILL / OOM), lower `--corpus-batch` (default
64) and/or run only `--methods dense,hybrid` so the FlashRank model is never
loaded. The wrapper runs Python unbuffered, so progress and any OOM point stay
visible in the log rather than being lost to stdout buffering.

### Key flags

| flag | default | meaning |
|------|---------|---------|
| `--dataset` | `scifact` | BEIR dataset name |
| `--methods` | `dense,hybrid,hybrid_rerank` | which retrievers to score |
| `--top-k` | `100` | retrieval depth (the k in Recall@k) |
| `--ndcg-k` | `10` | the k in nDCG@k |
| `--rerank-candidates` | `100` | hybrid candidates fed to the reranker |
| `--max-corpus` | `0` (all) | cap corpus size for a quick run |
| `--keep-collection` | off | keep the temp Qdrant collection |
| `--json-out` | — | also write results as JSON |

## Honest caveats (read before quoting numbers)

- **Queries use the shipped embedding path** — no special bge query-instruction
  prefix — because that is exactly how the deployed pipeline embeds queries.
  This measures *our system*, and may sit a little below published bge-base
  numbers that use the prefix.
- The reranker is MS-MARCO-trained, so it is mildly in-domain on BEIR; expect it
  to help most on MS-MARCO-like datasets and little on others.
- `hybrid_rerank` only re-orders the hybrid candidate pool, so its `Recall@100`
  equals `hybrid`; the rerank effect appears in `nDCG@10`.
- Latency (`ms/query`) is retrieval-only, measured on this box, and includes
  model warmth effects — treat it as relative, not absolute.
