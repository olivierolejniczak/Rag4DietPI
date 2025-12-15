# RAG System v44 - Quality Feedback Loop

## Overview

v44 builds on v43 by adding a **Quality Feedback Loop** - a self-evaluation system that allows the RAG to assess its own confidence and abstain when uncertain, rather than hallucinating.

## What's New in v44

### 1. Quality Ledger (SQLite)

Persistent storage of quality metrics for every query:

```
cache/quality_ledger.sqlite
```

**Schema:**
- `query_id` - Unique identifier
- `timestamp` - When query was processed
- `query` - The original question
- `retrieval_count` - Number of chunks retrieved
- `retrieval_sources` - List of source files
- `retrieval_crag_used` - Whether CRAG was triggered
- `retrieval_web_used` - Whether web search was used
- `score_retrieval_confidence` - [0-1] Confidence in retrieval
- `score_answer_coverage` - [0-1] Query coverage by answer
- `score_grounding` - [0-1] Answer grounded in sources
- `score_composite` - [0-1] Weighted overall score
- `decision` - confident/low_confidence/abstained
- `feedback` - Human feedback if provided

### 2. Scoring System (NO LLM)

All scores are calculated **deterministically without using the LLM**:

#### Retrieval Confidence
Based on:
- Score dispersion (coefficient of variation)
- Number of distinct sources
- CRAG/web fallback penalties

#### Answer Coverage
Based on:
- Lexical overlap between query and answer
- Penalties for verbose low-coverage answers

#### Grounding Score
Based on:
- Bigram overlap between answer and source chunks
- Ensures answer is grounded in retrieved documents

### 3. Decision Engine

Three possible outcomes:

| Decision | Badge | Meaning |
|----------|-------|---------|
| `confident` | ✅ | High scores, trustworthy answer |
| `low_confidence` | ⚠️ | Some concerns, use with caution |
| `abstained` | 🛑 | Too uncertain, sources listed instead |

**Thresholds (configurable in config.env):**
```bash
CONFIDENCE_THRESHOLD_HIGH=0.7
CONFIDENCE_THRESHOLD_LOW=0.4
COVERAGE_THRESHOLD=0.3
GROUNDING_THRESHOLD=0.5
```

### 4. Abstention

When the system abstains, it:
- Does NOT generate a potentially hallucinated answer
- Lists available sources
- Shows quality scores
- Recommends human verification

**Example abstention output:**
```
Je ne peux pas répondre avec confiance à cette question. Voici les sources trouvées :

Sources disponibles:
  • document1.pdf
  • document2.docx

[Scores: retrieval=0.35, grounding=0.28, coverage=0.42]

Recommandation: Vérifiez manuellement les sources ou reformulez la question.
```

### 5. Human Feedback

Record feedback for continuous improvement:

```bash
./query.sh "question" --feedback correct
./query.sh "question" --feedback incorrect
./query.sh "question" --feedback partial
```

## New CLI Options

```bash
# Quality features
--no-abstention    # Disable abstention (always answer)
--feedback X       # Record feedback (correct/incorrect/partial)
--ledger-stats     # Show quality statistics
--ledger-recent    # Show recent 10 entries

# Debug
--debug            # Shows quality scores and decision
DEBUG_QUALITY=true # Environment variable for quality-specific debug
```

## Example: Query with Debug

```bash
DEBUG=true ./query.sh "What is NinjaRMM?"
```

Output:
```
============================================================
RAG Query v44
============================================================
Query: What is NinjaRMM?...
Mode: DEFAULT | HyDE=false | Rerank=false
============================================================

[SEARCH] Found 5 results

[POST-RETRIEVAL] 5 → 3 chunks

[QUALITY SCORES]
  Retrieval confidence: 0.723
  Answer coverage:      0.856
  Grounding:            0.691
  Composite:            0.748
  Decision: confident - High confidence: composite=0.75
  Ledger ID: a1b2c3d4e5f6

============================================================
ANSWER
============================================================
NinjaRMM is a remote monitoring and management platform...

✅ Confidence: CONFIDENT
   [Ledger: a1b2c3d4e5f6]

---
Sources:
[1] ninjaone-overview.pdf
[2] rmm-tools-comparison.docx

⏱️ 45.2s
```

## Example: Ledger Entry

```bash
./query.sh --ledger-recent
```

```json
[
  {
    "query_id": "a1b2c3d4e5f6",
    "timestamp": "2024-12-13T10:30:45.123456",
    "query": "What is NinjaRMM?",
    "retrieval_count": 3,
    "retrieval_sources": "[\"ninjaone-overview.pdf\", \"rmm-tools.docx\"]",
    "retrieval_crag_used": 0,
    "retrieval_web_used": 0,
    "score_retrieval_confidence": 0.723,
    "score_answer_coverage": 0.856,
    "score_grounding": 0.691,
    "score_composite": 0.748,
    "decision": "confident",
    "answer_length": 234,
    "response_time_ms": 45200,
    "feedback": null,
    "metadata": "{\"model\": \"qwen2.5:1.5b\", \"mode\": \"default\"}"
  }
]
```

## Statistics

```bash
./query.sh --ledger-stats
```

```json
{
  "total": 127,
  "confident": 89,
  "low_confidence": 31,
  "abstained": 7,
  "avg_composite_score": 0.672,
  "avg_response_time_ms": 52340
}
```

## Configuration (config.env)

New v44 settings:

```bash
# Quality Feedback Loop
QUALITY_LEDGER_ENABLED=true
QUALITY_LEDGER_PATH=cache/quality_ledger.sqlite

# Thresholds
CONFIDENCE_THRESHOLD_HIGH=0.7
CONFIDENCE_THRESHOLD_LOW=0.4
COVERAGE_THRESHOLD=0.3
GROUNDING_THRESHOLD=0.5

# Abstention
ABSTENTION_ENABLED=true
ABSTENTION_MESSAGE="Je ne peux pas répondre avec confiance..."

# Debug
DEBUG_QUALITY=false
```

## Backward Compatibility

v44 is fully backward compatible with v43:
- All v43 features preserved
- Default behavior unchanged if quality features disabled
- Config v43 works without modification (sensible defaults)

To disable v44 features:
```bash
QUALITY_LEDGER_ENABLED=false
ABSTENTION_ENABLED=false
```

## Architecture

```
lib/
├── quality_ledger.py    # SQLite persistence
├── scoring.py           # Deterministic scoring (NO LLM)
├── decision_engine.py   # Confidence decision logic
├── ... (all v43 modules unchanged)

cache/
├── quality_ledger.sqlite  # Quality metrics DB
├── bm25_index.pkl
├── vocabulary.json
├── query_cache.json
├── memory.json
```

## Why This Matters

> **v44 is the first truly self-evaluating version.**
> It doesn't learn intelligently yet, but it **knows when it knows and when it doesn't.**

This is crucial for:
- **Production reliability** - No silent hallucinations
- **Trust building** - Users know when to verify
- **Continuous improvement** - Feedback loop enables future learning
- **Audit trail** - Every query documented with quality metrics

## Installation

```bash
# Fresh install
./setup-rag-core-v44.sh
./setup-rag-ingest-v44.sh
./setup-rag-query-v44.sh
./ingest.sh
./query.sh "test question"

# Upgrade from v43
./setup-rag-query-v44.sh  # Only query script needs update
```

## Files

| File | Lines | Changes from v43 |
|------|-------|------------------|
| setup-rag-core-v44.sh | ~920 | +Quality config |
| setup-rag-ingest-v44.sh | ~1340 | Unchanged |
| setup-rag-query-v44.sh | ~3450 | +Quality modules, +CLI flags |

## Troubleshooting

### Ledger not created
```bash
ls -la cache/
# Should show quality_ledger.sqlite
# If missing, check write permissions
```

### Scores always low
- Check vocabulary.json exists (for coverage)
- Verify documents are properly indexed
- Try --debug to see intermediate scores

### Abstention too frequent
Lower thresholds in config.env:
```bash
GROUNDING_THRESHOLD=0.3  # Was 0.5
```

### Abstention never triggers
Raise thresholds or check scores with --debug
