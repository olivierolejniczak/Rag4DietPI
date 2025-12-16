# RAG System v44 - Quick Installation

## One-Line Install (Recommended)

For a complete fresh installation with all CRAG fixes:

```bash
bash <(curl -s https://raw.githubusercontent.com/olivierolejniczak/Rag4DietPI/claude/debug-rag-query-F6HAr/fresh-install.sh)
```

This will:
- Clone the repository (if needed)
- Install all dependencies
- Set up the RAG system in `/root`
- Install SearXNG (Docker)
- Download sample data (optional)
- Run ingestion
- Verify everything works

## Manual Installation

If you prefer to do it step by step:

```bash
# 1. Clone repository
git clone https://github.com/olivierolejniczak/Rag4DietPI.git
cd Rag4DietPI
git checkout claude/debug-rag-query-F6HAr

# 2. Run fresh install script
./fresh-install.sh /root
```

## Custom Installation Directory

To install somewhere other than `/root`:

```bash
# Clone first
git clone https://github.com/olivierolejniczak/Rag4DietPI.git
cd Rag4DietPI
git checkout claude/debug-rag-query-F6HAr

# Install to custom location
./fresh-install.sh /home/user/myrag
```

## What Gets Installed

The installation creates:
- `/root/` (or your chosen directory):
  - `query.sh` - Query your documents
  - `ingest.sh` - Ingest documents
  - `status.sh` - Check system status
  - `verify.sh` - Run verification
  - `documents/` - Your document directory
  - `lib/` - Python modules (with CRAG fixes)
  - `cache/` - BM25 index and query cache

- `~/Rag4DietPI/` - Repository with setup scripts

- SearXNG container on port 8085 (if Docker available)

## Prerequisites

The script checks and installs:
- Python 3.8+
- pip and venv
- Docker (optional, for SearXNG)
- Git

On DietPi/Debian/Ubuntu:
```bash
apt update
apt install -y python3 python3-pip python3-venv git
```

For Docker:
```bash
curl -fsSL https://get.docker.com | sh
```

## After Installation

```bash
cd /root

# Test a query
./query.sh "your question"

# Test CRAG (should trigger web search)
./query.sh --debug --full "Alticap"

# Add more documents
cp /path/to/files/* documents/
./ingest.sh
```

## Troubleshooting

If something goes wrong, see detailed guides:
- [FRESH-INSTALL-GUIDE.md](FRESH-INSTALL-GUIDE.md) - Step-by-step installation
- [FIX-ALTICAP-QUERY.md](FIX-ALTICAP-QUERY.md) - CRAG fixes and debugging
- [README-v44.md](README-v44.md) - Full documentation

## What's Fixed in This Branch

This branch (`claude/debug-rag-query-F6HAr`) includes critical CRAG fixes:

✅ Fixed early-return bug that prevented CRAG from triggering
✅ Added debug output for CRAG evaluation
✅ Fixed SearXNG bot detection with proper headers
✅ Added error messages when web search fails

These fixes ensure CRAG web search works correctly when querying content not in your database.
