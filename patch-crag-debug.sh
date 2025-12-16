#!/bin/bash

# Patch query.sh to add CRAG debug logging
# This shows when CRAG is triggered and why web search fails

echo "============================================================"
echo "CRAG Debug Patch"
echo "============================================================"
echo "This patch adds debug logging to show:"
echo "  - When CRAG evaluates retrieval quality"
echo "  - Retrieval quality score vs threshold"
echo "  - When CRAG triggers web search"
echo "  - Web search errors (SearXNG not running, etc.)"
echo ""

if [ ! -f "query.sh" ]; then
    echo "✗ Error: query.sh not found in current directory"
    echo "  Run from /root or wherever query.sh is installed"
    exit 1
fi

echo "Backing up query.sh..."
cp query.sh query.sh.backup-crag-debug
echo "✓ Backup created: query.sh.backup-crag-debug"

echo ""
echo "Applying patch..."

# Create patched version with debug logging
python3 << 'EOFPY'
import re

with open('query.sh', 'r') as f:
    content = f.read()

# Find and patch evaluate_retrieval_quality function
old_eval = r'''def evaluate_retrieval_quality\(query, chunks, threshold=0\.4\):
    """
    Evaluate if retrieved chunks are relevant enough\.
    Returns: \(is_sufficient, confidence_score\)
    """
    if not chunks:
        return False, 0\.0

    # Simple heuristic: check keyword overlap and scores
    query_words = set\(query\.lower\(\)\.split\(\)\)

    total_score = 0
    for chunk in chunks:
        text = chunk\.get\("text", ""\)\.lower\(\) if isinstance\(chunk, dict\) else str\(chunk\)\.lower\(\)
        chunk_words = set\(text\.split\(\)\)

        # Keyword overlap
        overlap = len\(query_words & chunk_words\) / max\(len\(query_words\), 1\)

        # RRF/rerank score
        score = chunk\.get\("rrf_score", 0\) or chunk\.get\("rerank_score", 0\)

        total_score \+= overlap \* 0\.5 \+ score \* 0\.5

    avg_score = total_score / len\(chunks\)
    return avg_score >= threshold, avg_score'''

new_eval = '''def evaluate_retrieval_quality(query, chunks, threshold=0.4):
    """
    Evaluate if retrieved chunks are relevant enough.
    Returns: (is_sufficient, confidence_score)
    """
    import os
    debug = os.environ.get("DEBUG", "false").lower() == "true"

    if not chunks:
        if debug:
            print(f"  [CRAG] No chunks to evaluate")
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

    if debug:
        print(f"  [CRAG] Retrieval quality evaluation:")
        print(f"    Score: {avg_score:.4f} | Threshold: {threshold}")
        print(f"    Decision: {'✓ SUFFICIENT' if avg_score >= threshold else '✗ INSUFFICIENT - triggering web search'}")

    return avg_score >= threshold, avg_score'''

# Replace evaluate_retrieval_quality
content = re.sub(old_eval, new_eval, content, flags=re.DOTALL)

# Find and patch search_web function to show errors in DEBUG mode
old_search_error = r'''    except requests\.exceptions\.ConnectionError:
        search_record\["error"\] = "Connection failed - is SearXNG running\?"
        _web_debug\["searches"\]\.append\(search_record\)
        return \[\]'''

new_search_error = '''    except requests.exceptions.ConnectionError:
        search_record["error"] = "Connection failed - is SearXNG running?"
        _web_debug["searches"].append(search_record)

        # Show error in DEBUG mode (not just DEBUG_WEB)
        import os
        if os.environ.get("DEBUG", "false").lower() == "true":
            print(f"  [WEB SEARCH] ✗ SearXNG not accessible at {searxng_url}")
            print(f"  [WEB SEARCH] Install: docker run -d -p 8085:8080 searxng/searxng")

        return []'''

content = re.sub(old_search_error, new_search_error, content, flags=re.DOTALL)

# Also patch timeout error
old_timeout = r'''    except requests\.exceptions\.Timeout:
        search_record\["error"\] = f"Timeout \(\{timeout\}s\)"
        _web_debug\["searches"\]\.append\(search_record\)
        return \[\]'''

new_timeout = '''    except requests.exceptions.Timeout:
        search_record["error"] = f"Timeout ({timeout}s)"
        _web_debug["searches"].append(search_record)

        # Show error in DEBUG mode
        import os
        if os.environ.get("DEBUG", "false").lower() == "true":
            print(f"  [WEB SEARCH] ✗ Timeout after {timeout}s")

        return []'''

content = re.sub(old_timeout, new_timeout, content, flags=re.DOTALL)

# Write patched version
with open('query.sh', 'w') as f:
    f.write(content)

print("✓ Patched evaluate_retrieval_quality with debug logging")
print("✓ Patched search_web to show errors in DEBUG mode")
EOFPY

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ Patch applied successfully"
    echo "============================================================"
    echo ""
    echo "Now when you run queries with --debug, you'll see:"
    echo "  - CRAG quality evaluation score"
    echo "  - Whether CRAG triggers web search"
    echo "  - Web search errors if SearXNG is not running"
    echo ""
    echo "Test with:"
    echo "  ./query.sh --clear-cache"
    echo "  ./query.sh --debug --full \"Alticap\""
    echo ""
    echo "To restore original:"
    echo "  cp query.sh.backup-crag-debug query.sh"
    echo ""
else
    echo "✗ Patch failed"
    echo "Restoring backup..."
    cp query.sh.backup-crag-debug query.sh
    exit 1
fi
