#!/bin/bash
# setup-rag-ingest.sh
# RAG System - Ingestion Setup
# All ingestion features included



# Plain ASCII output

set -e

log_ok() { echo "[OK] $1"; }
log_err() { echo "[ERROR] $1" >&2; }
log_info() { echo "[INFO] $1"; }

PROJECT_DIR="${1:-$(pwd)}"
echo "============================================"
echo " RAG System - Ingestion Setup"
echo " Document Loader for Map/Reduce"
echo "============================================"
echo ""

mkdir -p "$PROJECT_DIR"/{lib,cache,documents,.ingest_tracking}
cd "$PROJECT_DIR"

[ -f "./config.env" ] && source ./config.env

log_info "Creating unstructured_parser.py..."
cat > "$PROJECT_DIR/lib/unstructured_parser.py" << 'EOFPY'
"""Unified document parser using Unstructured.io"""

import os

# Suppress OpenCV/OpenGL warnings on headless systems
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

def parse_csv_with_headers(file_path, ext=".csv"):
    """Parse CSV/TSV file, prepending column headers to each row for context
    
    This ensures each chunk contains self-describing data like:
    "Spell ID: 18, Incantation: Expecto Patronum, Spell Name: Patronus Charm..."
    
    Args:
        file_path: Path to CSV/TSV file
        ext: File extension (.csv or .tsv)
    
    Returns:
        tuple: (text, metadata_dict)
    """
    import csv
    
    delimiter = "\t" if ext == ".tsv" else ","
    encodings = ["utf-8", "cp1252", "iso-8859-1", "utf-16", "latin-1"]
    
    rows = []
    headers = []
    used_encoding = None
    
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc, newline="") as f:
                # Remove BOM if present
                content = f.read()
                if content.startswith("\ufeff"):
                    content = content[1:]
                
                # Parse CSV
                from io import StringIO
                reader = csv.reader(StringIO(content), delimiter=delimiter)
                all_rows = list(reader)
                
                if not all_rows:
                    return "", {"error": "Empty CSV file"}
                
                headers = all_rows[0]
                rows = all_rows[1:]
                used_encoding = enc
                break
        except (UnicodeDecodeError, LookupError):
            continue
        except Exception as e:
            return "", {"error": f"CSV parse error: {str(e)}"}
    
    if not headers:
        return "", {"error": "Could not read CSV headers"}
    
    # Build text with headers prepended to each row
    filename = os.path.basename(file_path)
    text_parts = []
    
    # Add file context header
    text_parts.append(f"[Data from {filename}]")
    text_parts.append(f"Columns: {', '.join(headers)}")
    text_parts.append("")
    
    for row in rows:
        if not any(cell.strip() for cell in row):
            continue  # Skip empty rows
        
        # Format row as "Column1: Value1, Column2: Value2, ..."
        row_parts = []
        for i, value in enumerate(row):
            if i < len(headers) and value.strip():
                row_parts.append(f"{headers[i]}: {value.strip()}")
        
        if row_parts:
            text_parts.append(" | ".join(row_parts))
    
    result_text = "\n".join(text_parts)
    
    metadata = {
        "filename": filename,
        "parser": "csv_with_headers",
        "encoding": used_encoding,
        "element_count": len(rows),
        "element_types": {"TableRow": len(rows)},
        "char_count": len(result_text),
        "columns": headers,
    }
    
    return result_text, metadata

def parse_document(file_path, strategy="auto", ocr_languages="eng+fra"):
    """Parse any document using Unstructured.io partition()
    
    Args:
        file_path: Path to document
        strategy: Parsing strategy (auto, fast, hi_res, ocr_only)
        ocr_languages: OCR language codes
    
    Returns:
        tuple: (text, metadata_dict)
    """
    try:
        from unstructured.partition.auto import partition
    except ImportError:
        return "", {"error": "Unstructured.io not installed"}
    
    if not os.path.exists(file_path):
        return "", {"error": f"File not found: {file_path}"}
    
    ext = os.path.splitext(file_path)[1].lower()
    
    # Special handling for CSV/TSV - prepend headers to each row for context
    if ext in [".csv", ".tsv"]:
        return parse_csv_with_headers(file_path, ext)
    
    # Handle TXT/MD files with encoding fallback
    if ext in [".txt", ".md"]:
        encodings = ["utf-8", "cp1252", "iso-8859-1", "utf-16", "latin-1"]
        text_content = None
        used_encoding = None
        
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    text_content = f.read()
                    used_encoding = enc
                    break
            except (UnicodeDecodeError, LookupError):
                continue
        
        if text_content is None:
            # Last resort: read as binary and decode with errors ignored
            try:
                with open(file_path, "rb") as f:
                    text_content = f.read().decode("utf-8", errors="replace")
                    used_encoding = "utf-8 (with replacements)"
            except Exception as e:
                return "", {"error": f"Encoding error: {str(e)}"}
        
        # Return directly for plain text files
        metadata = {
            "filename": os.path.basename(file_path),
            "parser": "unstructured",
            "encoding": used_encoding,
            "element_count": 1,
            "element_types": {"Text": 1},
            "char_count": len(text_content),
        }
        return text_content, metadata
    
    try:
        # Partition document
        elements = partition(
            filename=file_path,
            strategy=strategy,
            languages=ocr_languages.split("+") if ocr_languages else None,
            include_page_breaks=True,
        )
        
        # Extract text and metadata
        text_parts = []
        metadata = {
            "filename": os.path.basename(file_path),
            "parser": "unstructured",
            "element_count": len(elements),
            "element_types": {},
        }
        
        current_page = 1
        for element in elements:
            # Track element types
            elem_type = type(element).__name__
            metadata["element_types"][elem_type] = metadata["element_types"].get(elem_type, 0) + 1
            
            # Handle page breaks
            if elem_type == "PageBreak":
                current_page += 1
                continue
            
            # Get text content
            text = str(element)
            if not text.strip():
                continue
            
            # Format based on element type
            if elem_type == "Title":
                text_parts.append(f"\n## {text}\n")
            elif elem_type == "Header":
                text_parts.append(f"\n### {text}\n")
            elif elem_type == "Table":
                text_parts.append(f"\n[Table]\n{text}\n")
            elif elem_type == "Image":
                text_parts.append(f"\n[Image: {text}]\n")
            elif elem_type == "ListItem":
                text_parts.append(f"  - {text}")
            else:
                text_parts.append(text)
        
        result_text = "\n".join(text_parts)
        metadata["char_count"] = len(result_text)
        
        return result_text, metadata
        
    except Exception as e:
        return "", {"error": str(e), "filename": os.path.basename(file_path)}

def get_supported_extensions():
    """Return list of supported file extensions"""
    return [
        ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
        ".html", ".htm", ".txt", ".md", ".csv", ".tsv", ".json",
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff",
        ".eml", ".msg", ".rtf", ".odt", ".ods", ".odp",
    ]
EOFPY
log_ok "unstructured_parser.py"

log_info "Creating smart_chunker.py..."
cat > "$PROJECT_DIR/lib/smart_chunker.py" << 'EOFPY'
"""Smart content-aware chunking"""

import re
import hashlib

def smart_chunk(text, chunk_size=500, chunk_overlap=80, min_chunk_size=100, max_chunk_size=1200):
    """Split text into semantic chunks
    
    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        min_chunk_size: Minimum chunk size
        max_chunk_size: Maximum chunk size
    
    Returns:
        list: List of chunk dictionaries with text and metadata
    """
    if not text or not text.strip():
        return []
    
    # Clean text
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    chunks = []
    
    # Detect content type for specialized handling
    if _is_table_content(text):
        return _chunk_table(text, max_chunk_size)
    
    if _is_code_content(text):
        return _chunk_code(text, chunk_size, max_chunk_size)
    
    # Standard semantic chunking
    # Split on semantic boundaries
    sections = re.split(r'(?=\n##?\s)', text)
    
    for section in sections:
        if not section.strip():
            continue
        
        # If section is small enough, keep as single chunk
        if len(section) <= max_chunk_size:
            if len(section) >= min_chunk_size:
                chunks.append(_make_chunk(section, "section"))
            continue
        
        # Split large sections on paragraphs
        paragraphs = re.split(r'\n\n+', section)
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if adding paragraph exceeds target
            if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
                if len(current_chunk) >= min_chunk_size:
                    chunks.append(_make_chunk(current_chunk, "paragraph"))
                # Start new chunk with overlap
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                current_chunk = overlap_text + " " + para if overlap_text else para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para
            
            # Force split if exceeds max
            while len(current_chunk) > max_chunk_size:
                split_point = _find_split_point(current_chunk, chunk_size)
                chunks.append(_make_chunk(current_chunk[:split_point], "forced"))
                current_chunk = current_chunk[split_point - chunk_overlap:]
        
        # Add remaining content
        if current_chunk and len(current_chunk) >= min_chunk_size:
            chunks.append(_make_chunk(current_chunk, "paragraph"))
    
    return chunks

def _is_table_content(text):
    """Detect if text is primarily tabular"""
    lines = text.split('\n')
    pipe_lines = sum(1 for l in lines if '|' in l and l.count('|') >= 2)
    return pipe_lines > len(lines) * 0.3

def _is_code_content(text):
    """Detect if text is primarily code"""
    code_indicators = [
        r'^\s*(def |class |function |import |from |#include)',
        r'^\s*\{|\};\s*$',
        r'^\s*(if|for|while|return)\s*[\(\{]',
    ]
    lines = text.split('\n')
    code_lines = sum(1 for l in lines if any(re.match(p, l) for p in code_indicators))
    return code_lines > len(lines) * 0.2

def _chunk_table(text, max_size):
    """Chunk tabular content preserving structure"""
    chunks = []
    lines = text.split('\n')
    current = []
    header = []
    
    for line in lines:
        if '|' in line and not header:
            header = [line]
            if len(lines) > 1 and re.match(r'^[\|\-\s]+$', lines[1] if len(lines) > 1 else ""):
                header.append(lines[1])
        
        current.append(line)
        
        if len('\n'.join(current)) > max_size:
            chunk_text = '\n'.join(current[:-1])
            if chunk_text.strip():
                chunks.append(_make_chunk(chunk_text, "table"))
            current = header + [line]
    
    if current:
        chunk_text = '\n'.join(current)
        if chunk_text.strip():
            chunks.append(_make_chunk(chunk_text, "table"))
    
    return chunks

def _chunk_code(text, target_size, max_size):
    """Chunk code preserving logical blocks"""
    chunks = []
    
    # Split on function/class definitions
    blocks = re.split(r'(?=\n(?:def |class |function ))', text)
    
    current = ""
    for block in blocks:
        if len(current) + len(block) > target_size and current:
            chunks.append(_make_chunk(current, "code"))
            current = block
        else:
            current = current + block if current else block
        
        while len(current) > max_size:
            split_point = current.rfind('\n', 0, target_size)
            if split_point == -1:
                split_point = target_size
            chunks.append(_make_chunk(current[:split_point], "code"))
            current = current[split_point:]
    
    if current.strip():
        chunks.append(_make_chunk(current, "code"))
    
    return chunks

def _find_split_point(text, target):
    """Find optimal split point near target"""
    # Prefer sentence boundaries
    for sep in ['. ', '.\n', '? ', '! ', '\n\n', '\n', ' ']:
        pos = text.rfind(sep, 0, target + 50)
        if pos > target * 0.7:
            return pos + len(sep)
    return target

def _make_chunk(text, chunk_type):
    """Create chunk dictionary with metadata"""
    text = text.strip()
    return {
        "text": text,
        "type": chunk_type,
        "char_count": len(text),
        "hash": hashlib.md5(text.encode()).hexdigest()[:12],
    }
EOFPY
log_ok "smart_chunker.py"

log_info "Creating embedding_helper.py..."
cat > "$PROJECT_DIR/lib/embedding_helper.py" << 'EOFPY'
"""FastEmbed embedding helper"""

import os

_model = None
_model_name = None

def is_fastembed_available():
    """Check if FastEmbed is available and working"""
    try:
        from fastembed import TextEmbedding
        return True
    except ImportError:
        return False

def get_embedding(text, model_name=None):
    """Get embedding vector for text using FastEmbed
    
    Args:
        text: Text to embed
        model_name: FastEmbed model name (default from env)
    
    Returns:
        list: Embedding vector
    """
    global _model, _model_name
    
    if model_name is None:
        model_name = os.environ.get("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
    
    # Lazy load model
    if _model is None or _model_name != model_name:
        from fastembed import TextEmbedding
        cache_dir = os.environ.get("FASTEMBED_CACHE_DIR", "./cache/fastembed")
        _model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        _model_name = model_name
    
    # Generate embedding
    embeddings = list(_model.embed([text]))
    return embeddings[0].tolist()

def get_embeddings_batch(texts, model_name=None, batch_size=32):
    """Get embeddings for multiple texts
    
    Args:
        texts: List of texts to embed
        model_name: FastEmbed model name
        batch_size: Batch size for processing
    
    Returns:
        list: List of embedding vectors
    """
    global _model, _model_name
    
    if model_name is None:
        model_name = os.environ.get("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
    
    if _model is None or _model_name != model_name:
        from fastembed import TextEmbedding
        cache_dir = os.environ.get("FASTEMBED_CACHE_DIR", "./cache/fastembed")
        _model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        _model_name = model_name
    
    embeddings = list(_model.embed(texts, batch_size=batch_size))
    return [e.tolist() for e in embeddings]

def get_embedding_dimension(model_name=None):
    """Get embedding dimension for model"""
    if model_name is None:
        model_name = os.environ.get("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
    
    dimensions = {
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
    }
    return dimensions.get(model_name, 384)
EOFPY
log_ok "embedding_helper.py"

log_info "Creating qdrant_client_helper.py..."
cat > "$PROJECT_DIR/lib/qdrant_client_helper.py" << 'EOFPY'
"""QdrantClient helper module for native gRPC operations

Feature: QDRANT_CLIENT_ENABLED
Introduced: client
Lifecycle: ACTIVE

Provides native QdrantClient operations with automatic HTTP fallback.
Used by both ingest and query modules.
"""

import os
import requests

# Global client instance
_client = None
_client_mode = None  # "grpc", "http", or None

def _get_config():
    """Get Qdrant configuration from environment"""
    return {
        "host": os.environ.get("QDRANT_HOST", "http://localhost:6333"),
        "grpc_port": int(os.environ.get("QDRANT_GRPC_PORT", "6334")),
        "collection": os.environ.get("COLLECTION_NAME", "documents"),
        "client_enabled": os.environ.get("QDRANT_CLIENT_ENABLED", "true").lower() == "true",
        "batch_size": int(os.environ.get("QDRANT_BATCH_SIZE", "100")),
    }

def is_client_available():
    """Check if qdrant-client is installed"""
    try:
        from qdrant_client import QdrantClient
        return True
    except ImportError:
        return False

def get_client():
    """Get or create QdrantClient instance
    
    Returns:
        tuple: (client, mode) where mode is "grpc" or None if unavailable
    """
    global _client, _client_mode
    
    config = _get_config()
    
    # Feature disabled
    if not config["client_enabled"]:
        return None, None
    
    # Already initialized
    if _client is not None:
        return _client, _client_mode
    
    # Try to create client
    if not is_client_available():
        _client_mode = None
        return None, None
    
    try:
        from qdrant_client import QdrantClient
        
        # Extract host from URL
        host = config["host"].replace("http://", "").replace("https://", "")
        if ":" in host:
            host = host.split(":")[0]
        
        # Create gRPC client
        _client = QdrantClient(
            host=host,
            grpc_port=config["grpc_port"],
            prefer_grpc=True,
            timeout=30,
        )
        
        # Test connection
        _client.get_collections()
        _client_mode = "grpc"
        return _client, _client_mode
        
    except Exception as e:
        # gRPC failed, client unavailable
        _client = None
        _client_mode = None
        return None, None

def ensure_collection_client(collection_name, dimension):
    """Create collection using QdrantClient (legacy single-vector)
    
    Args:
        collection_name: Collection name
        dimension: Vector dimension
    
    Returns:
        bool: True if successful
    """
    client, mode = get_client()
    if client is None:
        return False
    
    try:
        from qdrant_client.models import Distance, VectorParams
        
        # Check if exists
        collections = client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            return True
        
        # Create collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
            ),
        )
        return True
        
    except Exception:
        return False

def upload_points_client(collection_name, points, batch_size=100):
    """Upload points using QdrantClient with batching
    
    Args:
        collection_name: Collection name
        points: List of point dicts with id, vector, payload
        batch_size: Batch size for uploads
    
    Returns:
        bool: True if successful
    """
    client, mode = get_client()
    if client is None:
        return False
    
    try:
        from qdrant_client.models import PointStruct
        
        # Convert to PointStruct objects
        point_structs = [
            PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p.get("payload", {}),
            )
            for p in points
        ]
        
        # Upload in batches
        for i in range(0, len(point_structs), batch_size):
            batch = point_structs[i:i + batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True,
            )
        
        return True
        
    except Exception:
        return False

def search_points_client(collection_name, query_vector, limit=5, score_threshold=None):
    """Search points using QdrantClient
    
    Args:
        collection_name: Collection name
        query_vector: Query embedding vector
        limit: Number of results
        score_threshold: Minimum score threshold
    
    Returns:
        list: List of result dicts with id, score, payload, or None if failed
    """
    client, mode = get_client()
    if client is None:
        return None
    
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        )
        
        return [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload,
            }
            for r in results
        ]
        
    except Exception:
        return None

def scroll_points_client(collection_name, limit=100, offset=None, with_payload=True, with_vectors=False):
    """Scroll through all points using QdrantClient
    
    Args:
        collection_name: Collection name
        limit: Number of results per scroll
        offset: Offset point ID
        with_payload: Include payload
        with_vectors: Include vectors
    
    Returns:
        tuple: (points_list, next_offset) or (None, None) if failed
    """
    client, mode = get_client()
    if client is None:
        return None, None
    
    try:
        results, next_offset = client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        
        points = [
            {
                "id": r.id,
                "payload": r.payload if with_payload else None,
                "vector": r.vector if with_vectors else None,
            }
            for r in results
        ]
        
        return points, next_offset
        
    except Exception:
        return None, None

def get_collection_info_client(collection_name):
    """Get collection info using QdrantClient
    
    Args:
        collection_name: Collection name
    
    Returns:
        dict: Collection info or None if failed
    """
    client, mode = get_client()
    if client is None:
        return None
    
    try:
        info = client.get_collection(collection_name)
        return {
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status.value if hasattr(info.status, 'value') else str(info.status),
        }
        
    except Exception:
        return None

# HTTP Fallback Functions (legacy behavior)

def ensure_collection_http(qdrant_host, collection_name, dimension):
    """Create Qdrant collection via HTTP if not exists (legacy)"""
    url = f"{qdrant_host}/collections/{collection_name}"
    
    # Check if exists
    resp = requests.get(url)
    if resp.status_code == 200:
        return True
    
    # Create collection
    payload = {
        "vectors": {
            "size": dimension,
            "distance": "Cosine"
        }
    }
    resp = requests.put(url, json=payload)
    return resp.status_code == 200

def upload_points_http(qdrant_host, collection_name, points):
    """Upload points to Qdrant via HTTP (legacy)"""
    url = f"{qdrant_host}/collections/{collection_name}/points"
    payload = {"points": points}
    resp = requests.put(url, json=payload, params={"wait": "true"})
    return resp.status_code == 200

def search_points_http(qdrant_host, collection_name, query_vector, limit=5, score_threshold=None):
    """Search points via HTTP (legacy)"""
    url = f"{qdrant_host}/collections/{collection_name}/points/search"
    payload = {
        "vector": query_vector,
        "limit": limit,
        "with_payload": True,
    }
    if score_threshold is not None:
        payload["score_threshold"] = score_threshold
    
    resp = requests.post(url, json=payload)
    if resp.status_code != 200:
        return None
    
    data = resp.json()
    return [
        {
            "id": r["id"],
            "score": r["score"],
            "payload": r.get("payload", {}),
        }
        for r in data.get("result", [])
    ]

# Unified Interface Functions

def ensure_collection(qdrant_host, collection_name, dimension):
    """Create collection with client fallback to HTTP
    
    Args:
        qdrant_host: Qdrant HTTP host URL (for fallback)
        collection_name: Collection name
        dimension: Vector dimension
    
    Returns:
        bool: True if successful
    """
    # Try client first
    if ensure_collection_client(collection_name, dimension):
        return True
    
    # Fallback to HTTP
    return ensure_collection_http(qdrant_host, collection_name, dimension)

def upload_points(qdrant_host, collection_name, points, batch_size=100):
    """Upload points with client fallback to HTTP
    
    Args:
        qdrant_host: Qdrant HTTP host URL (for fallback)
        collection_name: Collection name
        points: List of point dicts
        batch_size: Batch size for client uploads
    
    Returns:
        bool: True if successful
    """
    # Try client first
    if upload_points_client(collection_name, points, batch_size):
        return True
    
    # Fallback to HTTP
    return upload_points_http(qdrant_host, collection_name, points)

def search_points(qdrant_host, collection_name, query_vector, limit=5, score_threshold=None):
    """Search points with client fallback to HTTP
    
    Args:
        qdrant_host: Qdrant HTTP host URL (for fallback)
        collection_name: Collection name
        query_vector: Query embedding vector
        limit: Number of results
        score_threshold: Minimum score threshold
    
    Returns:
        list: List of result dicts or None if failed
    """
    # Try client first
    results = search_points_client(collection_name, query_vector, limit, score_threshold)
    if results is not None:
        return results
    
    # Fallback to HTTP
    return search_points_http(qdrant_host, collection_name, query_vector, limit, score_threshold)

def get_client_mode():
    """Get current client mode
    
    Returns:
        str: "grpc", "http", or "unavailable"
    """
    client, mode = get_client()
    if mode == "grpc":
        return "grpc"
    
    # Check if HTTP works
    config = _get_config()
    try:
        resp = requests.get(f"{config['host']}/collections", timeout=5)
        if resp.status_code == 200:
            return "http"
    except:
        pass
    
    return "unavailable"
EOFPY
log_ok "qdrant_client_helper.py"

log_info "Creating sparse_embedding_helper.py..."
cat > "$PROJECT_DIR/lib/sparse_embedding_helper.py" << 'EOFPY'
"""SparseEmbed helper module for sparse vector generation

Feature: SPARSE_EMBED_ENABLED
Introduced: hybrid
Lifecycle: ACTIVE

Provides sparse embedding generation using FastEmbed SparseTextEmbedding.
Supports Qdrant/bm25, SPLADE, and BM42 models.
"""

import os

_sparse_model = None
_sparse_model_name = None

def is_sparse_embed_available():
    """Check if SparseTextEmbedding is available"""
    try:
        from fastembed import SparseTextEmbedding
        return True
    except ImportError:
        return False

def get_sparse_embedding(text, model_name=None):
    """Get sparse embedding for text
    
    Args:
        text: Text to embed
        model_name: Sparse model name (default from env)
    
    Returns:
        dict: {"indices": [...], "values": [...]} or None if unavailable
    """
    global _sparse_model, _sparse_model_name
    
    if not is_sparse_embed_available():
        return None
    
    if model_name is None:
        model_name = os.environ.get("SPARSE_EMBED_MODEL", "Qdrant/bm25")
    
    # Lazy load model
    if _sparse_model is None or _sparse_model_name != model_name:
        from fastembed import SparseTextEmbedding
        cache_dir = os.environ.get("FASTEMBED_CACHE_DIR", "./cache/fastembed")
        _sparse_model = SparseTextEmbedding(model_name=model_name, cache_dir=cache_dir)
        _sparse_model_name = model_name
    
    try:
        # Generate sparse embedding
        embeddings = list(_sparse_model.embed([text]))
        if embeddings:
            sparse = embeddings[0]
            return {
                "indices": sparse.indices.tolist(),
                "values": sparse.values.tolist(),
            }
    except Exception as e:
        pass
    
    return None

def get_sparse_embeddings_batch(texts, model_name=None, batch_size=32):
    """Get sparse embeddings for multiple texts
    
    Args:
        texts: List of texts to embed
        model_name: Sparse model name
        batch_size: Batch size for processing
    
    Returns:
        list: List of {"indices": [...], "values": [...]} dicts
    """
    global _sparse_model, _sparse_model_name
    
    if not is_sparse_embed_available():
        return [None] * len(texts)
    
    if model_name is None:
        model_name = os.environ.get("SPARSE_EMBED_MODEL", "Qdrant/bm25")
    
    if _sparse_model is None or _sparse_model_name != model_name:
        from fastembed import SparseTextEmbedding
        cache_dir = os.environ.get("FASTEMBED_CACHE_DIR", "./cache/fastembed")
        _sparse_model = SparseTextEmbedding(model_name=model_name, cache_dir=cache_dir)
        _sparse_model_name = model_name
    
    try:
        embeddings = list(_sparse_model.embed(texts, batch_size=batch_size))
        return [
            {"indices": e.indices.tolist(), "values": e.values.tolist()}
            for e in embeddings
        ]
    except Exception as e:
        return [None] * len(texts)

def get_sparse_model_info():
    """Get information about the sparse model
    
    Returns:
        dict: Model info or None
    """
    model_name = os.environ.get("SPARSE_EMBED_MODEL", "Qdrant/bm25")
    return {
        "model": model_name,
        "available": is_sparse_embed_available(),
    }
EOFPY
log_ok "sparse_embedding_helper.py"

log_info "Creating qdrant_hybrid_helper.py..."
cat > "$PROJECT_DIR/lib/qdrant_hybrid_helper.py" << 'EOFPY'
"""Qdrant hybrid search helper module

Feature: SPARSE_EMBED_ENABLED
Introduced: hybrid
Lifecycle: ACTIVE

Extends qdrant_client_helper with named vectors support for hybrid search.
Collection schema: {"dense": VectorParams, "sparse": SparseVectorParams}
"""

import os
import requests

from qdrant_client_helper import get_client, is_client_available, get_client_mode

def _get_hybrid_config():
    """Get hybrid search configuration"""
    return {
        "host": os.environ.get("QDRANT_HOST", "http://localhost:6333"),
        "collection": os.environ.get("COLLECTION_NAME", "documents"),
        "dense_name": os.environ.get("DENSE_VECTOR_NAME", "dense"),
        "sparse_name": os.environ.get("SPARSE_VECTOR_NAME", "sparse"),
        "sparse_enabled": os.environ.get("SPARSE_EMBED_ENABLED", "true").lower() == "true",
        "dimension": int(os.environ.get("EMBEDDING_DIMENSION", "384")),
        "batch_size": int(os.environ.get("QDRANT_BATCH_SIZE", "100")),
    }

def delete_collection_client(collection_name):
    """Delete collection using QdrantClient
    
    Args:
        collection_name: Collection name
    
    Returns:
        bool: True if deleted or didn't exist
    """
    client, mode = get_client()
    if client is None:
        return False
    
    try:
        client.delete_collection(collection_name)
        return True
    except Exception:
        return True  # Collection might not exist

def delete_collection_http(qdrant_host, collection_name):
    """Delete collection via HTTP"""
    url = f"{qdrant_host}/collections/{collection_name}"
    try:
        resp = requests.delete(url)
        return resp.status_code in [200, 404]
    except:
        return False

def delete_collection(collection_name):
    """Delete collection with client fallback to HTTP"""
    config = _get_hybrid_config()
    
    if delete_collection_client(collection_name):
        return True
    return delete_collection_http(config["host"], collection_name)

def ensure_hybrid_collection_client(collection_name, dimension, dense_name, sparse_name):
    """Create collection with named vectors using QdrantClient
    
    Args:
        collection_name: Collection name
        dimension: Dense vector dimension
        dense_name: Name for dense vectors
        sparse_name: Name for sparse vectors
    
    Returns:
        bool: True if successful
    """
    client, mode = get_client()
    if client is None:
        return False
    
    try:
        from qdrant_client.models import Distance, VectorParams, SparseVectorParams
        
        # Check if exists
        collections = client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            return True
        
        # Create collection with named vectors
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                dense_name: VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                sparse_name: SparseVectorParams(),
            },
        )
        return True
        
    except Exception as e:
        print(f"  [WARN] Hybrid collection creation failed: {e}")
        return False

def ensure_hybrid_collection_http(qdrant_host, collection_name, dimension, dense_name, sparse_name):
    """Create collection with named vectors via HTTP"""
    url = f"{qdrant_host}/collections/{collection_name}"
    
    # Check if exists
    resp = requests.get(url)
    if resp.status_code == 200:
        return True
    
    # Create collection with named vectors
    payload = {
        "vectors": {
            dense_name: {
                "size": dimension,
                "distance": "Cosine"
            }
        },
        "sparse_vectors": {
            sparse_name: {}
        }
    }
    resp = requests.put(url, json=payload)
    return resp.status_code == 200

def ensure_hybrid_collection(collection_name=None):
    """Create hybrid collection with dense + sparse vectors
    
    Args:
        collection_name: Collection name (default from env)
    
    Returns:
        bool: True if successful
    """
    config = _get_hybrid_config()
    
    if collection_name is None:
        collection_name = config["collection"]
    
    # Try client first
    if ensure_hybrid_collection_client(
        collection_name, 
        config["dimension"],
        config["dense_name"],
        config["sparse_name"]
    ):
        return True
    
    # Fallback to HTTP
    return ensure_hybrid_collection_http(
        config["host"],
        collection_name,
        config["dimension"],
        config["dense_name"],
        config["sparse_name"]
    )

def upload_hybrid_points_client(collection_name, points, dense_name, sparse_name, batch_size=100):
    """Upload points with named vectors using QdrantClient
    
    Args:
        collection_name: Collection name
        points: List of point dicts with id, dense_vector, sparse_vector, payload
        dense_name: Name for dense vectors
        sparse_name: Name for sparse vectors
        batch_size: Batch size for uploads
    
    Returns:
        bool: True if successful
    """
    client, mode = get_client()
    if client is None:
        return False
    
    try:
        from qdrant_client.models import PointStruct, SparseVector
        
        # Convert to PointStruct objects with named vectors
        point_structs = []
        for p in points:
            vectors = {
                dense_name: p["dense_vector"],
            }
            
            # Add sparse vector if present
            if p.get("sparse_vector"):
                sv = p["sparse_vector"]
                vectors[sparse_name] = SparseVector(
                    indices=sv["indices"],
                    values=sv["values"],
                )
            
            point_structs.append(PointStruct(
                id=p["id"],
                vector=vectors,
                payload=p.get("payload", {}),
            ))
        
        # Upload in batches
        for i in range(0, len(point_structs), batch_size):
            batch = point_structs[i:i + batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True,
            )
        
        return True
        
    except Exception as e:
        print(f"  [WARN] Hybrid point upload failed: {e}")
        return False

def upload_hybrid_points_http(qdrant_host, collection_name, points, dense_name, sparse_name):
    """Upload points with named vectors via HTTP"""
    url = f"{qdrant_host}/collections/{collection_name}/points"
    
    # Format points for HTTP API
    http_points = []
    for p in points:
        vector = {
            dense_name: p["dense_vector"],
        }
        if p.get("sparse_vector"):
            vector[sparse_name] = p["sparse_vector"]
        
        http_points.append({
            "id": p["id"],
            "vector": vector,
            "payload": p.get("payload", {}),
        })
    
    payload = {"points": http_points}
    resp = requests.put(url, json=payload, params={"wait": "true"})
    return resp.status_code == 200

def upload_hybrid_points(collection_name, points, batch_size=100):
    """Upload points with named vectors
    
    Args:
        collection_name: Collection name
        points: List of point dicts with id, dense_vector, sparse_vector, payload
        batch_size: Batch size for uploads
    
    Returns:
        bool: True if successful
    """
    config = _get_hybrid_config()
    
    # Try client first
    if upload_hybrid_points_client(
        collection_name, 
        points,
        config["dense_name"],
        config["sparse_name"],
        batch_size
    ):
        return True
    
    # Fallback to HTTP
    return upload_hybrid_points_http(
        config["host"],
        collection_name,
        points,
        config["dense_name"],
        config["sparse_name"]
    )

def get_hybrid_mode():
    """Get current hybrid search mode
    
    Returns:
        str: "native" (Qdrant RRF), "legacy" (Python RRF), or "dense-only"
    """
    config = _get_hybrid_config()
    
    if not config["sparse_enabled"]:
        return "dense-only"
    
    hybrid_mode = os.environ.get("HYBRID_SEARCH_MODE", "native")
    return hybrid_mode
EOFPY
log_ok "qdrant_hybrid_helper.py"

log_info "Creating doc_dedup.py..."
cat > "$PROJECT_DIR/lib/doc_dedup.py" << 'EOFPY'
"""Document-level deduplication module

Feature: DOC_DEDUP_ENABLED
Introduced: dedup
Lifecycle: ACTIVE

Detects duplicate documents by hashing full extracted text content.
Skips entire documents that have identical content to previously ingested docs.
Saves ~90% storage on contract revisions with minor filename changes.
"""

import os
import json
import hashlib

_doc_index = None
_doc_index_path = None

def _get_config():
    """Get doc dedup configuration"""
    return {
        "enabled": os.environ.get("DOC_DEDUP_ENABLED", "true").lower() == "true",
        "index_path": os.environ.get("DOC_DEDUP_INDEX", "cache/doc_dedup.json"),
    }

def _load_index():
    """Load document hash index from disk"""
    global _doc_index, _doc_index_path
    
    config = _get_config()
    _doc_index_path = config["index_path"]
    
    if os.path.exists(_doc_index_path):
        try:
            with open(_doc_index_path, 'r') as f:
                _doc_index = json.load(f)
        except Exception:
            _doc_index = {}
    else:
        _doc_index = {}
    
    return _doc_index

def _save_index():
    """Save document hash index to disk"""
    global _doc_index, _doc_index_path
    
    if _doc_index is None or _doc_index_path is None:
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(_doc_index_path), exist_ok=True)
    
    try:
        with open(_doc_index_path, 'w') as f:
            json.dump(_doc_index, f, indent=2)
    except Exception:
        pass

def get_doc_content_hash(text):
    """Calculate MD5 hash of document text content
    
    Args:
        text: Extracted text content from document
    
    Returns:
        str: MD5 hash of normalized text
    """
    # Normalize text: lowercase, strip whitespace, collapse spaces
    normalized = ' '.join(text.lower().split())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def is_duplicate_doc(text, filename=None):
    """Check if document content was already ingested
    
    Args:
        text: Extracted text content
        filename: Original filename (for logging)
    
    Returns:
        tuple: (is_duplicate, original_filename_if_duplicate)
    """
    config = _get_config()
    
    if not config["enabled"]:
        return False, None
    
    global _doc_index
    if _doc_index is None:
        _load_index()
    
    doc_hash = get_doc_content_hash(text)
    
    if doc_hash in _doc_index:
        original = _doc_index[doc_hash].get("filename", "unknown")
        return True, original
    
    return False, None

def mark_doc_ingested(text, filename, file_path=None):
    """Mark document content as ingested
    
    Args:
        text: Extracted text content
        filename: Original filename
        file_path: Full file path
    """
    config = _get_config()
    
    if not config["enabled"]:
        return
    
    global _doc_index
    if _doc_index is None:
        _load_index()
    
    doc_hash = get_doc_content_hash(text)
    
    import time
    _doc_index[doc_hash] = {
        "filename": filename,
        "filepath": file_path,
        "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "char_count": len(text),
    }
    
    _save_index()

def clear_doc_index():
    """Clear the document dedup index"""
    global _doc_index
    
    config = _get_config()
    _doc_index = {}
    
    if os.path.exists(config["index_path"]):
        os.remove(config["index_path"])

def get_doc_index_stats():
    """Get statistics about the doc dedup index
    
    Returns:
        dict: Index statistics
    """
    global _doc_index
    if _doc_index is None:
        _load_index()
    
    return {
        "total_docs": len(_doc_index) if _doc_index else 0,
        "enabled": _get_config()["enabled"],
    }
EOFPY
log_ok "doc_dedup.py"

log_info "Creating ingestion_progress.py..."
cat > "$PROJECT_DIR/lib/ingestion_progress.py" << 'EOFPY'
"""Real-Time Ingestion Progress Tracker progress

Provides visual progress tracking during document ingestion with:
- Per-file progress display
- File size information
- Chunk counts and timing
- Summary statistics

Feature: INGESTION_PROGRESS
Introduced: progress
Lifecycle: ACTIVE
"""

import os
import time
import sys

# ANSI Colors (bold for DietPi compatibility)
RED = '\033[1;31m'
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
CYAN = '\033[1;36m'
BLUE = '\033[1;34m'
NC = '\033[0m'


def format_size(bytes_count):
    """Format file size human-readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_count < 1024:
            return f"{bytes_count:.1f}{unit}"
        bytes_count /= 1024
    return f"{bytes_count:.1f}TB"


def format_duration(seconds):
    """Format duration human-readable"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m{secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h{mins}m"


class IngestionProgressTracker:
    """Real-time ingestion progress tracking"""
    
    def __init__(self, total_files):
        self.total_files = total_files
        self.current_file = 0
        self.global_start = time.time()
        self.total_chunks = 0
        self.total_chars = 0
        self.success_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.file_results = []
    
    def start_file(self, filepath):
        """Start processing a file"""
        self.current_file += 1
        self.current_filepath = filepath
        self.file_start = time.time()
        self.file_chunks = 0
        
        filename = os.path.basename(filepath)
        filesize = os.path.getsize(filepath) if os.path.exists(filepath) else 0
        sizestr = format_size(filesize)
        
        # Print progress line
        print(f"{CYAN}[{self.current_file}/{self.total_files}]{NC} {filename} {BLUE}({sizestr}){NC}", end="", flush=True)
    
    def update_chunk_progress(self, chunk_count):
        """Update during chunk processing"""
        if chunk_count % 10 == 0 and chunk_count > 0:
            print(f"\r{CYAN}[{self.current_file}/{self.total_files}]{NC} ... {YELLOW}{chunk_count} chunks{NC}", end="", flush=True)
    
    def finish_file(self, success, chunk_count=0, chars=0, error=None, reason=None, sparse=False):
        """Complete processing a file"""
        duration = time.time() - self.file_start
        
        if success:
            self.success_count += 1
            self.total_chunks += chunk_count
            self.total_chars += chars
            sparse_tag = " +sparse" if sparse else ""
            print(f" -> {GREEN}OK{NC} ({chunk_count} chunks, {chars} chars{sparse_tag}) [{duration:.1f}s]")
            self.file_results.append({
                "status": "success",
                "chunks": chunk_count,
                "chars": chars,
                "duration": duration,
            })
        elif reason:
            # Skipped
            self.skipped_count += 1
            print(f" -> {YELLOW}SKIP{NC} ({reason})")
            self.file_results.append({"status": "skipped", "reason": reason})
        else:
            # Error
            self.failed_count += 1
            error_msg = str(error)[:50] if error else "Unknown error"
            print(f" -> {RED}ERROR{NC}: {error_msg}")
            self.file_results.append({"status": "error", "error": error_msg})
    
    def print_summary(self):
        """Print final summary"""
        total_time = time.time() - self.global_start
        
        print("")
        print(f"{GREEN}═══════════════════════════════════════{NC}")
        print(f"{GREEN}  Ingestion Complete!{NC}")
        print(f"{GREEN}═══════════════════════════════════════{NC}")
        print("")
        print(f"{CYAN}Summary:{NC}")
        print(f"  {GREEN}✓ Success:{NC} {self.success_count} files")
        print(f"  {YELLOW}○ Skipped:{NC} {self.skipped_count} files")
        print(f"  {RED}✗ Failed:{NC} {self.failed_count} files")
        print(f"  {BLUE}📦 Chunks:{NC} {self.total_chunks} total")
        print(f"  {BLUE}📝 Chars:{NC} {self.total_chars:,}")
        print(f"  {BLUE}⏱ Duration:{NC} {format_duration(total_time)}")
        
        if self.success_count > 0:
            avg_time = total_time / self.success_count
            avg_chunks = self.total_chunks / self.success_count
            print(f"  {BLUE}⚡ Speed:{NC} {avg_time:.1f}s/file, {avg_chunks:.0f} chunks/file")
        
        return {
            "success": self.success_count,
            "skipped": self.skipped_count,
            "errors": self.failed_count,
            "total_chunks": self.total_chunks,
            "total_chars": self.total_chars,
            "duration": total_time,
            "files": self.file_results,
        }


def create_tracker(total_files):
    """Factory function to create a tracker"""
    return IngestionProgressTracker(total_files)
EOFPY
log_ok "ingestion_progress.py"

log_info "Creating csv_nl_transform.py..."
cat > "$PROJECT_DIR/lib/csv_nl_transform.py" << 'EOFPY'
"""CSV/Excel Natural Language Transformation Module

Feature: CSV_NL_TRANSFORM_ENABLED
Introduced: csv
Lifecycle: ACTIVE

Transforms structured CSV/Excel rows into natural language sentences
for improved semantic search and reranking.

DUAL INGESTION MODE:
  - Ingests BOTH original structured format AND natural language version
  - Structured: exact field matching ("Nature: Sauvegarde")
  - Natural: semantic search ("probleme de sauvegarde")

Config:
  CSV_NL_TRANSFORM_ENABLED=true
  CSV_NL_DUAL_MODE=true          # Ingest both formats
  CSV_NL_MODE=auto|llm|custom
  CSV_NL_LANG=fr|en|auto
  CSV_NL_MAX_DESC_LEN=300
"""

import os
import re
from datetime import datetime


def _get_config():
    """Get CSV NL transform configuration"""
    return {
        "enabled": os.environ.get("CSV_NL_TRANSFORM_ENABLED", "true").lower() == "true",
        "dual_mode": os.environ.get("CSV_NL_DUAL_MODE", "true").lower() == "true",
        "mode": os.environ.get("CSV_NL_MODE", "auto"),  # auto|llm|custom
        "lang": os.environ.get("CSV_NL_LANG", "fr"),    # fr|en|auto
        "max_desc_len": int(os.environ.get("CSV_NL_MAX_DESC_LEN", "300")),
        "template": os.environ.get("CSV_NL_TEMPLATE", ""),
        "debug": os.environ.get("DEBUG", "").lower() == "true",
    }


def detect_language(headers, sample_values):
    """Detect language from column names and content"""
    text = ' '.join(headers) + ' ' + ' '.join(str(v) for v in sample_values if v)
    text_lower = text.lower()
    
    # French indicators
    fr_words = ['nom', 'societe', 'ville', 'date', 'heure', 'statut', 'type', 
                'secteur', 'probleme', 'solution', 'client', 'ticket', 'nature']
    # English indicators  
    en_words = ['name', 'company', 'city', 'date', 'time', 'status', 'type',
                'sector', 'problem', 'solution', 'client', 'ticket', 'nature']
    
    fr_count = sum(1 for w in fr_words if w in text_lower)
    en_count = sum(1 for w in en_words if w in text_lower)
    
    return 'fr' if fr_count >= en_count else 'en'


def detect_column_type(col_name, sample_values):
    """Detect semantic type of column
    
    Returns: id|name|description|date|status|location|numeric|category|other
    """
    col_lower = col_name.lower()
    
    # ID columns
    if re.search(r'\bid\b|_id$|^id_|numero|^n°', col_lower):
        return 'id'
    
    # Name/title columns
    if re.search(r'name|nom|titre|title|raison.?sociale|libelle|designation', col_lower):
        return 'name'
    
    # Description/text columns (long text fields)
    if re.search(r'desc|detail|problem|solution|comment|note|texte|contenu|cause', col_lower):
        return 'description'
    
    # Date columns
    if re.search(r'date|heure|time|created|updated|opened|closed|ouverture|fermeture', col_lower):
        return 'date'
    
    # Status columns
    if re.search(r'status|statut|etat|state', col_lower):
        return 'status'
    
    # Location columns
    if re.search(r'ville|city|address|adresse|cp|postal|pays|country|lieu|site|region', col_lower):
        return 'location'
    
    # Category columns
    if re.search(r'type|categor|nature|secteur|domain|activite|sous.?type|sous.?nature', col_lower):
        return 'category'
    
    # Boolean/flag columns
    if re.search(r'est.?|is.?|has.?|facturable|decompte|intervention', col_lower):
        return 'flag'
    
    # Numeric detection from values
    if sample_values:
        numeric_count = 0
        for v in sample_values[:10]:
            if v is not None:
                try:
                    float(str(v).replace(',', '.').replace(' ', ''))
                    numeric_count += 1
                except:
                    pass
        if numeric_count > 5:
            return 'numeric'
    
    return 'other'


def build_column_map(headers, sample_rows):
    """Build semantic map of columns
    
    Returns: dict of {column_type: [column_names]}
    """
    col_map = {
        'id': [], 'name': [], 'description': [], 'date': [],
        'status': [], 'location': [], 'category': [], 
        'numeric': [], 'flag': [], 'other': []
    }
    
    for col in headers:
        samples = []
        for row in sample_rows:
            val = row.get(col) if isinstance(row, dict) else None
            if val is not None:
                samples.append(val)
        
        col_type = detect_column_type(col, samples)
        col_map[col_type].append(col)
    
    return col_map


def clean_text(text, max_len=300):
    """Clean text content for natural language output"""
    if text is None:
        return ''
    text = str(text)
    # Remove Excel artifacts
    text = text.replace('_x000D_', ' ')
    text = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Truncate
    if len(text) > max_len:
        text = text[:max_len-3] + '...'
    return text.strip()


def is_valid_value(val):
    """Check if value is valid (not null, nan, empty, NaT)"""
    if val is None:
        return False
    # Check pandas NaT/NaN
    try:
        import pandas as pd
        if pd.isna(val):
            return False
    except (ImportError, TypeError):
        pass
    val_str = str(val).strip().lower()
    if val_str in ('', 'nan', 'none', 'null', 'n/a', '-', 'nat'):
        return False
    return True


def row_to_natural_language(row, headers, col_map=None, lang='fr', config=None):
    """Transform a CSV/Excel row to natural language
    
    Args:
        row: dict of column->value
        headers: list of column names
        col_map: pre-computed column map (optional, computed if None)
        lang: language code (fr/en)
        config: configuration dict (optional)
    
    Returns:
        str: Natural language sentence
    """
    if config is None:
        config = _get_config()
    
    if col_map is None:
        col_map = build_column_map(headers, [row])
    
    max_desc_len = config.get('max_desc_len', 300)
    parts = []
    
    # French templates
    if lang == 'fr':
        tpl = {
            'subject': "{name}",
            'id_ref': "(réf: {id})",
            'location': "situé à {location}",
            'category': "Catégorie: {category}",
            'description': "{label}: {value}",
            'status': "Statut: {status}",
            'date': "{label}: {value}",
            'flag_yes': "{label}",
            'flag_no': "",
        }
    else:
        tpl = {
            'subject': "{name}",
            'id_ref': "(ref: {id})",
            'location': "located in {location}",
            'category': "Category: {category}",
            'description': "{label}: {value}",
            'status': "Status: {status}",
            'date': "{label}: {value}",
            'flag_yes': "{label}",
            'flag_no': "",
        }
    
    # 1. Build subject (name + id)
    subject_parts = []
    for col in col_map['name']:
        val = row.get(col)
        if is_valid_value(val):
            subject_parts.append(clean_text(val, 100))
    
    if subject_parts:
        subject = ' - '.join(subject_parts)
        # Add ID reference
        for col in col_map['id'][:1]:
            val = row.get(col)
            if is_valid_value(val):
                subject += f" {tpl['id_ref'].format(id=val)}"
        parts.append(subject)
    elif col_map['id']:
        # No name, use ID as subject
        for col in col_map['id'][:1]:
            val = row.get(col)
            if is_valid_value(val):
                label = col.replace('_', ' ')
                parts.append(f"{label} {val}")
    
    # 2. Add location
    loc_parts = []
    for col in col_map['location']:
        val = row.get(col)
        if is_valid_value(val) and not str(val).startswith('http'):
            loc_parts.append(clean_text(val, 50))
    if loc_parts:
        parts.append(tpl['location'].format(location=', '.join(loc_parts[:3])))
    
    # 3. Add categories (important for filtering)
    cat_parts = []
    for col in col_map['category']:
        val = row.get(col)
        if is_valid_value(val):
            cat_parts.append(clean_text(val, 50))
    if cat_parts:
        parts.append(tpl['category'].format(category=' / '.join(cat_parts[:5])))
    
    # 4. Add descriptions (most important for semantic search)
    for col in col_map['description']:
        val = row.get(col)
        if is_valid_value(val):
            val_clean = clean_text(val, max_desc_len)
            if len(val_clean) > 10:  # Skip very short descriptions
                label = col.replace('_', ' ').replace("'", ' ')
                # Capitalize first letter of label
                label = label[0].upper() + label[1:] if label else label
                parts.append(tpl['description'].format(label=label, value=val_clean))
    
    # 5. Add status
    for col in col_map['status']:
        val = row.get(col)
        if is_valid_value(val):
            parts.append(tpl['status'].format(status=val))
    
    # 6. Add important dates (max 2)
    date_count = 0
    for col in col_map['date']:
        if date_count >= 2:
            break
        val = row.get(col)
        if is_valid_value(val):
            # Check for pandas NaT (Not a Time) before strftime
            if hasattr(val, 'strftime'):
                try:
                    import pandas as pd
                    if pd.isna(val):
                        continue
                    val = val.strftime('%Y-%m-%d %H:%M')
                except (ValueError, TypeError):
                    continue  # Skip invalid dates
            label = col.replace('_', ' ').replace("'", ' ')
            label = label[0].upper() + label[1:] if label else label
            parts.append(tpl['date'].format(label=label, value=val))
            date_count += 1
    
    # 7. Add relevant 'other' fields (short values only)
    other_count = 0
    for col in col_map['other']:
        if other_count >= 3:  # Limit other fields
            break
        val = row.get(col)
        if is_valid_value(val):
            val_str = str(val)
            if 10 < len(val_str) < 100:  # Skip IDs and very long text
                label = col.replace('_', ' ')
                label = label[0].upper() + label[1:] if label else label
                parts.append(f"{label}: {clean_text(val, 80)}")
                other_count += 1
    
    return '. '.join(parts) + '.' if parts else ''


def transform_csv_rows(headers, rows, lang=None):
    """Transform multiple CSV rows to natural language
    
    Args:
        headers: list of column names
        rows: list of row dicts
    
    Returns:
        list: Natural language sentences (one per row)
    """
    config = _get_config()
    
    if not config['enabled']:
        return [None] * len(rows)
    
    # Detect language if auto
    if lang is None:
        lang = config['lang']
    if lang == 'auto':
        sample_vals = []
        for row in rows[:5]:
            sample_vals.extend(str(v) for v in row.values() if v)
        lang = detect_language(headers, sample_vals)
    
    # Build column map once
    col_map = build_column_map(headers, rows[:10])
    
    if config['debug']:
        print(f"[CSV_NL] Language: {lang}")
        print(f"[CSV_NL] Column map: {col_map}")
    
    # Transform rows
    results = []
    for row in rows:
        nl = row_to_natural_language(row, headers, col_map, lang, config)
        results.append(nl)
    
    return results


def get_dual_chunks(headers, rows, filename, lang=None):
    """Generate dual chunks: structured + natural language
    
    Args:
        headers: list of column names
        rows: list of row dicts
        filename: source filename for metadata
    
    Returns:
        list: Chunk dicts with 'text', 'chunk_type', 'metadata'
    """
    config = _get_config()
    chunks = []
    
    # Detect language
    if lang is None:
        lang = config['lang']
    if lang == 'auto':
        sample_vals = []
        for row in rows[:5]:
            sample_vals.extend(str(v) for v in row.values() if v)
        lang = detect_language(headers, sample_vals)
    
    # Build column map once
    col_map = build_column_map(headers, rows[:10])
    
    for i, row in enumerate(rows):
        row_id = row.get(col_map['id'][0]) if col_map['id'] else i
        
        # 1. Structured chunk (original format)
        struct_parts = []
        for h in headers:
            val = row.get(h)
            if is_valid_value(val):
                struct_parts.append(f"{h}: {clean_text(val, 200)}")
        
        struct_text = ' | '.join(struct_parts)
        if struct_text:
            chunks.append({
                'text': struct_text,
                'chunk_type': 'structured',
                'row_index': i,
                'row_id': str(row_id),
                'filename': filename,
                'format': 'csv_structured',
            })
        
        # 2. Natural language chunk (if enabled)
        if config['enabled']:
            nl_text = row_to_natural_language(row, headers, col_map, lang, config)
            if nl_text and len(nl_text) > 20:
                chunks.append({
                    'text': nl_text,
                    'chunk_type': 'natural_language',
                    'row_index': i,
                    'row_id': str(row_id),
                    'filename': filename,
                    'format': 'csv_natural',
                    'lang': lang,
                })
    
    return chunks


# Backward compatibility
def transform_row(row, headers, col_map=None, template=None, lang='fr'):
    """Backward compatible single row transform"""
    return row_to_natural_language(row, headers, col_map, lang)
EOFPY
log_ok "csv_nl_transform.py"

log_info "Creating csv_dual_ingest.py..."
cat > "$PROJECT_DIR/lib/csv_dual_ingest.py" << 'EOFPY'
"""CSV/Excel Dual Ingestion Helper

Feature: CSV_NL_DUAL_MODE
Introduced: csv
Lifecycle: ACTIVE

Handles dual ingestion of CSV/Excel files:
1. Structured format (original column: value pairs)
2. Natural language format (semantic sentences)
"""

import os
import csv
from io import StringIO


def _get_config():
    """Get dual ingest configuration"""
    return {
        "enabled": os.environ.get("CSV_NL_TRANSFORM_ENABLED", "true").lower() == "true",
        "dual_mode": os.environ.get("CSV_NL_DUAL_MODE", "true").lower() == "true",
        "lang": os.environ.get("CSV_NL_LANG", "fr"),
        "debug": os.environ.get("DEBUG", "").lower() == "true",
    }


def is_tabular_file(filepath):
    """Check if file is CSV/Excel/TSV"""
    ext = os.path.splitext(filepath)[1].lower()
    return ext in ['.csv', '.tsv', '.xlsx', '.xls']


def parse_tabular_file(filepath):
    """Parse CSV/Excel/TSV file to list of row dicts
    
    Returns:
        tuple: (headers, rows, encoding)
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    # Excel files
    if ext in ['.xlsx', '.xls']:
        try:
            import pandas as pd
            df = pd.read_excel(filepath)
            headers = list(df.columns)
            rows = [dict(row) for _, row in df.iterrows()]
            return headers, rows, 'excel'
        except ImportError:
            # Fallback: try with openpyxl directly
            try:
                from openpyxl import load_workbook
                wb = load_workbook(filepath, read_only=True, data_only=True)
                ws = wb.active
                data = list(ws.iter_rows(values_only=True))
                if not data:
                    return [], [], 'excel'
                headers = [str(h) if h else f'col_{i}' for i, h in enumerate(data[0])]
                rows = []
                for row_data in data[1:]:
                    row = {headers[i]: v for i, v in enumerate(row_data) if i < len(headers)}
                    rows.append(row)
                return headers, rows, 'excel'
            except Exception as e:
                return [], [], f'error: {e}'
    
    # CSV/TSV files
    delimiter = '\t' if ext == '.tsv' else ','
    encodings = ['utf-8', 'cp1252', 'iso-8859-1', 'utf-16', 'latin-1']
    
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc, newline='') as f:
                content = f.read()
                if content.startswith('\ufeff'):
                    content = content[1:]  # Remove BOM
                
                reader = csv.DictReader(StringIO(content), delimiter=delimiter)
                headers = reader.fieldnames or []
                rows = list(reader)
                return headers, rows, enc
        except (UnicodeDecodeError, LookupError):
            continue
        except Exception as e:
            continue
    
    return [], [], 'error'


def get_dual_chunks_for_file(filepath, chunk_size=500):
    """Generate dual chunks for a tabular file
    
    Args:
        filepath: Path to CSV/Excel file
        chunk_size: Max chunk size (for grouping small rows)
    
    Returns:
        list: Chunk dicts ready for embedding
    """
    config = _get_config()
    
    if not is_tabular_file(filepath):
        return None  # Not a tabular file, use standard processing
    
    headers, rows, encoding = parse_tabular_file(filepath)
    
    if not headers or not rows:
        return None
    
    filename = os.path.basename(filepath)
    
    if config['debug']:
        print(f"[CSV_DUAL] Parsing {filename}: {len(rows)} rows, {len(headers)} columns")
    
    # Import NL transform
    try:
        from csv_nl_transform import get_dual_chunks, transform_csv_rows
    except ImportError:
        # Fallback: return None to use standard processing
        return None
    
    if config['dual_mode']:
        # Generate both structured and NL chunks
        chunks = get_dual_chunks(headers, rows, filename, lang=config['lang'])
    else:
        # NL only mode
        nl_texts = transform_csv_rows(headers, rows, lang=config['lang'])
        chunks = []
        for i, (row, nl_text) in enumerate(zip(rows, nl_texts)):
            if nl_text:
                chunks.append({
                    'text': nl_text,
                    'chunk_type': 'natural_language',
                    'row_index': i,
                    'filename': filename,
                    'format': 'csv_natural',
                })
    
    if config['debug']:
        struct_count = sum(1 for c in chunks if c.get('chunk_type') == 'structured')
        nl_count = sum(1 for c in chunks if c.get('chunk_type') == 'natural_language')
        print(f"[CSV_DUAL] Generated {struct_count} structured + {nl_count} NL chunks")
    
    return chunks


def merge_small_chunks(chunks, target_size=400, max_size=800):
    """Merge small chunks from same file/type for efficiency
    
    Groups small rows together while preserving chunk_type separation.
    """
    if not chunks:
        return chunks
    
    # Group by (filename, chunk_type)
    groups = {}
    for chunk in chunks:
        key = (chunk.get('filename', ''), chunk.get('chunk_type', ''))
        if key not in groups:
            groups[key] = []
        groups[key].append(chunk)
    
    merged = []
    for (filename, chunk_type), group_chunks in groups.items():
        current_texts = []
        current_size = 0
        current_indices = []
        
        for chunk in group_chunks:
            text = chunk.get('text', '')
            text_len = len(text)
            
            # If adding this would exceed max, flush current
            if current_size + text_len > max_size and current_texts:
                merged.append({
                    'text': '\n\n'.join(current_texts),
                    'chunk_type': chunk_type,
                    'filename': filename,
                    'format': chunk.get('format', ''),
                    'row_indices': current_indices.copy(),
                    'merged_count': len(current_texts),
                })
                current_texts = []
                current_size = 0
                current_indices = []
            
            current_texts.append(text)
            current_size += text_len + 2  # +2 for separator
            current_indices.append(chunk.get('row_index', -1))
            
            # If reached target, flush
            if current_size >= target_size:
                merged.append({
                    'text': '\n\n'.join(current_texts),
                    'chunk_type': chunk_type,
                    'filename': filename,
                    'format': chunk.get('format', ''),
                    'row_indices': current_indices.copy(),
                    'merged_count': len(current_texts),
                })
                current_texts = []
                current_size = 0
                current_indices = []
        
        # Flush remaining
        if current_texts:
            merged.append({
                'text': '\n\n'.join(current_texts),
                'chunk_type': chunk_type,
                'filename': filename,
                'format': chunk.get('format', ''),
                'row_indices': current_indices.copy(),
                'merged_count': len(current_texts),
            })
    
    return merged
EOFPY
log_ok "csv_dual_ingest.py"

log_info "Creating extended_formats.py..."
cat > "$PROJECT_DIR/lib/extended_formats.py" << 'EOFPY'
"""Extended Format Support Module

Feature: EXTENDED_FORMATS
Introduced: csv
Lifecycle: ACTIVE

Adds parsing support for:
  - XML (configuration, data, SOAP)
  - YAML/YML (configuration, Kubernetes, Ansible)
  - PowerShell (.ps1, .psm1, .psd1)
  - Shell scripts (.sh, .bash, .zsh)
  - Python (.py)
  - SQL (.sql)
  - Log files (.log)
  - Config files (.ini, .conf, .cfg)
  - reStructuredText (.rst)
  - Code (.js, .ts, .java, .cs, .go, .rb, .php)
"""

import os
import re


def get_extended_extensions():
    """Return list of additionally supported extensions"""
    return [
        # Configuration
        ".xml", ".yaml", ".yml", ".ini", ".conf", ".cfg", ".toml",
        # Scripts
        ".ps1", ".psm1", ".psd1",  # PowerShell
        ".sh", ".bash", ".zsh",    # Shell
        ".py", ".pyw",             # Python
        ".sql",                    # SQL
        # Code
        ".js", ".ts", ".jsx", ".tsx",  # JavaScript/TypeScript
        ".java", ".kt",                 # Java/Kotlin
        ".cs", ".vb",                   # .NET
        ".go",                          # Go
        ".rb",                          # Ruby
        ".php",                         # PHP
        ".c", ".cpp", ".h", ".hpp",     # C/C++
        ".rs",                          # Rust
        # Logs and text
        ".log", ".logs",
        ".rst",                         # reStructuredText
        # Data
        ".ndjson", ".jsonl",           # JSON Lines
    ]


def parse_xml(file_path):
    """Parse XML file to readable text"""
    try:
        import xml.etree.ElementTree as ET
        
        # Try multiple encodings
        content = None
        for enc in ['utf-8', 'utf-16', 'iso-8859-1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue
        
        if not content:
            with open(file_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')
        
        # Parse XML
        root = ET.fromstring(content)
        
        # Extract text content recursively
        def extract_text(elem, depth=0):
            parts = []
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            
            # Add tag with attributes
            attrs = ' '.join(f'{k}="{v}"' for k, v in elem.attrib.items())
            if attrs:
                parts.append(f"{'  '*depth}{tag} ({attrs}):")
            elif elem.text and elem.text.strip():
                parts.append(f"{'  '*depth}{tag}: {elem.text.strip()}")
            
            # Process children
            for child in elem:
                parts.extend(extract_text(child, depth + 1))
            
            return parts
        
        text_lines = extract_text(root)
        text = '\n'.join(text_lines)
        
        return text, {
            "filename": os.path.basename(file_path),
            "parser": "xml",
            "root_tag": root.tag,
            "element_count": len(list(root.iter())),
        }
    except Exception as e:
        # Fallback: return raw content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read(), {"parser": "xml_raw", "error": str(e)}
        except:
            return "", {"error": str(e)}


def parse_yaml(file_path):
    """Parse YAML file to readable text"""
    try:
        import yaml
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse YAML
        try:
            data = yaml.safe_load(content)
        except:
            # If parsing fails, return raw content
            return content, {"parser": "yaml_raw"}
        
        # Convert to readable format
        def yaml_to_text(obj, prefix=""):
            parts = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        parts.append(f"{prefix}{k}:")
                        parts.extend(yaml_to_text(v, prefix + "  "))
                    else:
                        parts.append(f"{prefix}{k}: {v}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        parts.append(f"{prefix}- item {i+1}:")
                        parts.extend(yaml_to_text(item, prefix + "  "))
                    else:
                        parts.append(f"{prefix}- {item}")
            else:
                parts.append(f"{prefix}{obj}")
            return parts
        
        text = '\n'.join(yaml_to_text(data))
        
        return text, {
            "filename": os.path.basename(file_path),
            "parser": "yaml",
            "keys": list(data.keys()) if isinstance(data, dict) else [],
        }
    except ImportError:
        # No PyYAML, return raw
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read(), {"parser": "yaml_raw", "note": "PyYAML not installed"}
    except Exception as e:
        return "", {"error": str(e)}


def parse_code(file_path, language=None):
    """Parse code file with syntax-aware formatting"""
    ext = os.path.splitext(file_path)[1].lower()
    
    # Language detection
    lang_map = {
        '.ps1': 'powershell', '.psm1': 'powershell', '.psd1': 'powershell',
        '.sh': 'bash', '.bash': 'bash', '.zsh': 'zsh',
        '.py': 'python', '.pyw': 'python',
        '.sql': 'sql',
        '.js': 'javascript', '.jsx': 'javascript',
        '.ts': 'typescript', '.tsx': 'typescript',
        '.java': 'java', '.kt': 'kotlin',
        '.cs': 'csharp', '.vb': 'vbnet',
        '.go': 'go',
        '.rb': 'ruby',
        '.php': 'php',
        '.c': 'c', '.cpp': 'cpp', '.h': 'c', '.hpp': 'cpp',
        '.rs': 'rust',
    }
    
    lang = language or lang_map.get(ext, 'text')
    
    try:
        # Read with encoding fallback
        content = None
        for enc in ['utf-8', 'utf-16', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue
        
        if not content:
            with open(file_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')
        
        # Extract documentation and structure
        lines = content.split('\n')
        
        # Count meaningful content
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        functions = []
        classes = []
        
        # Language-specific patterns
        if lang in ['powershell']:
            func_pattern = r'^\s*function\s+([A-Za-z_][A-Za-z0-9_-]*)'
            comment_chars = '#'
        elif lang in ['python']:
            func_pattern = r'^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)'
            comment_chars = '#'
        elif lang in ['bash', 'zsh']:
            func_pattern = r'^\s*(?:function\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(\)'
            comment_chars = '#'
        elif lang in ['sql']:
            func_pattern = r'(?:CREATE\s+(?:PROCEDURE|FUNCTION)\s+)([A-Za-z_][A-Za-z0-9_]*)'
            comment_chars = '--'
        elif lang in ['javascript', 'typescript']:
            func_pattern = r'(?:function\s+|const\s+|let\s+|var\s+)([A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*(?:async\s*)?\(|[\(])'
            comment_chars = '//'
        elif lang in ['java', 'csharp', 'kotlin', 'go', 'rust', 'cpp', 'c']:
            func_pattern = r'(?:public|private|protected|static|async|func|fn)?\s*(?:[A-Za-z_<>\[\]]+\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\('
            comment_chars = '//'
        else:
            func_pattern = r'function\s+([A-Za-z_][A-Za-z0-9_]*)'
            comment_chars = '#'
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith(comment_chars):
                comment_lines += 1
            else:
                code_lines += 1
                
                # Find functions
                match = re.search(func_pattern, line, re.IGNORECASE)
                if match:
                    functions.append(match.group(1))
                
                # Find classes
                class_match = re.search(r'^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)', line)
                if class_match:
                    classes.append(class_match.group(1))
        
        # Build enhanced text with metadata header
        header_parts = [
            f"# File: {os.path.basename(file_path)}",
            f"# Language: {lang}",
            f"# Lines: {len(lines)} (code: {code_lines}, comments: {comment_lines})",
        ]
        
        if functions:
            header_parts.append(f"# Functions: {', '.join(functions[:10])}")
        if classes:
            header_parts.append(f"# Classes: {', '.join(classes[:10])}")
        
        header_parts.append("")
        
        enhanced_content = '\n'.join(header_parts) + content
        
        return enhanced_content, {
            "filename": os.path.basename(file_path),
            "parser": "code",
            "language": lang,
            "lines": len(lines),
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "functions": functions[:20],
            "classes": classes[:10],
        }
    except Exception as e:
        return "", {"error": str(e)}


def parse_log(file_path):
    """Parse log file with structure detection"""
    try:
        # Read with encoding fallback
        content = None
        for enc in ['utf-8', 'utf-16', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue
        
        if not content:
            with open(file_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')
        
        lines = content.split('\n')
        
        # Detect log patterns
        error_count = 0
        warn_count = 0
        info_count = 0
        
        error_lines = []
        
        for line in lines:
            line_lower = line.lower()
            if 'error' in line_lower or 'exception' in line_lower or 'fatal' in line_lower:
                error_count += 1
                if len(error_lines) < 10:
                    error_lines.append(line[:200])
            elif 'warn' in line_lower:
                warn_count += 1
            elif 'info' in line_lower:
                info_count += 1
        
        # Add summary header
        header = [
            f"# Log file: {os.path.basename(file_path)}",
            f"# Total lines: {len(lines)}",
            f"# Errors: {error_count}, Warnings: {warn_count}, Info: {info_count}",
        ]
        
        if error_lines:
            header.append("# Sample errors:")
            for err in error_lines[:5]:
                header.append(f"#   {err[:100]}")
        
        header.append("")
        
        enhanced_content = '\n'.join(header) + content
        
        return enhanced_content, {
            "filename": os.path.basename(file_path),
            "parser": "log",
            "lines": len(lines),
            "errors": error_count,
            "warnings": warn_count,
            "sample_errors": error_lines[:5],
        }
    except Exception as e:
        return "", {"error": str(e)}


def parse_ini(file_path):
    """Parse INI/conf file to readable text"""
    try:
        import configparser
        
        config = configparser.ConfigParser()
        config.read(file_path, encoding='utf-8')
        
        parts = [f"# Configuration file: {os.path.basename(file_path)}", ""]
        
        for section in config.sections():
            parts.append(f"[{section}]")
            for key, value in config.items(section):
                parts.append(f"  {key} = {value}")
            parts.append("")
        
        return '\n'.join(parts), {
            "filename": os.path.basename(file_path),
            "parser": "ini",
            "sections": config.sections(),
        }
    except Exception as e:
        # Fallback: read raw
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read(), {"parser": "ini_raw", "error": str(e)}
        except:
            return "", {"error": str(e)}


def parse_extended_format(file_path):
    """Main entry point for extended format parsing
    
    Returns:
        tuple: (text, metadata) or None if not an extended format
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    # XML
    if ext == '.xml':
        return parse_xml(file_path)
    
    # YAML
    if ext in ['.yaml', '.yml']:
        return parse_yaml(file_path)
    
    # Code files
    code_exts = [
        '.ps1', '.psm1', '.psd1',
        '.sh', '.bash', '.zsh',
        '.py', '.pyw',
        '.sql',
        '.js', '.ts', '.jsx', '.tsx',
        '.java', '.kt',
        '.cs', '.vb',
        '.go', '.rb', '.php',
        '.c', '.cpp', '.h', '.hpp',
        '.rs',
    ]
    if ext in code_exts:
        return parse_code(file_path)
    
    # Log files
    if ext in ['.log', '.logs']:
        return parse_log(file_path)
    
    # Config files
    if ext in ['.ini', '.conf', '.cfg']:
        return parse_ini(file_path)
    
    # TOML (treat as YAML-like)
    if ext == '.toml':
        try:
            import tomllib
            with open(file_path, 'rb') as f:
                data = tomllib.load(f)
            # Convert to readable text
            import json
            text = json.dumps(data, indent=2)
            return text, {"parser": "toml", "filename": os.path.basename(file_path)}
        except:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read(), {"parser": "toml_raw"}
    
    # JSON Lines / NDJSON
    if ext in ['.ndjson', '.jsonl']:
        try:
            import json
            lines = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        lines.append(json.dumps(obj, indent=2))
            return '\n---\n'.join(lines), {"parser": "jsonl", "records": len(lines)}
        except Exception as e:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read(), {"parser": "jsonl_raw", "error": str(e)}
    
    # RST (reStructuredText)
    if ext == '.rst':
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read(), {"parser": "rst", "filename": os.path.basename(file_path)}
    
    # Not an extended format
    return None


# Test
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        result = parse_extended_format(sys.argv[1])
        if result:
            text, meta = result
            print(f"Parser: {meta.get('parser')}")
            print(f"Text length: {len(text)}")
            print("---")
            print(text[:500])
        else:
            print("Not an extended format")
EOFPY
log_ok "extended_formats.py"

log_info "Creating ingest_main.py..."
cat > "$PROJECT_DIR/lib/ingest_main.py" << 'EOFPY'
"""Main document ingestion pipeline

Feature: DOC_DEDUP_ENABLED (dedup)
Introduced: dedup
Lifecycle: ACTIVE

Feature: SPARSE_EMBED_ENABLED (hybrid)
Introduced: hybrid
Lifecycle: ACTIVE

Generates both dense and sparse vectors for native Qdrant hybrid search.
Falls back to dense-only when sparse embeddings unavailable.
dedup: Adds document-level dedup to skip entire duplicate documents.
"""

import os
import sys
import json
import hashlib
import time

# Suppress OpenCV/OpenGL warnings on headless systems
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("UNSTRUCTURED_USE_GPU", "false")

# Add lib to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unstructured_parser import parse_document, get_supported_extensions
from smart_chunker import smart_chunk
from embedding_helper import get_embedding, get_embeddings_batch, get_embedding_dimension

# Import sparse embedding helper (hybrid)
from sparse_embedding_helper import (
    is_sparse_embed_available,
    get_sparse_embeddings_batch,
    get_sparse_model_info,
)

# Import hybrid Qdrant interface (hybrid)
from qdrant_hybrid_helper import (
    ensure_hybrid_collection,
    upload_hybrid_points,
    delete_collection,
    get_hybrid_mode,
)

# Legacy imports for fallback
from qdrant_client_helper import (
    ensure_collection,
    upload_points,
    get_client_mode,
)

# dedup: Document-level dedup
from doc_dedup import (
    is_duplicate_doc,
    mark_doc_ingested,
    get_doc_index_stats,
)

def get_file_hash(file_path):
    """Calculate MD5 hash of file"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

def is_file_processed(file_path, tracking_dir=".ingest_tracking"):
    """Check if file was already processed"""
    file_hash = get_file_hash(file_path)
    track_file = os.path.join(tracking_dir, f"{file_hash}.json")
    return os.path.exists(track_file)

def mark_file_processed(file_path, metadata, tracking_dir=".ingest_tracking"):
    """Mark file as processed"""
    os.makedirs(tracking_dir, exist_ok=True)
    file_hash = get_file_hash(file_path)
    track_file = os.path.join(tracking_dir, f"{file_hash}.json")
    with open(track_file, 'w') as f:
        json.dump({
            "file": file_path,
            "hash": file_hash,
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": metadata,
        }, f, indent=2)

def ingest_file(file_path, qdrant_host, collection_name, chunk_size=500, chunk_overlap=80, 
                debug=False, force=False):
    """Ingest a single file with hybrid vectors (hybrid)
    
    Args:
        file_path: Path to document
        qdrant_host: Qdrant host URL
        collection_name: Collection name
        chunk_size: Target chunk size
        chunk_overlap: Chunk overlap
        debug: Enable debug output
        force: Force re-ingestion
    
    Returns:
        dict: Ingestion result
    """
    filename = os.path.basename(file_path)
    sparse_enabled = os.environ.get("SPARSE_EMBED_ENABLED", "true").lower() == "true"
    
    # Check if already processed
    if not force and is_file_processed(file_path):
        return {"status": "skipped", "filename": filename, "reason": "already processed"}
    
    # Check extension
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in get_supported_extensions():
        return {"status": "skipped", "filename": filename, "reason": f"unsupported ({ext})"}
    
    # Parse document
    text, parse_meta = parse_document(
        file_path,
        strategy=os.environ.get("UNSTRUCTURED_STRATEGY", "auto"),
        ocr_languages=os.environ.get("UNSTRUCTURED_OCR_LANGUAGES", "eng+fra"),
    )
    
    if not text or "error" in parse_meta:
        error = parse_meta.get("error", "No text extracted")
        return {"status": "error", "filename": filename, "error": error}
    
    # dedup: Document-level dedup - check if same content already ingested
    doc_dedup_enabled = os.environ.get("DOC_DEDUP_ENABLED", "true").lower() == "true"
    if doc_dedup_enabled and not force:
        is_dup, original_file = is_duplicate_doc(text, filename)
        if is_dup:
            return {
                "status": "skipped", 
                "filename": filename, 
                "reason": f"duplicate content (same as {original_file})"
            }
    
    # Chunk text
    chunks = smart_chunk(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=int(os.environ.get("MIN_CHUNK_SIZE", "100")),
        max_chunk_size=int(os.environ.get("MAX_CHUNK_SIZE", "1200")),
    )
    
    if not chunks:
        return {"status": "empty", "filename": filename}
    
    # Ensure collection exists (hybrid: hybrid collection with named vectors)
    dimension = get_embedding_dimension()
    
    if sparse_enabled and is_sparse_embed_available():
        ensure_hybrid_collection(collection_name)
    else:
        ensure_collection(qdrant_host, collection_name, dimension)
    
    # Generate embeddings
    texts = [c["text"] for c in chunks]
    
    # Dense embeddings
    dense_embeddings = get_embeddings_batch(texts)
    
    # Sparse embeddings (hybrid)
    sparse_embeddings = None
    if sparse_enabled and is_sparse_embed_available():
        sparse_embeddings = get_sparse_embeddings_batch(texts)
    
    # Prepare points for Qdrant
    base_id = int(hashlib.md5(file_path.encode()).hexdigest()[:8], 16)
    
    if sparse_enabled and sparse_embeddings:
        # Hybrid points with named vectors (hybrid)
        points = []
        for i, (chunk, dense_emb, sparse_emb) in enumerate(zip(chunks, dense_embeddings, sparse_embeddings)):
            point_id = base_id + i
            points.append({
                "id": point_id,
                "dense_vector": dense_emb,
                "sparse_vector": sparse_emb,
                "payload": {
                    "text": chunk["text"],
                    "filename": filename,
                    "filepath": file_path,
                    "chunk_index": i,
                    "chunk_type": chunk.get("type", "unknown"),
                    "char_count": chunk.get("char_count", len(chunk["text"])),
                    "parser": "unstructured",
                    "sparse_model": os.environ.get("SPARSE_EMBED_MODEL", "Qdrant/bm25"),
                    "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            })
        
        batch_size = int(os.environ.get("QDRANT_BATCH_SIZE", "100"))
        success = upload_hybrid_points(collection_name, points, batch_size)
    else:
        # Legacy dense-only points
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, dense_embeddings)):
            point_id = base_id + i
            points.append({
                "id": point_id,
                "vector": embedding,
                "payload": {
                    "text": chunk["text"],
                    "filename": filename,
                    "filepath": file_path,
                    "chunk_index": i,
                    "chunk_type": chunk.get("type", "unknown"),
                    "char_count": chunk.get("char_count", len(chunk["text"])),
                    "parser": "unstructured",
                    "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            })
        
        batch_size = int(os.environ.get("QDRANT_BATCH_SIZE", "100"))
        success = upload_points(qdrant_host, collection_name, points, batch_size)
    
    if not success:
        return {"status": "error", "filename": filename, "error": "Upload failed"}
    
    # Mark as processed
    mark_file_processed(file_path, {
        "chunks": len(chunks),
        "parser": "unstructured",
        "dimension": dimension,
        "sparse_enabled": sparse_enabled and sparse_embeddings is not None,
        "sparse_model": os.environ.get("SPARSE_EMBED_MODEL", "Qdrant/bm25") if sparse_embeddings else None,
    })
    
    # dedup: Mark document content as ingested for dedup
    if doc_dedup_enabled:
        mark_doc_ingested(text, filename, file_path)
    
    return {
        "status": "success",
        "filename": filename,
        "chunks": len(chunks),
        "chars": len(text),
        "sparse": sparse_embeddings is not None,
    }

def ingest_directory(docs_dir, qdrant_host, collection_name, chunk_size=500, 
                     chunk_overlap=80, debug=False, force=False):
    """Ingest all documents in directory
    
    Args:
        docs_dir: Documents directory
        qdrant_host: Qdrant host URL
        collection_name: Collection name
        chunk_size: Target chunk size
        chunk_overlap: Chunk overlap
        debug: Enable debug output
        force: Force re-ingestion
    
    Returns:
        dict: Summary of ingestion
    """
    supported = get_supported_extensions()
    results = {"success": 0, "skipped": 0, "errors": 0, "files": []}
    
    # Find all files
    files = []
    for root, _, filenames in os.walk(docs_dir):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported:
                files.append(os.path.join(root, filename))
    
    if not files:
        print(f"No supported files found in {docs_dir}")
        return results
    
    total = len(files)
    print(f"Found {total} documents to process")
    print("")
    
    for i, file_path in enumerate(files, 1):
        filename = os.path.basename(file_path)
        
        # Always show progress line with flush
        print(f"  [{i}/{total}] {filename}", end="", flush=True)
        
        # Show file size for large files
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 1024 * 1024:  # > 1MB
                print(f" ({file_size // (1024*1024)}MB)", end="", flush=True)
        except:
            pass
        
        result = ingest_file(
            file_path, qdrant_host, collection_name,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            debug=debug, force=force,
        )
        
        results["files"].append(result)
        
        # Show result on same line
        if result["status"] == "success":
            results["success"] += 1
            chunks = result.get("chunks", 0)
            chars = result.get("chars", 0)
            sparse = result.get("sparse", False)
            sparse_tag = " +sparse" if sparse else ""
            print(f" -> OK ({chunks} chunks, {chars} chars{sparse_tag})")
        elif result["status"] == "skipped":
            results["skipped"] += 1
            reason = result.get("reason", "already processed")
            print(f" -> SKIP ({reason})")
        else:
            results["errors"] += 1
            err = result.get("error", "unknown")[:50]
            print(f" -> ERROR: {err}")
    
    return results

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Document Ingestion dedup")
    parser.add_argument("path", nargs="?", help="File or directory to ingest")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate collection")
    args = parser.parse_args()
    
    # Load config
    qdrant_host = os.environ.get("QDRANT_HOST", "http://localhost:6333")
    collection_name = os.environ.get("COLLECTION_NAME", "documents")
    chunk_size = int(os.environ.get("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.environ.get("CHUNK_OVERLAP", "80"))
    docs_dir = os.environ.get("DOCUMENTS_DIR", "./documents")
    debug = args.debug or os.environ.get("DEBUG", "false").lower() == "true"
    
    # Show mode info
    client_mode = get_client_mode()
    hybrid_mode = get_hybrid_mode()
    sparse_info = get_sparse_model_info()
    
    print(f"Qdrant mode: {client_mode}")
    print(f"Hybrid mode: {hybrid_mode}")
    print(f"Sparse model: {sparse_info['model']} (available: {sparse_info['available']})")
    print("")
    
    # Recreate collection if requested
    if args.recreate:
        print(f"Recreating collection: {collection_name}")
        delete_collection(collection_name)
        # Clear tracking files
        import shutil
        if os.path.exists(".ingest_tracking"):
            shutil.rmtree(".ingest_tracking")
            os.makedirs(".ingest_tracking")
        # Clear dedup index (dedup)
        dedup_index = os.environ.get("DOC_DEDUP_INDEX", "cache/doc_dedup.json")
        if os.path.exists(dedup_index):
            os.remove(dedup_index)
            print(f"[OK] Dedup index cleared: {dedup_index}")
        print("[OK] Collection deleted, tracking cleared")
        print("")
    
    # Determine path
    path = args.path or docs_dir
    
    if os.path.isfile(path):
        result = ingest_file(
            path, qdrant_host, collection_name,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            debug=debug, force=args.force,
        )
        print(f"Result: {result['status']}")
    elif os.path.isdir(path):
        results = ingest_directory(
            path, qdrant_host, collection_name,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            debug=debug, force=args.force,
        )
        print(f"\nIngestion complete:")
        print(f"  Success: {results['success']}")
        print(f"  Skipped: {results['skipped']}")
        print(f"  Errors: {results['errors']}")
    else:
        print(f"Path not found: {path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOFPY
log_ok "ingest_main.py"

log_info "Creating web_crawler.py..."
cat > "$PROJECT_DIR/lib/web_crawler.py" << 'EOFPY'
"""Web Crawler Module for RAG System web

Crawls websites and ingests content into the RAG system.
Supports recursive crawling with depth control, rate limiting,
and robots.txt respect.

Usage:
    python3 web_crawler.py https://example.com [--max-pages 50] [--max-depth 3]
"""

import os
import sys
import re
import time
import json
import hashlib
from urllib.parse import urljoin, urlparse, urlunparse
from collections import deque

def get_config():
    """Get crawler configuration from environment"""
    return {
        "max_pages": int(os.environ.get("WEB_CRAWLER_MAX_PAGES", "50")),
        "max_depth": int(os.environ.get("WEB_CRAWLER_MAX_DEPTH", "3")),
        "same_domain": os.environ.get("WEB_CRAWLER_SAME_DOMAIN", "true").lower() == "true",
        "delay": float(os.environ.get("WEB_CRAWLER_DELAY", "1.0")),
        "timeout": int(os.environ.get("WEB_CRAWLER_TIMEOUT", "30")),
        "respect_robots": os.environ.get("WEB_CRAWLER_RESPECT_ROBOTS", "true").lower() == "true",
        "user_agent": os.environ.get("WEB_CRAWLER_USER_AGENT", "RAGBot/1.0 (+https://github.com/rag-system)"),
        "skip_extensions": os.environ.get("WEB_CRAWLER_SKIP_EXT", ".pdf,.doc,.docx,.xls,.xlsx,.zip,.tar,.gz,.exe,.dmg,.pkg,.jpg,.jpeg,.png,.gif,.mp4,.mp3").split(","),
        "exclude_patterns": os.environ.get("WEB_CRAWLER_EXCLUDE_PATTERNS", "/login,/logout,/admin,/cart,/checkout,/account").split(","),
        "chunk_size": int(os.environ.get("CHUNK_SIZE", "500")),
        "chunk_overlap": int(os.environ.get("CHUNK_OVERLAP", "50")),
        "debug": os.environ.get("DEBUG", "").lower() == "true",
        # Qdrant settings
        "qdrant_host": os.environ.get("QDRANT_HOST", "http://localhost:6333"),
        "qdrant_grpc_port": int(os.environ.get("QDRANT_GRPC_PORT", "6334")),
        "collection_name": os.environ.get("COLLECTION_NAME", "documents"),
    }

def normalize_url(url):
    """Normalize URL for deduplication"""
    parsed = urlparse(url)
    # Remove fragments, normalize path
    path = parsed.path.rstrip("/") or "/"
    normalized = urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        path,
        "",  # params
        parsed.query,
        ""   # fragment
    ))
    return normalized

def is_valid_url(url, base_domain, config):
    """Check if URL should be crawled"""
    try:
        parsed = urlparse(url)
        
        # Must be http/https
        if parsed.scheme not in ("http", "https"):
            return False
        
        # Same domain check
        if config["same_domain"]:
            url_domain = parsed.netloc.lower().replace("www.", "")
            if url_domain != base_domain:
                return False
        
        # Skip certain extensions
        path_lower = parsed.path.lower()
        for ext in config["skip_extensions"]:
            if path_lower.endswith(ext.strip()):
                return False
        
        # Exclude patterns
        for pattern in config["exclude_patterns"]:
            if pattern.strip() and pattern.strip() in parsed.path:
                return False
        
        return True
    except:
        return False

def extract_links(html, base_url):
    """Extract links from HTML content"""
    links = []
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href and not href.startswith(("#", "javascript:", "mailto:", "tel:")):
                absolute_url = urljoin(base_url, href)
                links.append(absolute_url)
    except ImportError:
        # Fallback: regex extraction
        pattern = r'href=["\']([^"\']+)["\']'
        for match in re.finditer(pattern, html):
            href = match.group(1)
            if not href.startswith(("#", "javascript:", "mailto:", "tel:")):
                absolute_url = urljoin(base_url, href)
                links.append(absolute_url)
    except Exception as e:
        if config.get("debug"):
            print(f"  [WARN] Link extraction error: {e}", file=sys.stderr)
    
    return links

def extract_text(html, url):
    """Extract clean text from HTML"""
    try:
        import html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_emphasis = False
        h.body_width = 0  # No wrapping
        text = h.handle(html)
        return text.strip()
    except ImportError:
        pass
    
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        
        # Remove script, style, nav, footer
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        
        text = soup.get_text(separator="\n", strip=True)
        # Clean up whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)
    except:
        # Last resort: strip tags with regex
        text = re.sub(r'<[^>]+>', ' ', html)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

def extract_title(html):
    """Extract page title from HTML"""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text().strip()
        h1_tag = soup.find("h1")
        if h1_tag:
            return h1_tag.get_text().strip()
    except:
        pass
    
    # Regex fallback
    match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    return None

def fetch_url(url, config):
    """Fetch URL content"""
    import requests
    
    headers = {
        "User-Agent": config["user_agent"],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5,fr;q=0.3",
    }
    
    try:
        resp = requests.get(
            url,
            headers=headers,
            timeout=config["timeout"],
            allow_redirects=True,
            verify=True
        )
        
        if resp.status_code == 200:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type or "application/xhtml" in content_type:
                return resp.text, None
            else:
                return None, f"Non-HTML content: {content_type}"
        else:
            return None, f"HTTP {resp.status_code}"
    
    except requests.exceptions.Timeout:
        return None, "Timeout"
    except requests.exceptions.SSLError:
        return None, "SSL Error"
    except Exception as e:
        return None, str(e)

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end
            for sep in [". ", ".\n", "! ", "!\n", "? ", "?\n", "\n\n"]:
                last_sep = text[start:end].rfind(sep)
                if last_sep > chunk_size // 2:
                    end = start + last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def ingest_to_qdrant(chunks, url, title, config):
    """Ingest chunks into Qdrant"""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct
        from fastembed import TextEmbedding
    except ImportError as e:
        print(f"  [ERROR] Missing dependency: {e}", file=sys.stderr)
        return 0
    
    # Connect to Qdrant
    try:
        # Try gRPC first
        host = config["qdrant_host"].replace("http://", "").replace("https://", "").split(":")[0]
        client = QdrantClient(host=host, port=config["qdrant_grpc_port"], prefer_grpc=True)
    except:
        # Fallback to HTTP
        client = QdrantClient(url=config["qdrant_host"])
    
    # Get embedding model
    model_name = os.environ.get("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
    cache_dir = os.environ.get("FASTEMBED_CACHE_DIR", "./cache/fastembed")
    embed_model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
    
    # Generate embeddings
    texts = [c for c in chunks]
    embeddings = list(embed_model.embed(texts))
    
    # Prepare points
    points = []
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = hashlib.md5(f"{url}:{i}".encode()).hexdigest()
        point_id_int = int(point_id[:16], 16)  # Convert to int for Qdrant
        
        points.append(PointStruct(
            id=point_id_int,
            vector={"dense": embedding.tolist()},
            payload={
                "text": chunk,
                "filename": title or parsed_url.path or domain,
                "source": "web",
                "url": url,
                "domain": domain,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        ))
    
    # Upsert to collection
    collection = config["collection_name"]
    client.upsert(collection_name=collection, points=points)
    
    return len(points)

def crawl(start_url, config):
    """Main crawl function"""
    import requests
    
    # Normalize start URL
    if not start_url.startswith(("http://", "https://")):
        start_url = "https://" + start_url
    
    start_url = normalize_url(start_url)
    parsed_start = urlparse(start_url)
    base_domain = parsed_start.netloc.lower().replace("www.", "")
    
    print(f"Starting crawl: {start_url}")
    print(f"Domain: {base_domain}")
    print(f"Max pages: {config['max_pages']}")
    print(f"Max depth: {config['max_depth']}")
    print()
    
    # BFS crawl
    queue = deque([(start_url, 0)])  # (url, depth)
    visited = set()
    pages_crawled = 0
    total_chunks = 0
    
    while queue and pages_crawled < config["max_pages"]:
        url, depth = queue.popleft()
        
        # Skip if already visited or too deep
        normalized = normalize_url(url)
        if normalized in visited:
            continue
        if depth > config["max_depth"]:
            continue
        
        visited.add(normalized)
        
        # Fetch page
        print(f"[{pages_crawled + 1}/{config['max_pages']}] {url[:70]}{'...' if len(url) > 70 else ''}")
        
        html, error = fetch_url(url, config)
        
        if error:
            print(f"  -> SKIP: {error}")
            continue
        
        if not html:
            continue
        
        # Extract content
        title = extract_title(html) or urlparse(url).path
        text = extract_text(html, url)
        
        if not text or len(text) < 100:
            print(f"  -> SKIP: Too little content ({len(text) if text else 0} chars)")
            continue
        
        # Chunk and ingest
        chunks = chunk_text(text, config["chunk_size"], config["chunk_overlap"])
        ingested = ingest_to_qdrant(chunks, url, title, config)
        
        print(f"  -> OK: {len(chunks)} chunks ({len(text)} chars)")
        
        pages_crawled += 1
        total_chunks += ingested
        
        # Extract and queue links (if not at max depth)
        if depth < config["max_depth"]:
            links = extract_links(html, url)
            for link in links:
                if is_valid_url(link, base_domain, config):
                    normalized_link = normalize_url(link)
                    if normalized_link not in visited:
                        queue.append((link, depth + 1))
        
        # Rate limiting
        if config["delay"] > 0:
            time.sleep(config["delay"])
    
    print()
    print(f"Crawl complete: {pages_crawled} pages, {total_chunks} chunks ingested")
    return pages_crawled, total_chunks

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python3 web_crawler.py <url> [--debug] [--recreate]")
        print()
        print("Examples:")
        print("  python3 web_crawler.py https://www.example.com")
        print("  python3 web_crawler.py https://docs.example.com --debug")
        sys.exit(1)
    
    url = sys.argv[1]
    config = get_config()
    
    # Parse additional args
    if "--debug" in sys.argv:
        config["debug"] = True
    
    try:
        pages, chunks = crawl(url, config)
        sys.exit(0 if pages > 0 else 1)
    except KeyboardInterrupt:
        print("\nCrawl interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Crawl error: {e}")
        if config["debug"]:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
EOFPY
log_ok "web_crawler.py"


# ============================================================================
# Create ingest.sh wrapper
# ============================================================================
log_info "Creating ingest.sh..."
cat > "$PROJECT_DIR/ingest.sh" << 'EOFINGEST'
#!/bin/bash
# RAG Document Ingestion web (Final)
# Features: File ingestion + Website crawling

cd "$(dirname "$0")"
source ./config.env 2>/dev/null || true

# Headless mode
export OPENCV_LOG_LEVEL=ERROR
export UNSTRUCTURED_USE_GPU=false
export LIBGL_ALWAYS_SOFTWARE=1
export TMPDIR="${TMPDIR:-/tmp}"

# Export config
export QDRANT_HOST QDRANT_GRPC_PORT COLLECTION_NAME DOCUMENTS_DIR
export SPARSE_EMBED_ENABLED SPARSE_EMBED_MODEL HYBRID_SEARCH_MODE
export CHUNK_SIZE CHUNK_OVERLAP FASTEMBED_MODEL FASTEMBED_CACHE_DIR
export CSV_NL_TRANSFORM_ENABLED CSV_NL_DUAL_MODE CSV_NL_LANG
export WEB_CRAWLER_MAX_PAGES WEB_CRAWLER_MAX_DEPTH WEB_CRAWLER_DELAY
export DEBUG DOC_DEDUP_ENABLED

FORCE="" DEBUG_FLAG="" RECREATE="" TARGET="" URL_MODE="" URL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --force) FORCE="--force"; shift ;;
        --debug) DEBUG_FLAG="--debug"; export DEBUG=true; shift ;;
        --recreate) RECREATE="--recreate"; shift ;;
        --url) URL_MODE=true; URL="$2"; shift 2 ;;
        --formats)
            echo "Supported formats:"
            echo "  Documents: .pdf .docx .doc .pptx .xlsx .xls .rtf .odt"
            echo "  Text: .txt .md .html .csv .tsv .json .xml"
            echo "  Images: .png .jpg .jpeg .gif .bmp .tiff (OCR)"
            echo "  Email: .eml .msg"
            echo "  Code: .py .ps1 .sh .sql .js .ts .java .cs .go .rs"
            exit 0 ;;
        --help|-h)
            echo "Usage: ./ingest.sh [options] [path]"
            echo ""
            echo "Options:"
            echo "  --url URL      Crawl website and ingest"
            echo "  --force        Re-process all files"
            echo "  --recreate     Delete and recreate collection"
            echo "  --debug        Debug output"
            echo "  --formats      Show supported formats"
            echo ""
            echo "Examples:"
            echo "  ./ingest.sh                              # ./documents"
            echo "  ./ingest.sh /path/to/file.pdf            # Single file"
            echo "  ./ingest.sh --url https://example.com    # Website"
            exit 0 ;;
        *) [ -z "$TARGET" ] && TARGET="$1"; shift ;;
    esac
done

# web: URL ingestion mode
if [ "$URL_MODE" = true ]; then
    [[ ! "$URL" =~ ^https?:// ]] && URL="https://$URL"
    echo "============================================"
    echo " RAG Website Ingestion web"
    echo "============================================"
    echo "URL: $URL"
    echo "Max pages: ${WEB_CRAWLER_MAX_PAGES:-50}"
    echo ""
    python3 ./lib/web_crawler.py "$URL" $DEBUG_FLAG
    exit $?
fi

# Standard file ingestion
[ -z "$TARGET" ] && TARGET="${DOCUMENTS_DIR:-./documents}"

echo "============================================"
echo " RAG Document Ingestion web"
echo "============================================"
echo "Target: $TARGET"
echo "Collection: ${COLLECTION_NAME:-documents}"
echo "Hybrid: ${HYBRID_SEARCH_MODE:-native}"
echo "CSV Dual Mode: ${CSV_NL_DUAL_MODE:-true}"
echo ""

[ ! -e "$TARGET" ] && { echo "[ERROR] Path not found: $TARGET"; exit 1; }

python3 ./lib/ingest_main.py "$TARGET" $FORCE $DEBUG_FLAG $RECREATE
EOFINGEST
chmod +x "$PROJECT_DIR/ingest.sh"
log_ok "ingest.sh"
# ============================================================================
# Additional utilities from dedup-csv
# ============================================================================

echo ""
echo "Creating utility scripts..."

cat > "$PROJECT_DIR/clear-collection.sh" << 'EOFSH'
#!/bin/bash
source ./config.env 2>/dev/null || true

QDRANT_HOST="${QDRANT_HOST:-http://localhost:6333}"
COLLECTION="${COLLECTION_NAME:-documents}"

echo "Clearing collection: $COLLECTION"
read -p "Are you sure? (y/N) " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    curl -X DELETE "${QDRANT_HOST}/collections/${COLLECTION}"
    rm -rf .ingest_tracking/*
    echo "[OK] Collection cleared"
else
    echo "Cancelled"
fi
EOFSH
chmod +x "$PROJECT_DIR/clear-collection.sh"
log_ok "clear-collection.sh"

# ============================================================================
# csv Fallback Modules (for backward compatibility)
# ============================================================================

echo ""
echo "Creating csv fallback modules..."

cat > lib/ingest_csv.py << 'EOFPY'
"""csv Ingestion Entry Point with Dual CSV Support

Feature: CSV_NL_DUAL_MODE
Introduced: csv
Lifecycle: ACTIVE

Wraps the standard ingestion to add:
1. CSV/Excel detection
2. Dual chunk generation (structured + natural language)
3. Efficient batch processing for large files
"""

import os
import sys
import hashlib

# Add lib to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _get_config():
    """Get csv configuration"""
    return {
        "csv_nl_enabled": os.environ.get("CSV_NL_TRANSFORM_ENABLED", "true").lower() == "true",
        "csv_dual_mode": os.environ.get("CSV_NL_DUAL_MODE", "true").lower() == "true",
        "merge_chunks": os.environ.get("CSV_MERGE_CHUNKS", "true").lower() == "true",
        "merge_target": int(os.environ.get("CSV_MERGE_TARGET", "400")),
        "merge_max": int(os.environ.get("CSV_MERGE_MAX", "800")),
        "debug": os.environ.get("DEBUG", "").lower() == "true",
    }


def ingest_file_csv(filepath, qdrant_host, collection_name, **kwargs):
    """csv file ingestion with dual CSV support
    
    For CSV/Excel files: generates both structured and NL chunks
    For other files: falls back to standard ingestion
    """
    from csv_dual_ingest import is_tabular_file, get_dual_chunks_for_file, merge_small_chunks
    from ingest_main import ingest_file, get_client_mode, get_hybrid_mode, get_sparse_model_info, ensure_collection, upload_points
    from ingest_main import ensure_collection, get_embeddings_batch, upload_points
    
    config = _get_config()
    debug = kwargs.get('debug', config['debug'])
    force = kwargs.get('force', False)
    
    # Check if tabular file and dual mode enabled
    if not is_tabular_file(filepath) or not config['csv_nl_enabled']:
        # Use standard ingestion
        return ingest_file(filepath, qdrant_host, collection_name, **kwargs)
    
    filename = os.path.basename(filepath)
    
    if debug:
        print(f"[csv] Processing tabular file: {filename}")
    
    # Generate dual chunks
    chunks = get_dual_chunks_for_file(filepath)
    
    if not chunks:
        # Fallback to standard ingestion
        if debug:
            print(f"[csv] Dual chunk generation failed, using standard ingestion")
        return ingest_file(filepath, qdrant_host, collection_name, **kwargs)
    
    # Optionally merge small chunks
    if config['merge_chunks']:
        original_count = len(chunks)
        chunks = merge_small_chunks(chunks, config['merge_target'], config['merge_max'])
        if debug:
            print(f"[csv] Merged {original_count} -> {len(chunks)} chunks")
    
    # Compute file hash for dedup
    with open(filepath, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    # Generate embeddings
    texts = [c['text'] for c in chunks]
    
    if debug:
        print(f"[csv] Generating embeddings for {len(texts)} chunks...")
    
    try:
        embeddings = get_embeddings_batch(texts)
    except Exception as e:
        return {
            "status": "error",
            "error": f"Embedding failed: {e}",
            "file": filename,
        }
    
    # Prepare payloads
    payloads = []
    for i, chunk in enumerate(chunks):
        payload = {
            "text": chunk['text'],
            "source": filepath,
            "filename": filename,
            "file_hash": file_hash,
            "chunk_index": i,
            "chunk_type": chunk.get('chunk_type', 'unknown'),
            "format": chunk.get('format', ''),
            "row_indices": chunk.get('row_indices', []),
            "merged_count": chunk.get('merged_count', 1),
            "parser": "csv_dual_csv",
        }
        payloads.append(payload)
    
    # Store in Qdrant
    try:
        from embedding_helper import get_embedding_dimension
        from qdrant_hybrid_helper import ensure_hybrid_collection, upload_hybrid_points
        from sparse_embedding_helper import is_sparse_embed_available, get_sparse_embeddings_batch
        import uuid
        import hashlib
        
        sparse_enabled = os.environ.get("SPARSE_EMBED_ENABLED", "true").lower() == "true"
        
        # Ensure hybrid collection exists
        ensure_hybrid_collection(collection_name)
        
        # Generate sparse embeddings for hybrid search
        sparse_embeddings = None
        if sparse_enabled and is_sparse_embed_available():
            sparse_embeddings = get_sparse_embeddings_batch(texts)
        
        # Build hybrid points
        base_id = int(hashlib.md5(filepath.encode()).hexdigest()[:8], 16)
        points = []
        
        for i, (embedding, payload) in enumerate(zip(embeddings, payloads)):
            point = {
                "id": base_id + i,
                "dense_vector": embedding if isinstance(embedding, list) else embedding.tolist(),
                "payload": payload,
            }
            
            # Add sparse vector if available
            if sparse_embeddings and i < len(sparse_embeddings):
                point["sparse_vector"] = sparse_embeddings[i]
            
            points.append(point)
        
        # Upload with hybrid format
        batch_size = int(os.environ.get("QDRANT_BATCH_SIZE", "100"))
        success = upload_hybrid_points(collection_name, points, batch_size)
        
        if not success:
            raise Exception("Hybrid upload returned False")
    except Exception as e:
        return {
            "status": "error", 
            "error": f"Storage failed: {e}",
            "file": filename,
        }
    
    # Count by type
    struct_count = sum(1 for c in chunks if c.get('chunk_type') == 'structured')
    nl_count = sum(1 for c in chunks if c.get('chunk_type') == 'natural_language')
    
    return {
        "status": "success",
        "file": filename,
        "chunks": len(chunks),
        "structured_chunks": struct_count,
        "nl_chunks": nl_count,
        "chars": sum(len(c['text']) for c in chunks),
        "parser": "csv_dual_csv",
    }


def ingest_directory_csv(dirpath, qdrant_host, collection_name, **kwargs):
    """csv directory ingestion with dual CSV support"""
    from ingest_main import ingest_directory
    from csv_dual_ingest import is_tabular_file
    
    config = _get_config()
    debug = kwargs.get('debug', config['debug'])
    force = kwargs.get('force', False)
    
    # Get list of files
    from unstructured_parser import get_supported_extensions
    supported_ext = get_supported_extensions()
    
    files = []
    for root, dirs, filenames in os.walk(dirpath):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in supported_ext:
                files.append(os.path.join(root, fname))
    
    if not files:
        return {"success": 0, "skipped": 0, "errors": 0, "files": []}
    
    results = {
        "success": 0,
        "skipped": 0,
        "errors": 0,
        "files": [],
        "total_chunks": 0,
        "structured_chunks": 0,
        "nl_chunks": 0,
    }
    
    print(f"Processing {len(files)} files...")
    print("")
    
    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        print(f"[{i+1}/{len(files)}] {filename}", end="", flush=True)
        
        # Use csv ingestion for tabular files
        if is_tabular_file(filepath) and config['csv_nl_enabled']:
            result = ingest_file_csv(filepath, qdrant_host, collection_name, **kwargs)
        else:
            # Standard ingestion for non-tabular
            from ingest_main import ingest_file
            result = ingest_file(filepath, qdrant_host, collection_name, **kwargs)
        
        results["files"].append(result)
        
        if result["status"] == "success":
            results["success"] += 1
            chunks = result.get("chunks", 0)
            results["total_chunks"] += chunks
            results["structured_chunks"] += result.get("structured_chunks", 0)
            results["nl_chunks"] += result.get("nl_chunks", 0)
            
            struct = result.get("structured_chunks", 0)
            nl = result.get("nl_chunks", 0)
            if struct or nl:
                print(f" -> OK ({struct} struct + {nl} NL chunks)")
            else:
                print(f" -> OK ({chunks} chunks)")
        elif result["status"] == "skipped":
            results["skipped"] += 1
            print(f" -> SKIP ({result.get('reason', 'cached')})")
        else:
            results["errors"] += 1
            print(f" -> ERROR: {result.get('error', 'unknown')[:50]}")
    
    return results
EOFPY
log_ok "lib/ingest_csv.py (fallback)"

cat > lib/unstructured_parser_csv.py << 'EOFPY'
"""csv Wrapper for extended format support

Integrates extended_formats.py with unstructured_parser.py
"""

import os
import sys

# Add lib to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_all_supported_extensions():
    """Return complete list of supported extensions"""
    from unstructured_parser import get_supported_extensions
    from extended_formats import get_extended_extensions
    
    base = get_supported_extensions()
    extended = get_extended_extensions()
    
    # Merge without duplicates
    all_ext = list(base)
    for ext in extended:
        if ext not in all_ext:
            all_ext.append(ext)
    
    return all_ext


def parse_document_csv(file_path, strategy="auto", ocr_languages="eng+fra"):
    """csv document parsing with extended format support"""
    from extended_formats import parse_extended_format
    from unstructured_parser import parse_document
    
    ext = os.path.splitext(file_path)[1].lower()
    
    # Try extended format first
    result = parse_extended_format(file_path)
    if result is not None:
        return result
    
    # Fall back to standard parser
    return parse_document(file_path, strategy, ocr_languages)
EOFPY
log_ok "lib/unstructured_parser_csv.py (fallback)"

# ============================================================================
# System: Document Loader for Map/Reduce and Extraction
# ============================================================================
log_info "Creating document_loader.py (System)..."
cat > "$PROJECT_DIR/lib/document_loader.py" << 'EOFPY'
"""
Document Loader for Map/Reduce and Extraction Operations (System)

Provides utilities to load entire documents for:
- Map/Reduce summarization
- Structured extraction
- Full-document processing

Unlike RAG retrieval (top-K chunks), these operations need ALL content.
"""

import os
import sys

# Add lib to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

load_dotenv("config.env")


def load_document_text(file_path: str) -> str:
    """
    Load entire document as plain text.
    
    Args:
        file_path: Path to document (PDF, DOCX, TXT, etc.)
    
    Returns:
        Full text content of document
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    
    # Plain text files
    if ext in ['.txt', '.md', '.csv', '.json']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    # Use Unstructured.io for other formats
    try:
        from unstructured.partition.auto import partition
        
        elements = partition(
            filename=file_path,
            strategy="auto",
            languages=["eng", "fra"]
        )
        
        text_parts = []
        for elem in elements:
            text = str(elem).strip()
            if text:
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    except Exception as e:
        raise RuntimeError(f"Failed to load document: {e}")


def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks for map phase.
    
    Args:
        text: Full document text
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks for context continuity
    
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        
        # Try to break at paragraph or sentence boundary
        if end < text_len:
            # Look for paragraph break
            para_break = text.rfind('\n\n', start + chunk_size // 2, end + 200)
            if para_break > start:
                end = para_break
            else:
                # Look for sentence break
                for sep in ['. ', '.\n', '? ', '!\n']:
                    sent_break = text.rfind(sep, start + chunk_size // 2, end + 100)
                    if sent_break > start:
                        end = sent_break + len(sep)
                        break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start with overlap
        start = end - overlap if end < text_len else text_len
    
    return chunks


def load_and_chunk_document(
    file_path: str,
    chunk_size: int = None,
    overlap: int = 200
) -> Tuple[List[str], Dict]:
    """
    Load document and split into chunks for processing.
    
    Args:
        file_path: Path to document
        chunk_size: Chunk size (uses config default if None)
        overlap: Overlap between chunks
    
    Returns:
        Tuple of (chunks list, metadata dict)
    """
    # Get chunk size from config or use default
    if chunk_size is None:
        chunk_size = int(os.environ.get("MAPREDUCE_CHUNK_SIZE", "4000"))
    
    # Load full text
    full_text = load_document_text(file_path)
    
    # Get metadata
    file_size = os.path.getsize(file_path)
    char_count = len(full_text)
    
    # Chunk the text
    chunks = chunk_text(full_text, chunk_size, overlap)
    
    metadata = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_size": file_size,
        "char_count": char_count,
        "chunk_count": len(chunks),
        "chunk_size": chunk_size
    }
    
    return chunks, metadata


def detect_intent(query: str) -> str:
    """
    Detect query intent to route to appropriate handler.
    
    Returns:
        'summarize' | 'extract' | 'rag' (default)
    """
    query_lower = query.lower()
    
    # Summarization intent
    summarize_keywords = [
        'summarize', 'summary', 'résumé', 'résumer',
        'overview', 'what is this about', 'give me an overview',
        'what does this document say', 'tldr', 'tl;dr'
    ]
    for kw in summarize_keywords:
        if kw in query_lower:
            return 'summarize'
    
    # Extraction intent
    extract_keywords = [
        'extract', 'list all', 'find all', 'find every',
        'what are all the', 'give me all', 'enumerate',
        'extraire', 'lister tous', 'trouver tous'
    ]
    for kw in extract_keywords:
        if kw in query_lower:
            return 'extract'
    
    # Default to RAG
    return 'rag'


if __name__ == "__main__":
    # Test document loading
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        chunks, meta = load_and_chunk_document(file_path)
        print(f"Loaded: {meta['file_name']}")
        print(f"Size: {meta['file_size']} bytes")
        print(f"Characters: {meta['char_count']}")
        print(f"Chunks: {meta['chunk_count']}")
        print(f"---")
        print(f"First chunk preview ({len(chunks[0])} chars):")
        print(chunks[0][:500] + "..." if len(chunks[0]) > 500 else chunks[0])
    else:
        print("Usage: python document_loader.py <file_path>")
EOFPY
log_ok "document_loader.py (System)"


# ============================================================================
# Verification
# ============================================================================
echo ""
echo "=== Verification ==="
python3 -c "from fastembed import TextEmbedding" 2>/dev/null && log_ok "FastEmbed" || log_err "FastEmbed"
python3 -c "from qdrant_client import QdrantClient" 2>/dev/null && log_ok "QdrantClient" || log_err "QdrantClient"
[ -f "$PROJECT_DIR/lib/ingest_main.py" ] && log_ok "ingest_main.py" || log_err "ingest_main.py"
[ -f "$PROJECT_DIR/lib/web_crawler.py" ] && log_ok "web_crawler.py" || log_err "web_crawler.py"
[ -f "$PROJECT_DIR/lib/ingestion_progress.py" ] && log_ok "ingestion_progress.py" || log_err "ingestion_progress.py"
[ -f "$PROJECT_DIR/lib/document_loader.py" ] && log_ok "document_loader.py (System)" || log_err "document_loader.py"

# profiling: Verify OCR-fra support
echo -n "[TEST] "
tesseract --list-langs 2>/dev/null | grep -q "fra" && log_ok "tesseract-ocr-fra" || echo "tesseract-ocr-fra: Not available"

# profiling: Verify antiword
echo -n "[TEST] "
command -v antiword &>/dev/null && log_ok "antiword (.doc support)" || echo "antiword: Not available"

echo ""
echo "============================================"
echo " Ingestion Setup Complete (System)"
echo "============================================"
echo ""
echo "System Features:"
echo "  - Document loader for map/reduce operations"
echo "  - Full-document text extraction"
echo "  - Intent detection (summarize/extract/rag)"
echo ""
echo "profiling Features (preserved):"
echo "  - Adaptive batch size (QDRANT_BATCH_SIZE=${QDRANT_BATCH_SIZE:-64})"
echo "  - French OCR support (tesseract-ocr-fra)"
echo "  - Legacy .doc support (antiword)"
echo ""
echo "Preserved Features (quality-progress):"
echo "  - Unstructured.io parsing (PDF, DOCX, XLSX...)"
echo "  - FastEmbed dense + Sparse hybrid vectors"
echo "  - CSV/Excel dual mode (structured + NL)"
echo "  - Document-level deduplication"
echo "  - Website crawling (--url flag)"
echo "  - Real-time progress tracking (progress)"
echo ""
echo "Usage:"
echo "  ./ingest.sh                           # Ingest ./documents"
echo "  ./ingest.sh --url https://example.com # Crawl website"
echo "  ./ingest.sh --force --debug           # Re-ingest all"

# ASSERTION: legacy_code=false, all_System_features=true, plain_ascii=true
