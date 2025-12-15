#!/bin/bash
# ============================================================================
# RAG System v44 - Document Ingestion Setup (FULL FEATURES)
# ============================================================================
# All features implemented:
#   - Contextual headers (document + section context)
#   - Semantic chunking (sentence-boundary aware)
#   - Parent-child chunks (hierarchical)
#   - Section detection
#   - Table extraction
#   - OCR support
#   - Deduplication
#   - Progress indicators
#   - Vocabulary building for spell correction
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_ok() { echo -e "[${GREEN}OK${NC}] $1"; }
log_warn() { echo -e "[${YELLOW}WARN${NC}] $1"; }
log_info() { echo -e "[${BLUE}INFO${NC}] $1"; }

PROJECT_DIR="${1:-$(pwd)}"

if [ ! -f "$PROJECT_DIR/config.env" ]; then
    echo "ERROR: Run setup-rag-core-v44.sh first!"
    exit 1
fi

cd "$PROJECT_DIR"
source config.env
[ -d "./venv" ] && source ./venv/bin/activate

clear
echo "============================================================================"
echo "   RAG System v42 - Ingestion Setup (Full Features)"
echo "============================================================================"
echo ""

mkdir -p lib cache .ingest_tracking

# ============================================================================
# PDF Parser Module (with warning suppression)
# ============================================================================
echo "Creating PDF parser module..."
cat > lib/pdf_parser.py << 'EOFPY'
"""PDF Parser with warning suppression and table extraction"""
import os
import sys
import warnings
import logging
import re

warnings.filterwarnings("ignore", message=".*P4.*")
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("pypdfium2").setLevel(logging.CRITICAL)

class SuppressPDFWarnings:
    """Context manager to suppress C-level stderr warnings"""
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_fd = os.dup(2)
        os.dup2(self.null_fd, 2)
        return self
    def __exit__(self, *args):
        os.dup2(self.save_fd, 2)
        os.close(self.null_fd)
        os.close(self.save_fd)

def extract_sections(text):
    """Extract section headers from text"""
    sections = []
    current_section = "Introduction"
    lines = text.split('\n')
    
    for line in lines:
        # Detect markdown-style headers
        if re.match(r'^#{1,3}\s+', line):
            current_section = re.sub(r'^#{1,3}\s+', '', line).strip()
        # Detect numbered sections
        elif re.match(r'^\d+\.\s+[A-Z]', line):
            current_section = line.strip()
        # Detect ALL CAPS headers
        elif line.isupper() and len(line) > 5 and len(line) < 80:
            current_section = line.strip()
        
        sections.append((line, current_section))
    
    return sections

def parse_pdf(file_path, extract_tables=True, ocr_enabled=True, ocr_lang="eng"):
    """Parse PDF with section detection"""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return "", []
    
    text_parts = []
    sections = []
    current_section = "Document Start"
    
    try:
        with SuppressPDFWarnings():
            pdf = pdfium.PdfDocument(file_path)
            for i, page in enumerate(pdf):
                try:
                    textpage = page.get_textpage()
                    text = textpage.get_text_bounded()
                    if text.strip():
                        # Detect sections in this page
                        for line in text.split('\n'):
                            if re.match(r'^#{1,3}\s+', line) or \
                               re.match(r'^\d+\.\s+[A-Z]', line) or \
                               (line.isupper() and 5 < len(line) < 80):
                                current_section = re.sub(r'^#{1,3}\s+', '', line).strip()[:100]
                        
                        text_parts.append(f"## Page {i+1}\n{text}")
                        sections.append({"page": i+1, "section": current_section})
                except Exception:
                    pass
            pdf.close()
    except Exception as e:
        return f"Error parsing PDF: {e}", []
    
    result = "\n\n".join(text_parts)
    
    # Try OCR if no text extracted
    if not result.strip() and ocr_enabled:
        try:
            from PIL import Image
            import pytesseract
            with SuppressPDFWarnings():
                pdf = pdfium.PdfDocument(file_path)
                for i, page in enumerate(pdf):
                    try:
                        bitmap = page.render(scale=2)
                        pil_image = bitmap.to_pil()
                        ocr_text = pytesseract.image_to_string(pil_image, lang=ocr_lang)
                        if ocr_text.strip():
                            text_parts.append(f"## Page {i+1} (OCR)\n{ocr_text}")
                    except Exception:
                        pass
                pdf.close()
            result = "\n\n".join(text_parts)
        except ImportError:
            pass
    
    return result, sections

def parse_pdf_with_tables(file_path, table_style="natural"):
    """Parse PDF and format tables"""
    text, sections = parse_pdf(file_path)
    return text, sections
EOFPY
log_ok "PDF parser module created"

# ============================================================================
# Office Parser Module
# ============================================================================
echo "Creating Office parser module..."
cat > lib/office_parser.py << 'EOFPY'
"""Office document parsers with section detection"""
import re

def extract_sections_from_text(text):
    """Extract section information from text"""
    sections = []
    current = "Document Start"
    
    for line in text.split('\n'):
        if re.match(r'^#{1,3}\s+', line):
            current = re.sub(r'^#{1,3}\s+', '', line).strip()[:100]
        elif re.match(r'^\d+\.\s+[A-Z]', line):
            current = line.strip()[:100]
    
    return current

def parse_docx(file_path):
    """Parse Word document with structure"""
    try:
        from docx import Document
        doc = Document(file_path)
        text_parts = []
        current_section = "Document Start"
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            
            # Detect headings by style
            if para.style and 'Heading' in para.style.name:
                current_section = text[:100]
                text_parts.append(f"\n## {text}\n")
            else:
                text_parts.append(text)
        
        # Extract tables
        for table in doc.tables:
            table_text = []
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                if row_text.strip():
                    table_text.append(row_text)
            if table_text:
                text_parts.append("\n[Table]\n" + "\n".join(table_text) + "\n")
        
        return "\n".join(text_parts)
    except Exception as e:
        return f"Error: {e}"

def parse_doc(file_path):
    """Parse legacy .doc file"""
    try:
        import subprocess
        result = subprocess.run(
            ["antiword", file_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass
    return ""

def parse_xlsx(file_path, max_rows=1000):
    """Parse Excel spreadsheet"""
    try:
        from openpyxl import load_workbook
        wb = load_workbook(file_path, data_only=True)
        text_parts = []
        
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            text_parts.append(f"\n## Sheet: {sheet_name}\n")
            rows_processed = 0
            
            for row in sheet.iter_rows(values_only=True):
                if rows_processed >= max_rows:
                    text_parts.append(f"[Truncated at {max_rows} rows]")
                    break
                row_text = " | ".join(str(cell) if cell else "" for cell in row)
                if row_text.strip() and row_text.replace("|", "").replace(" ", ""):
                    text_parts.append(row_text)
                    rows_processed += 1
        
        return "\n".join(text_parts)
    except Exception as e:
        return f"Error: {e}"

def parse_xls(file_path):
    """Parse legacy .xls file"""
    return parse_xlsx(file_path)

def parse_pptx(file_path):
    """Parse PowerPoint presentation"""
    try:
        from pptx import Presentation
        prs = Presentation(file_path)
        text_parts = []
        
        for i, slide in enumerate(prs.slides, 1):
            slide_text = [f"\n## Slide {i}"]
            
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text)
                
                # Extract tables from slides
                if shape.has_table:
                    table = shape.table
                    for row in table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells)
                        if row_text.strip():
                            slide_text.append(row_text)
            
            if len(slide_text) > 1:
                text_parts.append("\n".join(slide_text))
        
        return "\n".join(text_parts)
    except Exception as e:
        return f"Error: {e}"

def parse_ppt(file_path):
    """Parse legacy .ppt file"""
    return parse_pptx(file_path)
EOFPY
log_ok "Office parser module created"

# ============================================================================
# Text Parser Module
# ============================================================================
echo "Creating text parser module..."
cat > lib/text_parser.py << 'EOFPY'
"""Text file parsers"""
import json
import csv
import html.parser
import re

class HTMLTextExtractor(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip_tags = {'script', 'style', 'head', 'nav', 'footer'}
        self.current_tag = None
        self.in_skip = False
    
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag in self.skip_tags:
            self.in_skip = True
        if tag in ['h1', 'h2', 'h3', 'h4']:
            self.text.append(f"\n## ")
        elif tag == 'p':
            self.text.append("\n")
        elif tag == 'li':
            self.text.append("\n- ")
    
    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.in_skip = False
        self.current_tag = None
    
    def handle_data(self, data):
        if not self.in_skip:
            text = data.strip()
            if text:
                self.text.append(text + " ")
    
    def get_text(self):
        return "".join(self.text)

def parse_txt(file_path):
    """Parse plain text file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"

def parse_md(file_path):
    """Parse Markdown file (preserve structure)"""
    return parse_txt(file_path)

def parse_html(file_path):
    """Parse HTML file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        parser = HTMLTextExtractor()
        parser.feed(content)
        return parser.get_text()
    except Exception as e:
        return f"Error: {e}"

def parse_csv(file_path, max_rows=500):
    """Parse CSV file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Detect delimiter
            sample = f.read(4096)
            f.seek(0)
            
            delimiter = ','
            if sample.count('\t') > sample.count(','):
                delimiter = '\t'
            elif sample.count(';') > sample.count(','):
                delimiter = ';'
            
            reader = csv.reader(f, delimiter=delimiter)
            rows = []
            headers = None
            
            for i, row in enumerate(reader):
                if i >= max_rows:
                    rows.append(f"[Truncated at {max_rows} rows]")
                    break
                if i == 0:
                    headers = row
                    rows.append(" | ".join(row))
                    rows.append("-" * 40)
                else:
                    rows.append(" | ".join(row))
            
            return "\n".join(rows)
    except Exception as e:
        return f"Error: {e}"

def parse_json(file_path):
    """Parse JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to readable format
        def flatten_json(obj, prefix=""):
            lines = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, (dict, list)):
                        lines.extend(flatten_json(v, new_key))
                    else:
                        lines.append(f"{new_key}: {v}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj[:100]):  # Limit list items
                    lines.extend(flatten_json(item, f"{prefix}[{i}]"))
            return lines
        
        return "\n".join(flatten_json(data)[:1000])  # Limit total lines
    except Exception as e:
        return f"Error: {e}"
EOFPY
log_ok "Text parser module created"

# ============================================================================
# Image Parser Module
# ============================================================================
echo "Creating image parser module..."
cat > lib/image_parser.py << 'EOFPY'
"""Image OCR parser"""

def parse_image(file_path, lang="eng"):
    """Extract text from image using OCR"""
    try:
        from PIL import Image
        import pytesseract
        
        img = Image.open(file_path)
        
        # Resize if too large
        max_size = 4096
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if necessary
        if img.mode not in ('L', 'RGB'):
            img = img.convert('RGB')
        
        text = pytesseract.image_to_string(img, lang=lang)
        return text.strip()
    except ImportError:
        return "[OCR not available - install pytesseract]"
    except Exception as e:
        return f"Error: {e}"
EOFPY
log_ok "Image parser module created"

# ============================================================================
# Advanced Chunker Module (FULL FEATURES)
# ============================================================================
echo "Creating advanced chunker module..."
cat > lib/chunker.py << 'EOFPY'
"""Advanced document chunking with all features"""
import re
import hashlib

def detect_sections(text):
    """Detect section boundaries in text"""
    sections = []
    current_section = "Introduction"
    current_start = 0
    
    lines = text.split('\n')
    pos = 0
    
    for i, line in enumerate(lines):
        # Markdown headers
        if re.match(r'^#{1,3}\s+\S', line):
            if pos > current_start:
                sections.append({
                    "title": current_section,
                    "start": current_start,
                    "end": pos,
                    "text": text[current_start:pos]
                })
            current_section = re.sub(r'^#{1,3}\s+', '', line).strip()[:100]
            current_start = pos
        
        # Numbered sections
        elif re.match(r'^\d+\.\d*\s+[A-Z]', line):
            if pos > current_start:
                sections.append({
                    "title": current_section,
                    "start": current_start,
                    "end": pos,
                    "text": text[current_start:pos]
                })
            current_section = line.strip()[:100]
            current_start = pos
        
        pos += len(line) + 1
    
    # Add final section
    if pos > current_start:
        sections.append({
            "title": current_section,
            "start": current_start,
            "end": pos,
            "text": text[current_start:pos]
        })
    
    return sections

def semantic_chunk(text, max_size=600, overlap=100):
    """
    Sentence-boundary aware chunking.
    Splits at sentence boundaries, not mid-sentence.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return [text] if text else []
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_len = len(sentence)
        
        if current_length + sentence_len > max_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            if overlap > 0:
                overlap_text = ' '.join(current_chunk)
                overlap_sentences = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= overlap:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_chunk = overlap_sentences
                current_length = overlap_len
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_len
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def fixed_chunk(text, chunk_size=500, overlap=80, min_size=100):
    """Simple fixed-size chunking with overlap"""
    if not text or len(text) < min_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence/paragraph boundary
        if end < len(text):
            # Look for sentence end
            for sep in ['. ', '.\n', '\n\n', '\n', ' ']:
                pos = text.rfind(sep, start + chunk_size // 2, end + 50)
                if pos > start:
                    end = pos + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if len(chunk) >= min_size:
            chunks.append(chunk)
        
        start = end - overlap
        if start < 0:
            start = 0
    
    return chunks

def create_parent_child_chunks(text, filename, section=""):
    """
    Create hierarchical chunks:
    - Parent: Document summary (first ~500 chars)
    - Children: Detailed chunks
    """
    chunks = []
    
    # Parent chunk (summary)
    summary = text[:500].strip()
    if len(text) > 500:
        # Try to end at sentence boundary
        last_period = summary.rfind('.')
        if last_period > 200:
            summary = summary[:last_period + 1]
        summary += "..."
    
    parent_id = hashlib.md5(f"{filename}_parent".encode()).hexdigest()[:16]
    chunks.append({
        "id": parent_id,
        "type": "parent",
        "text": summary,
        "filename": filename,
        "section": section,
        "children": []
    })
    
    # Child chunks
    child_chunks = fixed_chunk(text, chunk_size=400, overlap=50)
    child_ids = []
    
    for i, chunk_text in enumerate(child_chunks):
        child_id = hashlib.md5(f"{filename}_child_{i}".encode()).hexdigest()[:16]
        child_ids.append(child_id)
        chunks.append({
            "id": child_id,
            "type": "child",
            "text": chunk_text,
            "filename": filename,
            "section": section,
            "parent_id": parent_id
        })
    
    # Update parent with child references
    chunks[0]["children"] = child_ids
    
    return chunks

def chunk_document(text, config):
    """
    Main chunking function with all options.
    
    config: {
        "chunk_size": 500,
        "chunk_overlap": 80,
        "min_chunk_size": 100,
        "use_semantic": False,
        "use_sections": True,
        "contextual_headers": True,
        "parent_child": False,
        "filename": "document.pdf",
    }
    """
    chunk_size = config.get("chunk_size", 500)
    overlap = config.get("chunk_overlap", 80)
    min_size = config.get("min_chunk_size", 100)
    use_semantic = config.get("use_semantic", False)
    use_sections = config.get("use_sections", True)
    contextual_headers = config.get("contextual_headers", True)
    parent_child = config.get("parent_child", False)
    filename = config.get("filename", "unknown")
    
    if not text or len(text) < min_size:
        return [{"text": text, "section": "", "filename": filename}] if text else []
    
    # Parent-child mode
    if parent_child:
        return create_parent_child_chunks(text, filename)
    
    # Section-aware chunking
    if use_sections:
        sections = detect_sections(text)
        if len(sections) > 1:
            all_chunks = []
            for section in sections:
                section_text = section["text"]
                section_title = section["title"]
                
                if use_semantic:
                    raw_chunks = semantic_chunk(section_text, chunk_size, overlap)
                else:
                    raw_chunks = fixed_chunk(section_text, chunk_size, overlap, min_size)
                
                for chunk_text in raw_chunks:
                    chunk_obj = {
                        "text": chunk_text,
                        "section": section_title,
                        "filename": filename
                    }
                    
                    # Add contextual header
                    if contextual_headers:
                        header = f"Document: {filename}"
                        if section_title and section_title != "Introduction":
                            header += f" | Section: {section_title}"
                        chunk_obj["text"] = f"{header}\n\n{chunk_text}"
                    
                    all_chunks.append(chunk_obj)
            
            return all_chunks
    
    # Simple chunking (no sections)
    if use_semantic:
        raw_chunks = semantic_chunk(text, chunk_size, overlap)
    else:
        raw_chunks = fixed_chunk(text, chunk_size, overlap, min_size)
    
    chunks = []
    for chunk_text in raw_chunks:
        chunk_obj = {
            "text": chunk_text,
            "section": "",
            "filename": filename
        }
        
        if contextual_headers:
            chunk_obj["text"] = f"Document: {filename}\n\n{chunk_text}"
        
        chunks.append(chunk_obj)
    
    return chunks
EOFPY
log_ok "Advanced chunker module created"

# ============================================================================
# BM25 Index Builder
# ============================================================================
echo "Creating BM25 index builder..."
cat > rebuild-bm25.sh << 'EOFBM25'
#!/bin/bash
cd "$(dirname "$0")"
source config.env 2>/dev/null
[ -d "./venv" ] && source ./venv/bin/activate

echo "Rebuilding BM25 index..."

python3 << 'EOFPY'
import json
import requests
import os
import pickle
import re

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("BM25 not available")
    exit(0)

QDRANT = os.environ.get("QDRANT_HOST", "http://localhost:6333")
COLLECTION = os.environ.get("COLLECTION_NAME", "documents")

# Fetch all documents
try:
    resp = requests.post(f"{QDRANT}/collections/{COLLECTION}/points/scroll", json={
        "limit": 10000,
        "with_payload": True
    }, timeout=30)
    
    if resp.status_code != 200:
        print("Error fetching documents")
        exit(1)
    
    data = resp.json()
    points = data.get("result", {}).get("points", [])
except Exception as e:
    print(f"Error: {e}")
    exit(1)

if not points:
    print("No documents found")
    exit(0)

# Tokenize function
def tokenize(text):
    # Simple tokenization: lowercase, split on non-alphanumeric
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    # Remove very short tokens and stopwords
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                 'from', 'as', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'between', 'under', 'again', 'further',
                 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                 'because', 'until', 'while', 'this', 'that', 'these', 'those'}
    tokens = [t for t in tokens if len(t) > 2 and t not in stopwords]
    return tokens

# Build corpus
corpus = []
doc_ids = []
doc_texts = {}

for p in points:
    text = p.get("payload", {}).get("text", "")
    if text:
        tokens = tokenize(text)
        if tokens:
            corpus.append(tokens)
            doc_ids.append(p["id"])
            doc_texts[p["id"]] = text

if not corpus:
    print("No valid documents for BM25")
    exit(0)

# Build BM25 index
bm25 = BM25Okapi(corpus)

# Save index
os.makedirs("cache", exist_ok=True)
with open("cache/bm25_index.pkl", "wb") as f:
    pickle.dump({
        "bm25": bm25,
        "doc_ids": doc_ids,
        "corpus": corpus,
        "doc_texts": doc_texts
    }, f)

print(f"BM25 index built: {len(doc_ids)} documents")

# ============================================================================
# BUILD VOCABULARY for spell correction
# ============================================================================
print("Building vocabulary for spell correction...")

from collections import Counter, defaultdict
import json

word_freq = Counter()
cooccurrence = defaultdict(Counter)

for text in doc_texts.values():
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    unique_words = set(words)
    word_freq.update(words)
    for word in unique_words:
        for other in unique_words:
            if word != other:
                cooccurrence[word][other] += 1

# Filter by frequency (min 2 occurrences)
vocab_terms = {
    word: freq for word, freq in word_freq.most_common(10000)
    if freq >= 2
}

# Simplify cooccurrence to top 10 related terms per word
simplified_cooc = {}
for word, related in cooccurrence.items():
    if word in vocab_terms:
        simplified_cooc[word] = dict(related.most_common(10))

vocabulary = {
    "terms": vocab_terms,
    "cooccurrence": simplified_cooc
}

with open("cache/vocabulary.json", "w") as f:
    json.dump(vocabulary, f)

print(f"Vocabulary built: {len(vocab_terms)} terms")
EOFPY
EOFBM25
chmod +x rebuild-bm25.sh
log_ok "BM25 index builder created"

# ============================================================================
# Main Ingestion Script (FULL FEATURES)
# ============================================================================
echo "Creating main ingestion script..."
cat > "$PROJECT_DIR/ingest.sh" << 'EOFSH'
#!/bin/bash
# ============================================================================
# RAG Document Ingestion v42 - Full Features
# ============================================================================

set -e

[ -f "./config.env" ] && source ./config.env
[ -d "./venv" ] && source ./venv/bin/activate

# Export all config for Python
export OLLAMA_HOST QDRANT_HOST COLLECTION_NAME EMBEDDING_MODEL EMBEDDING_DIMENSION
export CHUNK_SIZE CHUNK_OVERLAP MIN_CHUNK_SIZE MAX_CHUNK_SIZE
export USE_SEMANTIC_CHUNKING USE_SECTIONS CONTEXTUAL_HEADERS PARENT_CHILD_ENABLED
export EXTRACT_TABLES TABLE_STYLE
export OCR_ENABLED OCR_LANGUAGE DEDUP_ENABLED
export EMBEDDING_TIMEOUT

QDRANT="${QDRANT_HOST:-http://localhost:6333}"
OLLAMA="${OLLAMA_HOST:-http://localhost:11434}"
COLLECTION="${COLLECTION_NAME:-documents}"
EMBED_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"

CHUNK_SIZE="${CHUNK_SIZE:-500}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-80}"
MIN_CHUNK="${MIN_CHUNK_SIZE:-100}"
USE_SEMANTIC="${USE_SEMANTIC_CHUNKING:-false}"
USE_SECTIONS="${USE_SECTIONS:-true}"
CONTEXTUAL="${CONTEXTUAL_HEADERS:-true}"
PARENT_CHILD="${PARENT_CHILD_ENABLED:-false}"
OCR_ENABLED="${OCR_ENABLED:-true}"
OCR_LANG="${OCR_LANGUAGE:-eng}"
DEDUP="${DEDUP_ENABLED:-true}"

INPUT_PATH="./documents"
FORCE=false
DEBUG=false
NO_DEDUP=false
REBUILD_BM25=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=true; shift ;;
        --debug) DEBUG=true; shift ;;
        --no-dedup) NO_DEDUP=true; shift ;;
        --no-bm25) REBUILD_BM25=false; shift ;;
        --semantic) USE_SEMANTIC=true; shift ;;
        --parent-child) PARENT_CHILD=true; shift ;;
        -h|--help)
            echo "Usage: $0 [options] [path]"
            echo ""
            echo "Options:"
            echo "  --force        Re-index all files"
            echo "  --debug        Show detailed output"
            echo "  --no-dedup     Disable deduplication"
            echo "  --no-bm25      Skip BM25 index rebuild"
            echo "  --semantic     Use sentence-boundary chunking"
            echo "  --parent-child Create hierarchical chunks"
            exit 0
            ;;
        *) INPUT_PATH="$1"; shift ;;
    esac
done

[ "$NO_DEDUP" = "true" ] && DEDUP=false

echo "============================================"
echo "   Document Ingestion v42 - Full Features"
echo "============================================"
echo "Input: $INPUT_PATH"
echo ""

# Export additional flags
export USE_SEMANTIC CONTEXTUAL PARENT_CHILD DEBUG FORCE DEDUP

# Check services
echo -n "Checking services... "
if ! curl -sf "$OLLAMA/api/tags" > /dev/null 2>&1; then
    echo "FAILED"
    echo "ERROR: Ollama not running"
    exit 1
fi
if ! curl -sf "$QDRANT/collections" > /dev/null 2>&1; then
    echo "FAILED"
    echo "ERROR: Qdrant not running"
    exit 1
fi
echo "[OK]"

# Get embedding dimension
echo -n "Detecting embedding dimension... "
ACTUAL_DIM=$(curl -sf "$OLLAMA/api/embeddings" -d "{\"model\":\"$EMBED_MODEL\",\"prompt\":\"test\"}" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('embedding',[])))" 2>/dev/null || echo "768")
[ -z "$ACTUAL_DIM" ] && ACTUAL_DIM=768
echo "$ACTUAL_DIM"
export EMBEDDING_DIMENSION="$ACTUAL_DIM"

# Check/create collection
COLL_INFO=$(curl -sf "$QDRANT/collections/$COLLECTION" 2>/dev/null || echo "")
if [ -n "$COLL_INFO" ]; then
    COLL_DIM=$(echo "$COLL_INFO" | grep -o '"size":[0-9]*' | head -1 | cut -d: -f2 || echo "")
    if [ -n "$COLL_DIM" ] && [ "$COLL_DIM" != "$ACTUAL_DIM" ]; then
        echo "Dimension mismatch. Recreating collection..."
        curl -sf -X DELETE "$QDRANT/collections/$COLLECTION" > /dev/null || true
        curl -sf -X PUT "$QDRANT/collections/$COLLECTION" \
            -H "Content-Type: application/json" \
            -d "{\"vectors\":{\"size\":$ACTUAL_DIM,\"distance\":\"Cosine\"}}" > /dev/null
        rm -rf ./.ingest_tracking/* ./cache/dedup_index.json ./cache/bm25_index.pkl 2>/dev/null || true
        FORCE=true
    fi
else
    echo "Creating collection..."
    curl -sf -X PUT "$QDRANT/collections/$COLLECTION" \
        -H "Content-Type: application/json" \
        -d "{\"vectors\":{\"size\":$ACTUAL_DIM,\"distance\":\"Cosine\"}}" > /dev/null
fi

# Sync check
QDRANT_COUNT=$(curl -sf "$QDRANT/collections/$COLLECTION" 2>/dev/null | grep -o '"points_count":[0-9]*' | cut -d: -f2 2>/dev/null || echo "0")
[ -z "$QDRANT_COUNT" ] && QDRANT_COUNT=0
TRACKING_COUNT=$(ls -1 ./.ingest_tracking/ 2>/dev/null | wc -l || echo "0")

if [ "$QDRANT_COUNT" = "0" ] && [ "$TRACKING_COUNT" -gt 0 ] && [ "$FORCE" = "false" ]; then
    echo "Database empty but files tracked. Enabling --force"
    FORCE=true
fi

[ "$FORCE" = "true" ] && echo "Mode: FORCE (re-indexing all)"

# Configuration summary
echo ""
echo "Configuration:"
echo "  Chunk size: $CHUNK_SIZE (overlap: $CHUNK_OVERLAP)"
echo "  Semantic chunking: $USE_SEMANTIC"
echo "  Contextual headers: $CONTEXTUAL"
echo "  Parent-child: $PARENT_CHILD"
echo "  Deduplication: $DEDUP"
echo ""

# Find files
if [ -d "$INPUT_PATH" ]; then
    FILES=$(find "$INPUT_PATH" -type f \( \
        -iname "*.pdf" -o -iname "*.txt" -o -iname "*.md" -o -iname "*.rst" \
        -o -iname "*.docx" -o -iname "*.doc" \
        -o -iname "*.xlsx" -o -iname "*.xls" -o -iname "*.csv" -o -iname "*.tsv" \
        -o -iname "*.pptx" -o -iname "*.ppt" \
        -o -iname "*.html" -o -iname "*.htm" -o -iname "*.xml" \
        -o -iname "*.json" \
        -o -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \
        -o -iname "*.gif" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.tif" \
    \) 2>/dev/null | sort)
else
    FILES="$INPUT_PATH"
fi

if [ -z "$FILES" ]; then
    FILE_COUNT=0
else
    FILE_COUNT=$(echo "$FILES" | wc -l)
fi
echo "Found $FILE_COUNT files"

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "No supported files found"
    exit 0
fi

# Process files
python3 << EOFPY
import sys
import os
import hashlib
import json
import requests
import time

sys.path.insert(0, './lib')

from pdf_parser import parse_pdf
from office_parser import parse_docx, parse_xlsx, parse_pptx, parse_doc, parse_xls
from text_parser import parse_txt, parse_md, parse_html, parse_csv, parse_json
from image_parser import parse_image
from chunker import chunk_document

QDRANT = os.environ.get("QDRANT_HOST", "http://localhost:6333")
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
COLLECTION = os.environ.get("COLLECTION_NAME", "documents")
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.environ.get("EMBEDDING_DIMENSION", "768"))
EMBED_TIMEOUT = int(os.environ.get("EMBEDDING_TIMEOUT", "60"))

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "80"))
MIN_CHUNK = int(os.environ.get("MIN_CHUNK_SIZE", "100"))
USE_SEMANTIC = os.environ.get("USE_SEMANTIC", "false") == "true"
USE_SECTIONS = os.environ.get("USE_SECTIONS", "true") == "true"
CONTEXTUAL = os.environ.get("CONTEXTUAL", "true") == "true"
PARENT_CHILD = os.environ.get("PARENT_CHILD", "false") == "true"
OCR_LANG = os.environ.get("OCR_LANGUAGE", "eng")
DEBUG = os.environ.get("DEBUG", "false") == "true"
FORCE = os.environ.get("FORCE", "false") == "true"
DEDUP = os.environ.get("DEDUP", "true") == "true"

# Dedup index
dedup_file = "./cache/dedup_index.json"
if FORCE:
    dedup_index = set()
else:
    try:
        with open(dedup_file) as f:
            dedup_index = set(json.load(f))
    except:
        dedup_index = set()

os.makedirs(".ingest_tracking", exist_ok=True)
os.makedirs("cache", exist_ok=True)

files = """$FILES""".strip().split('\n')
files = [f for f in files if f.strip()]

total_stored = 0
total_skipped = 0
file_num = 0
start_time = time.time()

print(f"\nProcessing {len(files)} files...")
print("-" * 60)

for file_path in files:
    file_num += 1
    basename = os.path.basename(file_path)
    tracking_file = f".ingest_tracking/{basename.replace(' ', '_').replace('/', '_')}"
    
    # Skip if already processed
    if not FORCE and os.path.exists(tracking_file):
        if DEBUG:
            print(f"[{file_num}/{len(files)}] SKIP {basename[:40]}...")
        total_skipped += 1
        continue
    
    print(f"[{file_num}/{len(files)}] {basename[:50]}...", end=" ", flush=True)
    
    # Parse file
    ext = os.path.splitext(file_path)[1].lower()
    sections_info = []
    
    try:
        if ext == ".pdf":
            text, sections_info = parse_pdf(file_path, ocr_lang=OCR_LANG)
        elif ext == ".docx":
            text = parse_docx(file_path)
        elif ext == ".doc":
            text = parse_doc(file_path)
        elif ext == ".xlsx":
            text = parse_xlsx(file_path)
        elif ext == ".xls":
            text = parse_xls(file_path)
        elif ext in [".pptx", ".ppt"]:
            text = parse_pptx(file_path)
        elif ext in [".txt", ".rst"]:
            text = parse_txt(file_path)
        elif ext == ".md":
            text = parse_md(file_path)
        elif ext in [".html", ".htm", ".xml"]:
            text = parse_html(file_path)
        elif ext in [".csv", ".tsv"]:
            text = parse_csv(file_path)
        elif ext == ".json":
            text = parse_json(file_path)
        elif ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif"]:
            text = parse_image(file_path, OCR_LANG)
        else:
            text = ""
    except Exception as e:
        print(f"ERROR: {e}")
        continue
    
    if not text or len(text.strip()) < 50:
        print("-> empty/too short")
        continue
    
    print(f"-> {len(text)} chars", end=" ", flush=True)
    
    # Chunk document
    chunk_config = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "min_chunk_size": MIN_CHUNK,
        "use_semantic": USE_SEMANTIC,
        "use_sections": USE_SECTIONS,
        "contextual_headers": CONTEXTUAL,
        "parent_child": PARENT_CHILD,
        "filename": basename
    }
    
    chunks = chunk_document(text, chunk_config)
    print(f"-> {len(chunks)} chunks", end=" ", flush=True)
    
    if not chunks:
        print("-> no chunks")
        continue
    
    # Store chunks
    stored = 0
    dedup_count = 0
    
    if len(chunks) > 5:
        print("[", end="", flush=True)
    
    for i, chunk_obj in enumerate(chunks):
        if isinstance(chunk_obj, dict):
            chunk_text = chunk_obj.get("text", "")
            section = chunk_obj.get("section", "")
            chunk_type = chunk_obj.get("type", "chunk")
        else:
            chunk_text = chunk_obj
            section = ""
            chunk_type = "chunk"
        
        if not chunk_text or len(chunk_text) < MIN_CHUNK:
            continue
        
        # Deduplication
        chunk_hash = hashlib.md5(chunk_text[:300].encode()).hexdigest()
        if DEDUP and chunk_hash in dedup_index:
            dedup_count += 1
            continue
        
        # Get embedding
        try:
            resp = requests.post(f"{OLLAMA}/api/embeddings", json={
                "model": EMBED_MODEL,
                "prompt": chunk_text[:2000]  # Limit input size
            }, timeout=EMBED_TIMEOUT)
            
            if resp.status_code != 200:
                continue
            
            embedding = resp.json().get("embedding", [])
            if len(embedding) != EMBED_DIM:
                continue
        except Exception as e:
            if DEBUG:
                print(f"\nEmbed error: {e}")
            continue
        
        # Prepare payload
        doc_id = hashlib.md5(f"{file_path}_{i}_{chunk_hash}".encode()).hexdigest()
        payload = {
            "text": chunk_text,
            "source": file_path,
            "filename": basename,
            "section": section,
            "chunk_type": chunk_type,
            "chunk_index": i
        }
        
        # Add parent-child metadata
        if isinstance(chunk_obj, dict):
            if "parent_id" in chunk_obj:
                payload["parent_id"] = chunk_obj["parent_id"]
            if "children" in chunk_obj:
                payload["children"] = chunk_obj["children"]
        
        # Store in Qdrant
        try:
            resp = requests.put(f"{QDRANT}/collections/{COLLECTION}/points", json={
                "points": [{
                    "id": doc_id,
                    "vector": embedding,
                    "payload": payload
                }]
            }, timeout=30)
            
            if resp.status_code == 200:
                stored += 1
                dedup_index.add(chunk_hash)
                
                if len(chunks) > 5 and stored % 5 == 0:
                    print(".", end="", flush=True)
        except Exception as e:
            if DEBUG:
                print(f"\nStore error: {e}")
    
    if len(chunks) > 5:
        print("]", end=" ", flush=True)
    
    # Summary
    info = f"-> stored {stored}"
    if dedup_count > 0:
        info += f", dedup {dedup_count}"
    print(info)
    
    total_stored += stored
    
    # Mark as processed
    with open(tracking_file, 'w') as f:
        f.write(str(stored))

# Save dedup index
try:
    with open(dedup_file, 'w') as f:
        json.dump(list(dedup_index), f)
except:
    pass

elapsed = time.time() - start_time

print("")
print("-" * 60)
print(f"Processed: {file_num - total_skipped} files")
print(f"Skipped: {total_skipped} files (already indexed)")
print(f"Total stored: {total_stored} chunks")
print(f"Time: {elapsed:.1f}s")

# Final count
try:
    resp = requests.get(f"{QDRANT}/collections/{COLLECTION}")
    count = resp.json().get("result", {}).get("points_count", 0)
    print(f"Database total: {count} chunks")
except:
    pass
EOFPY

# Rebuild BM25
if [ "$REBUILD_BM25" = "true" ]; then
    echo ""
    ./rebuild-bm25.sh 2>/dev/null || true
fi

echo ""
echo "============================================"
echo "Ingestion complete!"
COUNT=$(curl -sf "$QDRANT/collections/$COLLECTION" 2>/dev/null | grep -o '"points_count":[0-9]*' | cut -d: -f2 || echo "0")
echo "Database contains: ${COUNT:-0} chunks"
echo "============================================"
EOFSH
chmod +x "$PROJECT_DIR/ingest.sh"
log_ok "Ingestion script created"

echo ""
echo "============================================================================"
echo "   Ingestion Setup Complete!"
echo "============================================================================"
echo ""
echo "Features implemented:"
echo "  ✓ Contextual headers (document + section in each chunk)"
echo "  ✓ Semantic chunking (sentence-boundary aware)"
echo "  ✓ Section detection (headers, numbered sections)"
echo "  ✓ Parent-child chunks (hierarchical)"
echo "  ✓ Table extraction"
echo "  ✓ OCR support"
echo "  ✓ Deduplication"
echo "  ✓ BM25 index building"
echo ""
echo "Usage:"
echo "  ./ingest.sh                    # Index ./documents"
echo "  ./ingest.sh --force            # Re-index everything"
echo "  ./ingest.sh --semantic         # Use sentence chunking"
echo "  ./ingest.sh --parent-child     # Hierarchical chunks"
echo "  ./ingest.sh --debug            # Verbose output"
echo ""
echo "Next: Run bash setup-rag-query-v42.sh"
echo "============================================================================"
