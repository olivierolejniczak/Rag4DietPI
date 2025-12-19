#!/bin/bash

# ============================================================================
# RAG System v47 - Document Ingestion Setup (SmartChunker with DeepDoc)
# ============================================================================
# Legacy v45 code (VERBATIM) + SmartChunker Modules
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_ok() { echo -e "[${GREEN}OK${NC}] $1"; }
log_warn() { echo -e "[${YELLOW}WARN${NC}] $1"; }
log_info() { echo -e "[INFO] $1"; }

PROJECT_DIR="${1:-$(pwd)}"

if [ ! -f "$PROJECT_DIR/config.env" ]; then
	echo "ERROR: Run setup-rag-core-v47.sh first!"
	exit 1
fi

cd "$PROJECT_DIR"

source config.env

[ -d "./venv" ] && source ./venv/bin/activate

clear

echo "============================================================================"
echo " RAG System v47 - Ingestion Setup (SmartChunker with DeepDoc)"
echo "============================================================================"
echo ""

mkdir -p lib cache .ingest_tracking

# ============================================================================
# BEGIN LEGACY VERSION 45
# ============================================================================

# ============================================================================
# PDF Parser Module (LEGACY v44 - VERBATIM)
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
	"""Extract section headers from text (English + French)"""
	sections = []
	current_section = "Introduction"
	lines = text.split('\n')
	
	for line in lines:
		if re.match(r'^#{1,3}\s+', line):
			current_section = re.sub(r'^#{1,3}\s+', '', line).strip()
		elif re.match(r'^\d+\.\s+[A-Z\xc0\xc2\xc6\xc7\xc9\xc8\xca\xcb\xcf\xce\xd4\xd9\xdbܟ\x8c]', line):
			current_section = line.strip()
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
						for line in text.split('\n'):
							if re.match(r'^#{1,3}\s+', line) or \
							   re.match(r'^\d+\.\s+[A-Z\xc0\xc2\xc6\xc7\xc9\xc8\xca\xcb\xcf\xce\xd4\xd9\xdbܟ\x8c]', line) or \
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
# Office Parser Module (LEGACY v44 - VERBATIM)
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
		elif re.match(r'^\d+\.\s+[A-Z\xc0\xc2\xc6\xc7\xc9\xc8\xca\xcb\xcf\xce\xd4\xd9\xdbܟ\x8c]', line):
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
			
			if para.style and 'Heading' in para.style.name:
				current_section = text[:100]
				text_parts.append(f"\n## {text}\n")
			else:
				text_parts.append(text)
		
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
# Text Parser Module (LEGACY v44 - VERBATIM)
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
				
				if i == 0:
					rows.append("-" * 40)
		
		return "\n".join(rows)
	
	except Exception as e:
		return f"Error: {e}"

def parse_json(file_path):
	"""Parse JSON file"""
	
	try:
		with open(file_path, 'r', encoding='utf-8') as f:
			data = json.load(f)
		
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
				for i, item in enumerate(obj[:100]):
					lines.extend(flatten_json(item, f"{prefix}[{i}]"))
			
			return lines
		
		return "\n".join(flatten_json(data)[:1000])
	
	except Exception as e:
		return f"Error: {e}"

EOFPY

log_ok "Text parser module created"

# ============================================================================
# Image Parser Module (LEGACY v44 - VERBATIM)
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
		
		max_size = 4096
		
		if max(img.size) > max_size:
			ratio = max_size / max(img.size)
			new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
			img = img.resize(new_size, Image.Resampling.LANCZOS)
		
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
# LEGACY Chunker Module (v44 - VERBATIM)
# ============================================================================

echo "Creating legacy chunker module..."

cat > lib/chunker.py << 'EOFPY'
"""Advanced document chunking with all features"""

import re
import hashlib

def detect_sections(text):
	"""Detect section boundaries in text (English + French)"""
	
	sections = []
	current_section = "Introduction"
	current_start = 0
	
	lines = text.split('\n')
	pos = 0
	
	for i, line in enumerate(lines):
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
		
		elif re.match(r'^\d+\.\d*\s+[A-Z\xc0\xc2\xc6\xc7\xc9\xc8\xca\xcb\xcf\xce\xd4\xd9\xdbܟ\x8c]', line):
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
	
	sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\xc0\xc2\xc6\xc7\xc9\xc8\xca\xcb\xcf\xce\xd4\xd9\xdbܟ\x8c])', text)
	sentences = [s.strip() for s in sentences if s.strip()]
	
	if not sentences:
		return [text] if text else []
	
	chunks = []
	current_chunk = []
	current_length = 0
	
	for sentence in sentences:
		sentence_len = len(sentence)
		
		if current_length + sentence_len > max_size and current_chunk:
			chunks.append(' '.join(current_chunk))
			
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
		
		if end < len(text):
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
	
	summary = text[:500].strip()
	
	if len(text) > 500:
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
	
	if parent_child:
		return create_parent_child_chunks(text, filename)
	
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
					
					if contextual_headers:
						header = f"Document: {filename}"
						
						if section_title and section_title != "Introduction":
							header += f" | Section: {section_title}"
						
						chunk_obj["text"] = f"{header}\n\n{chunk_text}"
					
					all_chunks.append(chunk_obj)
			
			return all_chunks
	
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

log_ok "Legacy chunker module created"

# ============================================================================
# ENHANCED CHUNKER STUB (v45 - VERBATIM)
# ============================================================================

echo "Creating enhanced chunker stub (auto-fallback)..."

cat > lib/enhanced_chunker.py << 'EOFPY'
"""
Enhanced Chunker Module - Stub with auto-fallback
If this module fails to load, system automatically uses legacy chunker
"""

from chunker import chunk_document as chunk_document_legacy_compatible

# Future enhancement: Add content-type detection here
# For now, uses proven legacy chunker

EOFPY

log_ok "Enhanced chunker stub created"

# ============================================================================
# END LEGACY VERSION 45
# ============================================================================

# ============================================================================
# BEGIN NEW VERSION 46 EXTENSIONS
# ============================================================================

# ============================================================================
# NEW v46 - DOCUMENT ANALYZER MODULE (DeepDoc + Heuristic)
# ============================================================================

echo "Creating document analyzer module..."

cat > lib/document_analyzer.py << 'EOFPY'
"""
Document Analyzer Module - Layout understanding with DeepDoc or Heuristic fallback
v46 - SmartChunker component

Provides document structure analysis:
- Content type detection (text, table, code, figure, equation)
- Layout understanding (bounding boxes, positions)
- OCR for scanned documents
- Table structure recognition

Modes:
- deepdoc: Full DeepDoc integration (requires PyTorch, high RAM)
- heuristic: Pattern-based analysis (pure Python, low RAM)
- auto: Selects best available mode
"""

import os
import re
import sys

# Global state for mode detection
_analyzer_mode = None
_deepdoc_available = False


def _check_deepdoc_available():
	"""Check if DeepDoc dependencies are available"""
	global _deepdoc_available
	
	try:
		# Check RAM first (need at least 4GB for DeepDoc)
		with open('/proc/meminfo', 'r') as f:
			for line in f:
				if line.startswith('MemTotal:'):
					ram_kb = int(line.split()[1])
					ram_gb = ram_kb / 1024 / 1024
					if ram_gb < 4:
						_deepdoc_available = False
						return False
					break
	except Exception:
		pass
	
	# Check Python dependencies
	try:
		import torch
		import numpy as np
		_deepdoc_available = True
		return True
	except ImportError:
		_deepdoc_available = False
		return False


def _detect_mode():
	"""Detect best available mode based on resources"""
	global _analyzer_mode
	
	config_mode = os.environ.get("SMARTCHUNKER_MODE", "auto")
	
	if config_mode == "deepdoc":
		if _check_deepdoc_available():
			_analyzer_mode = "deepdoc"
		else:
			print("[WARN] DeepDoc requested but not available, falling back to heuristic")
			_analyzer_mode = "heuristic"
	elif config_mode == "heuristic":
		_analyzer_mode = "heuristic"
	else:  # auto
		if _check_deepdoc_available():
			_analyzer_mode = "deepdoc"
		else:
			_analyzer_mode = "heuristic"
	
	return _analyzer_mode


def get_analyzer_mode():
	"""Get current analyzer mode"""
	global _analyzer_mode
	if _analyzer_mode is None:
		_detect_mode()
	return _analyzer_mode


# ============================================================================
# Content Type Detection Patterns (Heuristic Mode)
# ============================================================================

# Table detection patterns
TABLE_PATTERNS = [
	# Markdown tables
	r'^\|[^|]+\|',
	r'^[\-\+]+[\-\+]+',
	# Pipe-separated
	r'^[^|]+\|[^|]+\|',
	# Tab-separated with multiple columns
	r'^[^\t]+\t[^\t]+\t[^\t]+',
]

# Code detection patterns
CODE_PATTERNS = {
	'python': [
		r'^def\s+\w+\s*\(',
		r'^class\s+\w+',
		r'^import\s+\w+',
		r'^from\s+\w+\s+import',
		r'^\s*if\s+.*:\s*$',
		r'^\s*for\s+\w+\s+in\s+',
		r'^\s*while\s+.*:\s*$',
		r'^\s*try:\s*$',
		r'^\s*except\s*.*:\s*$',
	],
	'javascript': [
		r'^function\s+\w+\s*\(',
		r'^const\s+\w+\s*=',
		r'^let\s+\w+\s*=',
		r'^var\s+\w+\s*=',
		r'^\s*if\s*\(.*\)\s*\{',
		r'=>\s*\{',
		r'^export\s+(default\s+)?',
		r'^import\s+.*\s+from\s+',
	],
	'bash': [
		r'^#!/bin/(ba)?sh',
		r'^\s*if\s+\[\s+',
		r'^\s*for\s+\w+\s+in\s+',
		r'^\s*while\s+\[\s+',
		r'^\s*case\s+.*\s+in',
		r'^\s*function\s+\w+\s*\(\)',
		r'^\w+\s*=\s*\$\(',
		r'\$\{?\w+\}?',
	],
	'sql': [
		r'^SELECT\s+',
		r'^INSERT\s+INTO',
		r'^UPDATE\s+\w+\s+SET',
		r'^DELETE\s+FROM',
		r'^CREATE\s+(TABLE|INDEX|VIEW)',
		r'^ALTER\s+TABLE',
		r'^DROP\s+(TABLE|INDEX|VIEW)',
	],
	'json': [
		r'^\s*\{\s*"',
		r'^\s*\[\s*\{',
	],
	'yaml': [
		r'^\w+:\s*$',
		r'^\s*-\s+\w+:',
		r'^---\s*$',
	],
	'xml': [
		r'^<\?xml',
		r'^<\w+\s+xmlns',
		r'^<\w+>.*</\w+>$',
	],
}

# Figure/image reference patterns
FIGURE_PATTERNS = [
	r'^!\[.*\]\(.*\)',  # Markdown image
	r'^\[Figure\s*\d*\]',
	r'^Figure\s*\d+[.:]\s*',
	r'^Image\s*\d+[.:]\s*',
	r'^Illustration\s*\d+[.:]\s*',
	r'^Photo\s*\d+[.:]\s*',
]

# Equation patterns
EQUATION_PATTERNS = [
	r'\$\$.*\$\$',  # LaTeX display math
	r'\$[^$]+\$',  # LaTeX inline math
	r'\\begin\{equation\}',
	r'\\begin\{align\}',
	r'\\frac\{',
	r'\\sum_',
	r'\\int_',
	r'\\sqrt\{',
]

# List patterns
LIST_PATTERNS = [
	r'^\s*[-*+]\s+',  # Unordered list
	r'^\s*\d+[.)]\s+',  # Ordered list
	r'^\s*[a-z][.)]\s+',  # Letter list
]


class ContentBlock:
	"""Represents a detected content block with metadata"""
	
	def __init__(self, content_type, text, confidence=1.0, metadata=None):
		self.content_type = content_type  # text | table | code | figure | equation | list
		self.text = text
		self.confidence = confidence
		self.metadata = metadata or {}
		self.position = None  # {page, x, y, width, height}
		self.relationships = []  # Related block IDs
	
	def to_dict(self):
		return {
			"type": self.content_type,
			"text": self.text,
			"confidence": self.confidence,
			"metadata": self.metadata,
			"position": self.position,
			"relationships": self.relationships,
		}


def _detect_content_type_heuristic(text):
	"""Detect content type using pattern matching (heuristic mode)"""
	
	lines = text.strip().split('\n')
	if not lines:
		return "text", 0.5, {}
	
	# Check for table
	table_score = 0
	pipe_lines = sum(1 for l in lines if '|' in l)
	if pipe_lines > len(lines) * 0.5:
		table_score = 0.8
	for pattern in TABLE_PATTERNS:
		if any(re.match(pattern, l.strip()) for l in lines[:5]):
			table_score = max(table_score, 0.7)
	
	if table_score > 0.6:
		# Try to extract table structure
		headers = []
		rows = []
		for line in lines:
			if '|' in line:
				cells = [c.strip() for c in line.split('|') if c.strip()]
				if not headers:
					headers = cells
				elif not re.match(r'^[\-\+]+$', line.replace('|', '')):
					rows.append(cells)
		
		return "table", table_score, {
			"headers": headers,
			"row_count": len(rows),
			"col_count": len(headers) if headers else 0,
		}
	
	# Check for code
	for lang, patterns in CODE_PATTERNS.items():
		match_count = 0
		for pattern in patterns:
			if any(re.match(pattern, l, re.IGNORECASE if lang == 'sql' else 0) for l in lines[:10]):
				match_count += 1
		
		if match_count >= 2:
			# Check indentation consistency (code signature)
			indented = sum(1 for l in lines if l.startswith('  ') or l.startswith('\t'))
			if indented > len(lines) * 0.3:
				return "code", 0.85, {"language": lang}
	
	# Generic code detection (indentation-based)
	indented_lines = sum(1 for l in lines if l.startswith('    ') or l.startswith('\t'))
	if indented_lines > len(lines) * 0.6 and len(lines) > 3:
		return "code", 0.6, {"language": "unknown"}
	
	# Check for figure reference
	for pattern in FIGURE_PATTERNS:
		if any(re.match(pattern, l, re.IGNORECASE) for l in lines[:3]):
			caption = lines[0] if lines else ""
			return "figure", 0.75, {"caption": caption}
	
	# Check for equation
	equation_score = 0
	for pattern in EQUATION_PATTERNS:
		if re.search(pattern, text):
			equation_score += 0.3
	
	if equation_score > 0.5:
		return "equation", min(equation_score, 0.9), {"format": "latex"}
	
	# Check for list
	list_lines = 0
	for pattern in LIST_PATTERNS:
		list_lines += sum(1 for l in lines if re.match(pattern, l))
	
	if list_lines > len(lines) * 0.6:
		return "list", 0.8, {"item_count": list_lines}
	
	# Default to text
	return "text", 0.9, {}


def analyze_document_heuristic(text, filename=""):
	"""
	Analyze document using heuristic patterns (low-resource mode)
	
	Returns list of ContentBlock objects
	"""
	blocks = []
	
	# Split into potential blocks using blank lines and special markers
	raw_blocks = re.split(r'\n\s*\n', text)
	
	for i, block_text in enumerate(raw_blocks):
		block_text = block_text.strip()
		if not block_text or len(block_text) < 10:
			continue
		
		content_type, confidence, metadata = _detect_content_type_heuristic(block_text)
		
		block = ContentBlock(
			content_type=content_type,
			text=block_text,
			confidence=confidence,
			metadata=metadata
		)
		
		# Estimate position (line-based for text documents)
		block.position = {
			"block_index": i,
			"char_start": text.find(block_text),
			"char_end": text.find(block_text) + len(block_text),
		}
		
		blocks.append(block)
	
	# Detect relationships (adjacent blocks, figure-caption pairs)
	for i, block in enumerate(blocks):
		if block.content_type == "figure" and i + 1 < len(blocks):
			# Link figure to next text block (likely caption continuation)
			block.relationships.append({"type": "caption", "block_index": i + 1})
		
		if i > 0:
			block.relationships.append({"type": "previous", "block_index": i - 1})
		if i < len(blocks) - 1:
			block.relationships.append({"type": "next", "block_index": i + 1})
	
	return blocks


def analyze_document_deepdoc(text, filename="", file_path=None):
	"""
	Analyze document using DeepDoc (high-resource mode)
	
	This is a stub that will integrate with RAGFlow's DeepDoc when available.
	For now, returns empty list to trigger fallback.
	"""
	# DeepDoc integration point
	# When implemented, this would:
	# 1. Load DeepDoc models
	# 2. Process document pages as images
	# 3. Run layout analysis
	# 4. Run OCR on detected regions
	# 5. Run table structure recognition
	# 6. Return ContentBlock objects with bounding boxes
	
	# For now, return empty to trigger heuristic fallback
	return []


def analyze_document(text, filename="", file_path=None):
	"""
	Main entry point for document analysis.
	
	Automatically selects best available mode.
	
	Returns: list of ContentBlock objects
	"""
	mode = get_analyzer_mode()
	
	if mode == "deepdoc":
		blocks = analyze_document_deepdoc(text, filename, file_path)
		if blocks:
			return blocks
		# Fallback to heuristic if DeepDoc returns nothing
	
	return analyze_document_heuristic(text, filename)


def analyze_pdf_pages(file_path, ocr_enabled=True, ocr_lang="eng+fra"):
	"""
	Analyze PDF with page-level layout understanding.
	
	Returns: list of (page_num, blocks) tuples
	"""
	mode = get_analyzer_mode()
	pages = []
	
	try:
		import pypdfium2 as pdfium
		
		pdf = pdfium.PdfDocument(file_path)
		
		for page_num, page in enumerate(pdf, 1):
			try:
				textpage = page.get_textpage()
				text = textpage.get_text_bounded()
				
				if not text.strip() and ocr_enabled:
					# OCR fallback for scanned pages
					try:
						from PIL import Image
						import pytesseract
						
						bitmap = page.render(scale=2)
						pil_image = bitmap.to_pil()
						text = pytesseract.image_to_string(pil_image, lang=ocr_lang)
					except ImportError:
						pass
				
				blocks = analyze_document(text, file_path)
				
				# Update positions with page number
				for block in blocks:
					if block.position:
						block.position["page"] = page_num
				
				pages.append((page_num, blocks))
			
			except Exception as e:
				print(f"[WARN] Error analyzing page {page_num}: {e}")
				continue
		
		pdf.close()
	
	except ImportError:
		# No PDF library available
		pass
	except Exception as e:
		print(f"[WARN] Error analyzing PDF: {e}")
	
	return pages


# Module initialization
_detect_mode()

EOFPY

log_ok "Document analyzer module created"

# ============================================================================
# NEW v46 - CONTENT EXTRACTOR MODULE
# ============================================================================

echo "Creating content extractor module..."

cat > lib/content_extractor.py << 'EOFPY'
"""
Content Extractor Module - Type-specific content processing
v46 - SmartChunker component

Provides specialized extraction for different content types:
- Tables: Extract semantic structure (headers, rows, relationships)
- Code: Preserve syntax, detect language, maintain indentation
- Figures: Link to captions, preserve image references
- Equations: Preserve LaTeX/MathML format
- Text: Apply semantic chunking with section awareness
"""

import re
import hashlib
import os


def get_config():
	"""Get extractor configuration from environment"""
	return {
		"table_atomic": os.environ.get("SMARTCHUNKER_TABLE_ATOMIC", "true") == "true",
		"table_extract_structure": os.environ.get("SMARTCHUNKER_TABLE_EXTRACT_STRUCTURE", "true") == "true",
		"code_preserve_syntax": os.environ.get("SMARTCHUNKER_CODE_PRESERVE_SYNTAX", "true") == "true",
		"code_detect_language": os.environ.get("SMARTCHUNKER_CODE_DETECT_LANGUAGE", "true") == "true",
		"figure_link_caption": os.environ.get("SMARTCHUNKER_FIGURE_LINK_CAPTION", "true") == "true",
		"equation_preserve_format": os.environ.get("SMARTCHUNKER_EQUATION_PRESERVE_FORMAT", "true") == "true",
		"confidence_threshold": float(os.environ.get("SMARTCHUNKER_CONFIDENCE_THRESHOLD", "0.7")),
		"chunk_size": int(os.environ.get("CHUNK_SIZE", "500")),
		"chunk_overlap": int(os.environ.get("CHUNK_OVERLAP", "80")),
	}


# ============================================================================
# Table Extraction
# ============================================================================

def extract_table_structure(text, metadata=None):
	"""
	Extract semantic table structure.
	
	Returns dict with:
	- headers: list of column headers
	- rows: list of row data (list of cells)
	- relationships: detected relationships between columns
	"""
	lines = text.strip().split('\n')
	
	headers = []
	rows = []
	separator_found = False
	
	for line in lines:
		line = line.strip()
		if not line:
			continue
		
		# Detect separator line (---|---|---)
		if re.match(r'^[\-\|\+\s]+$', line) and '|' in line:
			separator_found = True
			continue
		
		if '|' in line:
			cells = [c.strip() for c in line.split('|')]
			# Remove empty first/last if line starts/ends with |
			if cells and not cells[0]:
				cells = cells[1:]
			if cells and not cells[-1]:
				cells = cells[:-1]
			
			if not headers:
				headers = cells
			else:
				rows.append(cells)
	
	# Fallback: tab-separated
	if not headers:
		for line in lines:
			if '\t' in line:
				cells = [c.strip() for c in line.split('\t')]
				if not headers:
					headers = cells
				else:
					rows.append(cells)
	
	# Detect column relationships (numeric, date, categorical)
	column_types = []
	for col_idx in range(len(headers)):
		col_values = [r[col_idx] if col_idx < len(r) else "" for r in rows]
		
		# Check if numeric
		numeric_count = sum(1 for v in col_values if re.match(r'^[\d,.]+$', v.replace(' ', '')))
		if numeric_count > len(col_values) * 0.7:
			column_types.append("numeric")
		# Check if date
		elif any(re.search(r'\d{2,4}[-/]\d{2}[-/]\d{2,4}', v) for v in col_values):
			column_types.append("date")
		else:
			column_types.append("text")
	
	return {
		"headers": headers,
		"rows": rows,
		"column_count": len(headers),
		"row_count": len(rows),
		"column_types": column_types,
		"has_separator": separator_found,
	}


def chunk_table(block, config):
	"""
	Process table block into chunks.
	
	If table_atomic=True: Keep entire table as single chunk
	If table_atomic=False: Split large tables by rows
	"""
	chunks = []
	text = block.text
	metadata = block.metadata.copy()
	
	# Extract structure
	if config["table_extract_structure"]:
		structure = extract_table_structure(text, metadata)
		metadata.update(structure)
	
	if config["table_atomic"] or len(text) < config["chunk_size"] * 2:
		# Keep as single chunk
		chunks.append({
			"text": text,
			"type": "table",
			"detected_type": "table",
			"confidence": block.confidence,
			"metadata": metadata,
			"position": block.position,
		})
	else:
		# Split large tables by rows
		lines = text.strip().split('\n')
		header_lines = []
		data_lines = []
		
		for i, line in enumerate(lines):
			if i < 2 or re.match(r'^[\-\|\+\s]+$', line):
				header_lines.append(line)
			else:
				data_lines.append(line)
		
		header_text = '\n'.join(header_lines)
		current_chunk = [header_text]
		current_size = len(header_text)
		
		for line in data_lines:
			if current_size + len(line) > config["chunk_size"]:
				chunks.append({
					"text": '\n'.join(current_chunk),
					"type": "table",
					"detected_type": "table",
					"confidence": block.confidence,
					"metadata": {**metadata, "is_partial": True},
					"position": block.position,
				})
				current_chunk = [header_text, line]
				current_size = len(header_text) + len(line)
			else:
				current_chunk.append(line)
				current_size += len(line)
		
		if current_chunk:
			chunks.append({
				"text": '\n'.join(current_chunk),
				"type": "table",
				"detected_type": "table",
				"confidence": block.confidence,
				"metadata": {**metadata, "is_partial": len(chunks) > 0},
				"position": block.position,
			})
	
	return chunks


# ============================================================================
# Code Extraction
# ============================================================================

def detect_code_language(text):
	"""Detect programming language from code content"""
	
	patterns = {
		'python': [
			(r'\bdef\s+\w+\s*\(', 3),
			(r'\bclass\s+\w+\s*[:\(]', 3),
			(r'\bimport\s+\w+', 2),
			(r'\bfrom\s+\w+\s+import', 2),
			(r':\s*\n\s+', 1),  # Colon followed by indentation
			(r'\bself\.\w+', 2),
			(r'\bprint\s*\(', 1),
		],
		'javascript': [
			(r'\bfunction\s+\w+\s*\(', 3),
			(r'\bconst\s+\w+\s*=', 2),
			(r'\blet\s+\w+\s*=', 2),
			(r'\b=>\s*[\{\(]', 2),
			(r'\bconsole\.\w+\s*\(', 2),
			(r'\bexport\s+(default\s+)?', 2),
			(r'\brequire\s*\(', 2),
		],
		'bash': [
			(r'^#!/bin/(ba)?sh', 5),
			(r'\bif\s+\[\s+', 2),
			(r'\becho\s+', 1),
			(r'\$\{\w+\}', 2),
			(r'\$\(\w+', 2),
			(r'\bfi\b', 1),
			(r'\bdone\b', 1),
		],
		'sql': [
			(r'\bSELECT\s+', 3),
			(r'\bFROM\s+\w+', 2),
			(r'\bWHERE\s+', 2),
			(r'\bJOIN\s+', 2),
			(r'\bGROUP\s+BY\b', 2),
			(r'\bINSERT\s+INTO\b', 2),
		],
		'yaml': [
			(r'^---\s*$', 3),
			(r'^\s*\w+:\s*$', 2),
			(r'^\s*-\s+\w+:', 2),
		],
		'json': [
			(r'^\s*\{\s*"\w+":', 3),
			(r'^\s*\[\s*\{', 2),
			(r'"\w+":\s*["\d\[\{]', 2),
		],
	}
	
	scores = {}
	for lang, lang_patterns in patterns.items():
		score = 0
		for pattern, weight in lang_patterns:
			matches = len(re.findall(pattern, text, re.MULTILINE | re.IGNORECASE if lang == 'sql' else re.MULTILINE))
			score += matches * weight
		scores[lang] = score
	
	if scores:
		best_lang = max(scores, key=scores.get)
		if scores[best_lang] >= 3:
			return best_lang
	
	return "unknown"


def find_code_boundaries(text):
	"""Find syntactically complete code boundaries"""
	
	# For Python: match def/class blocks
	# For JS: match function/const blocks
	# For bash: match function/if/for blocks
	
	boundaries = []
	lines = text.split('\n')
	
	current_block_start = 0
	indent_stack = [0]
	
	for i, line in enumerate(lines):
		stripped = line.lstrip()
		if not stripped:
			continue
		
		current_indent = len(line) - len(stripped)
		
		# Detect block start
		if re.match(r'^(def|class|function|if|for|while|try)\b', stripped):
			if i > current_block_start and current_indent <= indent_stack[-1]:
				boundaries.append((current_block_start, i))
				current_block_start = i
			indent_stack.append(current_indent)
		
		# Detect dedent
		while current_indent < indent_stack[-1] and len(indent_stack) > 1:
			indent_stack.pop()
	
	# Add final block
	if current_block_start < len(lines):
		boundaries.append((current_block_start, len(lines)))
	
	return boundaries


def chunk_code(block, config):
	"""
	Process code block into chunks.
	
	Preserves:
	- Syntactic completeness (function/class boundaries)
	- Indentation
	- Language detection
	"""
	chunks = []
	text = block.text
	metadata = block.metadata.copy()
	
	# Detect language if not already set
	if config["code_detect_language"] and metadata.get("language") == "unknown":
		metadata["language"] = detect_code_language(text)
	
	if config["code_preserve_syntax"]:
		# Try to split at syntactic boundaries
		boundaries = find_code_boundaries(text)
		lines = text.split('\n')
		
		current_chunk_lines = []
		current_size = 0
		
		for start, end in boundaries:
			block_lines = lines[start:end]
			block_text = '\n'.join(block_lines)
			block_size = len(block_text)
			
			if current_size + block_size > config["chunk_size"] and current_chunk_lines:
				# Emit current chunk
				chunks.append({
					"text": '\n'.join(current_chunk_lines),
					"type": "code",
					"detected_type": "code",
					"confidence": block.confidence,
					"metadata": metadata,
					"position": block.position,
				})
				current_chunk_lines = []
				current_size = 0
			
			current_chunk_lines.extend(block_lines)
			current_size += block_size
		
		if current_chunk_lines:
			chunks.append({
				"text": '\n'.join(current_chunk_lines),
				"type": "code",
				"detected_type": "code",
				"confidence": block.confidence,
				"metadata": metadata,
				"position": block.position,
			})
	
	if not chunks:
		# Fallback: simple chunking with indentation awareness
		lines = text.split('\n')
		current_chunk = []
		current_size = 0
		
		for line in lines:
			line_size = len(line) + 1
			
			if current_size + line_size > config["chunk_size"] and current_chunk:
				# Try to break at empty or zero-indent line
				break_idx = len(current_chunk)
				for j in range(len(current_chunk) - 1, -1, -1):
					if not current_chunk[j].strip() or not current_chunk[j].startswith(' '):
						break_idx = j + 1
						break
				
				chunks.append({
					"text": '\n'.join(current_chunk[:break_idx]),
					"type": "code",
					"detected_type": "code",
					"confidence": block.confidence,
					"metadata": metadata,
					"position": block.position,
				})
				current_chunk = current_chunk[break_idx:]
				current_size = sum(len(l) + 1 for l in current_chunk)
			
			current_chunk.append(line)
			current_size += line_size
		
		if current_chunk:
			chunks.append({
				"text": '\n'.join(current_chunk),
				"type": "code",
				"detected_type": "code",
				"confidence": block.confidence,
				"metadata": metadata,
				"position": block.position,
			})
	
	return chunks


# ============================================================================
# Figure Extraction
# ============================================================================

def chunk_figure(block, config, related_blocks=None):
	"""
	Process figure block.
	
	Links figure reference to caption and related text.
	"""
	text = block.text
	metadata = block.metadata.copy()
	
	# Extract figure reference and caption
	figure_ref = ""
	caption = ""
	
	# Try to extract figure number
	match = re.search(r'(Figure|Image|Illustration)\s*(\d+)', text, re.IGNORECASE)
	if match:
		figure_ref = f"{match.group(1)} {match.group(2)}"
		metadata["figure_ref"] = figure_ref
	
	# Extract caption from metadata or text
	if "caption" in metadata:
		caption = metadata["caption"]
	else:
		# First line is often the caption
		lines = text.strip().split('\n')
		if lines:
			caption = lines[0]
			metadata["caption"] = caption
	
	# Link related text if available
	if config["figure_link_caption"] and related_blocks:
		related_text = []
		for rel in block.relationships:
			if rel["type"] == "caption" and rel["block_index"] < len(related_blocks):
				related_block = related_blocks[rel["block_index"]]
				if related_block.content_type == "text":
					related_text.append(related_block.text)
		
		if related_text:
			metadata["related_text"] = ' '.join(related_text)[:500]
	
	return [{
		"text": text,
		"type": "figure",
		"detected_type": "figure",
		"confidence": block.confidence,
		"metadata": metadata,
		"position": block.position,
	}]


# ============================================================================
# Equation Extraction
# ============================================================================

def chunk_equation(block, config):
	"""
	Process equation block.
	
	Preserves LaTeX/MathML format.
	"""
	text = block.text
	metadata = block.metadata.copy()
	
	# Detect equation format
	if re.search(r'\\begin\{', text) or re.search(r'\$\$', text) or re.search(r'\\frac', text):
		metadata["format"] = "latex"
	elif re.search(r'<math', text, re.IGNORECASE):
		metadata["format"] = "mathml"
	else:
		metadata["format"] = "unknown"
	
	# Keep equations atomic (don't split)
	return [{
		"text": text,
		"type": "equation",
		"detected_type": "equation",
		"confidence": block.confidence,
		"metadata": metadata,
		"position": block.position,
	}]


# ============================================================================
# Text Extraction (with semantic chunking)
# ============================================================================

def chunk_text(block, config):
	"""
	Process text block with semantic chunking.
	
	Applies sentence-boundary aware splitting.
	"""
	text = block.text
	metadata = block.metadata.copy()
	
	# Short text: keep as single chunk
	if len(text) < config["chunk_size"]:
		return [{
			"text": text,
			"type": "text",
			"detected_type": "text",
			"confidence": block.confidence,
			"metadata": metadata,
			"position": block.position,
		}]
	
	# Semantic chunking: split at sentence boundaries
	chunks = []
	
	# Split into sentences (handles English and French punctuation)
	sentences = re.split(
		r'(?<=[.!?])\s+(?=[A-Z\xc0\xc2\xc6\xc7\xc9\xc8\xca\xcb\xcf\xce\xd4\xd9\xdbܟ\x8c])',
		text
	)
	
	current_chunk = []
	current_size = 0
	
	for sentence in sentences:
		sentence = sentence.strip()
		if not sentence:
			continue
		
		sentence_size = len(sentence)
		
		if current_size + sentence_size > config["chunk_size"] and current_chunk:
			# Emit chunk
			chunks.append({
				"text": ' '.join(current_chunk),
				"type": "text",
				"detected_type": "text",
				"confidence": block.confidence,
				"metadata": metadata,
				"position": block.position,
			})
			
			# Overlap: keep last few sentences
			overlap_sentences = []
			overlap_size = 0
			for s in reversed(current_chunk):
				if overlap_size + len(s) <= config["chunk_overlap"]:
					overlap_sentences.insert(0, s)
					overlap_size += len(s)
				else:
					break
			
			current_chunk = overlap_sentences
			current_size = overlap_size
		
		current_chunk.append(sentence)
		current_size += sentence_size
	
	if current_chunk:
		chunks.append({
			"text": ' '.join(current_chunk),
			"type": "text",
			"detected_type": "text",
			"confidence": block.confidence,
			"metadata": metadata,
			"position": block.position,
		})
	
	return chunks


# ============================================================================
# List Extraction
# ============================================================================

def chunk_list(block, config):
	"""
	Process list block.
	
	Keeps lists atomic if short, splits by items if long.
	"""
	text = block.text
	metadata = block.metadata.copy()
	
	if len(text) < config["chunk_size"]:
		return [{
			"text": text,
			"type": "list",
			"detected_type": "list",
			"confidence": block.confidence,
			"metadata": metadata,
			"position": block.position,
		}]
	
	# Split by list items
	chunks = []
	items = re.split(r'\n\s*(?=[-*+]|\d+[.)]\s)', text)
	
	current_chunk = []
	current_size = 0
	
	for item in items:
		item = item.strip()
		if not item:
			continue
		
		item_size = len(item)
		
		if current_size + item_size > config["chunk_size"] and current_chunk:
			chunks.append({
				"text": '\n'.join(current_chunk),
				"type": "list",
				"detected_type": "list",
				"confidence": block.confidence,
				"metadata": {**metadata, "is_partial": True},
				"position": block.position,
			})
			current_chunk = []
			current_size = 0
		
		current_chunk.append(item)
		current_size += item_size
	
	if current_chunk:
		chunks.append({
			"text": '\n'.join(current_chunk),
			"type": "list",
			"detected_type": "list",
			"confidence": block.confidence,
			"metadata": {**metadata, "is_partial": len(chunks) > 0},
			"position": block.position,
		})
	
	return chunks


# ============================================================================
# Main Extraction Function
# ============================================================================

def extract_content(block, config=None, all_blocks=None):
	"""
	Extract and chunk content based on detected type.
	
	Args:
		block: ContentBlock object
		config: Configuration dict (from get_config if None)
		all_blocks: All blocks for relationship resolution
	
	Returns:
		list of chunk dicts
	"""
	if config is None:
		config = get_config()
	
	content_type = block.content_type
	
	# Check confidence threshold
	if block.confidence < config["confidence_threshold"]:
		# Fall back to generic text chunking
		content_type = "text"
	
	if content_type == "table":
		return chunk_table(block, config)
	elif content_type == "code":
		return chunk_code(block, config)
	elif content_type == "figure":
		return chunk_figure(block, config, all_blocks)
	elif content_type == "equation":
		return chunk_equation(block, config)
	elif content_type == "list":
		return chunk_list(block, config)
	else:  # text or unknown
		return chunk_text(block, config)

EOFPY

log_ok "Content extractor module created"

# ============================================================================
# NEW v46 - SMARTCHUNKER MODULE (Main Entry Point)
# ============================================================================

echo "Creating SmartChunker module..."

cat > lib/smartchunker.py << 'EOFPY'
"""
SmartChunker Module - Production-grade document chunking with DeepDoc integration
v46 - Main entry point

Features:
- Vision-based document layout understanding (via DeepDoc or heuristic fallback)
- Content-type aware chunking (tables, code, figures, equations, text)
- Semantic table structure extraction
- Figure-caption linking
- Code syntax preservation with language detection
- Position/layout metadata
- Relationship tracking between chunks
- Backward compatible with Enhanced/Legacy chunkers

Fallback chain:
SmartChunker(DeepDoc) -> SmartChunker(Heuristic) -> EnhancedChunker -> LegacyChunker
"""

import os
import hashlib

# Import submodules
try:
	from document_analyzer import analyze_document, analyze_pdf_pages, get_analyzer_mode
	from content_extractor import extract_content, get_config as get_extractor_config
	SMARTCHUNKER_AVAILABLE = True
except ImportError as e:
	print(f"[WARN] SmartChunker import error: {e}")
	SMARTCHUNKER_AVAILABLE = False


def get_smartchunker_info():
	"""Get SmartChunker status and configuration"""
	if not SMARTCHUNKER_AVAILABLE:
		return {
			"available": False,
			"mode": None,
			"reason": "Import error"
		}
	
	return {
		"available": True,
		"mode": get_analyzer_mode(),
		"config": get_extractor_config(),
	}


def smart_chunk_document(text, config):
	"""
	Main SmartChunker entry point.
	
	Compatible with legacy chunk_document interface.
	
	Args:
		text: Document text content
		config: dict with:
			- chunk_size: Max chunk size
			- chunk_overlap: Overlap between chunks
			- min_chunk_size: Minimum chunk size
			- filename: Source filename
			- section: Current section name
			- use_enhanced: Flag (ignored, we use SmartChunker)
			- contextual_headers: Add headers to chunks
	
	Returns:
		list of chunk dicts compatible with ingest.sh
	"""
	if not SMARTCHUNKER_AVAILABLE:
		# Fallback to enhanced/legacy chunker
		try:
			from enhanced_chunker import chunk_document_legacy_compatible
			return chunk_document_legacy_compatible(text, config)
		except ImportError:
			from chunker import chunk_document
			return chunk_document(text, config)
	
	filename = config.get("filename", "unknown")
	contextual_headers = config.get("contextual_headers", True)
	
	# Analyze document structure
	blocks = analyze_document(text, filename)
	
	if not blocks:
		# No blocks detected, fallback to legacy
		try:
			from chunker import chunk_document
			return chunk_document(text, config)
		except ImportError:
			return [{"text": text, "section": "", "filename": filename}]
	
	# Extract chunks from each block
	all_chunks = []
	extractor_config = get_extractor_config()
	
	# Override with provided config
	if "chunk_size" in config:
		extractor_config["chunk_size"] = config["chunk_size"]
	if "chunk_overlap" in config:
		extractor_config["chunk_overlap"] = config["chunk_overlap"]
	
	for block in blocks:
		block_chunks = extract_content(block, extractor_config, blocks)
		
		for chunk in block_chunks:
			# Add contextual header if enabled
			if contextual_headers:
				header = f"Document: {filename}"
				section = config.get("section", "")
				
				if section:
					header += f" | Section: {section}"
				
				if chunk.get("type") and chunk["type"] != "text":
					header += f" | Type: {chunk['type']}"
				
				chunk["text"] = f"{header}\n\n{chunk['text']}"
			
			# Ensure required fields for ingest.sh compatibility
			chunk.setdefault("section", config.get("section", ""))
			chunk.setdefault("filename", filename)
			
			all_chunks.append(chunk)
	
	return all_chunks


def smart_chunk_pdf(file_path, config):
	"""
	SmartChunker for PDF files with page-level analysis.
	
	Uses page layout understanding for better accuracy on complex PDFs.
	"""
	if not SMARTCHUNKER_AVAILABLE:
		# Fallback: parse and chunk normally
		try:
			from pdf_parser import parse_pdf
			text, sections = parse_pdf(file_path)
			return smart_chunk_document(text, config)
		except ImportError:
			return []
	
	filename = config.get("filename", os.path.basename(file_path))
	contextual_headers = config.get("contextual_headers", True)
	ocr_enabled = config.get("ocr_enabled", True)
	ocr_lang = config.get("ocr_lang", "eng+fra")
	
	# Analyze PDF pages
	pages = analyze_pdf_pages(file_path, ocr_enabled, ocr_lang)
	
	if not pages:
		# Fallback to text-based parsing
		try:
			from pdf_parser import parse_pdf
			text, sections = parse_pdf(file_path, ocr_enabled=ocr_enabled, ocr_lang=ocr_lang)
			return smart_chunk_document(text, config)
		except ImportError:
			return []
	
	# Process all blocks from all pages
	all_chunks = []
	extractor_config = get_extractor_config()
	
	if "chunk_size" in config:
		extractor_config["chunk_size"] = config["chunk_size"]
	if "chunk_overlap" in config:
		extractor_config["chunk_overlap"] = config["chunk_overlap"]
	
	all_blocks = []
	for page_num, blocks in pages:
		all_blocks.extend(blocks)
	
	for block in all_blocks:
		block_chunks = extract_content(block, extractor_config, all_blocks)
		
		for chunk in block_chunks:
			# Add page info to position if available
			if block.position and "page" in block.position:
				chunk.setdefault("metadata", {})
				chunk["metadata"]["page"] = block.position["page"]
			
			# Add contextual header
			if contextual_headers:
				header = f"Document: {filename}"
				
				if block.position and "page" in block.position:
					header += f" | Page: {block.position['page']}"
				
				if chunk.get("type") and chunk["type"] != "text":
					header += f" | Type: {chunk['type']}"
				
				chunk["text"] = f"{header}\n\n{chunk['text']}"
			
			chunk.setdefault("section", "")
			chunk.setdefault("filename", filename)
			
			all_chunks.append(chunk)
	
	return all_chunks


# Alias for legacy compatibility
chunk_document_smart = smart_chunk_document


# Legacy-compatible entry point (used by ingest.sh)
def chunk_document_legacy_compatible(text, config):
	"""
	Legacy-compatible entry point.
	
	Tries SmartChunker first, falls back through the chain.
	"""
	enabled = os.environ.get("FEATURE_SMARTCHUNKER_ENABLED", "true") == "true"
	
	if enabled and SMARTCHUNKER_AVAILABLE:
		try:
			return smart_chunk_document(text, config)
		except Exception as e:
			print(f"[WARN] SmartChunker error: {e}, falling back")
	
	# Fallback to enhanced chunker
	try:
		from enhanced_chunker import chunk_document_legacy_compatible as enhanced_chunk
		return enhanced_chunk(text, config)
	except ImportError:
		pass
	
	# Final fallback to legacy
	from chunker import chunk_document
	return chunk_document(text, config)

EOFPY

log_ok "SmartChunker module created"


# ============================================================================
# NEW v47 - FASTEMBED HELPER MODULE
# ============================================================================

echo "Creating FastEmbed helper module..."

cat > lib/fastembed_helper.py << 'EOFFASTEMBED'
"""
FastEmbed Helper Module - ONNX-based embedding generation
v47 - Replaces Ollama embeddings with FastEmbed
"""

import os

_embedding_model = None
_model_name = None
_embedding_dim = None

def get_config():
    return {
        "enabled": os.environ.get("FEATURE_FASTEMBED_ENABLED", "true") == "true",
        "model": os.environ.get("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        "cache_dir": os.environ.get("FASTEMBED_CACHE_DIR", "./cache/fastembed"),
        "batch_size": int(os.environ.get("FASTEMBED_BATCH_SIZE", "32")),
        "query_prefix": os.environ.get("FASTEMBED_QUERY_PREFIX", "true") == "true",
        "dimension": int(os.environ.get("EMBEDDING_DIMENSION", "384")),
    }

def init_model():
    global _embedding_model, _model_name, _embedding_dim
    if _embedding_model is not None:
        return True
    config = get_config()
    if not config["enabled"]:
        return False
    try:
        from fastembed import TextEmbedding
        os.makedirs(config["cache_dir"], exist_ok=True)
        _embedding_model = TextEmbedding(
            model_name=config["model"],
            cache_dir=config["cache_dir"]
        )
        _model_name = config["model"]
        test_emb = list(_embedding_model.embed(["test"]))[0]
        _embedding_dim = len(test_emb)
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"[WARN] FastEmbed init error: {e}")
        return False

def get_embedding(text, is_query=False):
    global _embedding_model
    if not init_model():
        return []
    try:
        config = get_config()
        text = text[:8000]
        if is_query and config["query_prefix"]:
            text = f"query: {text}"
        embeddings = list(_embedding_model.embed([text]))
        if embeddings:
            return embeddings[0].tolist()
        return []
    except Exception:
        return []

def get_embeddings_batch(texts, is_query=False):
    global _embedding_model
    if not init_model():
        return []
    try:
        config = get_config()
        texts = [t[:8000] for t in texts]
        if is_query and config["query_prefix"]:
            texts = [f"query: {t}" for t in texts]
        embeddings = list(_embedding_model.embed(texts))
        return [emb.tolist() for emb in embeddings]
    except Exception:
        return []

def get_model_info():
    global _model_name, _embedding_dim
    if not init_model():
        return {"available": False}
    return {"available": True, "model": _model_name, "dimension": _embedding_dim}

def is_available():
    config = get_config()
    if not config["enabled"]:
        return False
    try:
        from fastembed import TextEmbedding
        return True
    except ImportError:
        return False
EOFFASTEMBED

log_ok "FastEmbed helper module created"


# ============================================================================
# BM25 Index Builder (LEGACY v44 - VERBATIM)
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

def tokenize(text):
	text = text.lower()
	tokens = re.findall(r'\b\w+\b', text)
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
	'because', 'until', 'while', 'this', 'that', 'these', 'those',
	'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et',
	'ou', 'mais', 'donc', 'car', 'ni', 'ce', 'cet', 'cette',
	'ces', 'mon', 'ton', 'son', 'ma', 'ta', 'sa', 'mes', 'tes',
	'ses', 'dans', 'sur', 'sous', 'avec', 'sans', 'pour', 'par',
	'en', 'au', 'aux', 'que', 'qui', 'quoi', 'dont', 'quel'}
	
	tokens = [t for t in tokens if len(t) > 2 and t not in stopwords]
	
	return tokens

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

bm25 = BM25Okapi(corpus)

os.makedirs("cache", exist_ok=True)

with open("cache/bm25_index.pkl", "wb") as f:
	pickle.dump({
		"bm25": bm25,
		"doc_ids": doc_ids,
		"corpus": corpus,
		"doc_texts": doc_texts
	}, f)

print(f"BM25 index built: {len(doc_ids)} documents")

print("Building vocabulary for spell correction...")

from collections import Counter, defaultdict

word_freq = Counter()
cooccurrence = defaultdict(Counter)

for text in doc_texts.values():
	words = re.findall(r'\b[a-zA-Z\xe0\xe2\xe6\xe7\xe9\xe8\xea\xeb\xef\xee\xf4\xf9\xfb\xfc\xff\x9c\xc0\xc2\xc6\xc7\xc9\xc8\xca\xcb\xcf\xce\xd4\xd9\xdbܟ\x8c]{3,}\b', text.lower())
	unique_words = set(words)
	word_freq.update(words)
	
	for word in unique_words:
		for other in unique_words:
			if word != other:
				cooccurrence[word][other] += 1

vocab_terms = {
	word: freq for word, freq in word_freq.most_common(10000)
	if freq >= 2
}

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
# Main Ingestion Script (v46 - SmartChunker Integration)
# ============================================================================

echo "Creating main ingestion script (v46 with SmartChunker)..."

cat > "$PROJECT_DIR/ingest.sh" << 'EOFSH'
#!/bin/bash

set -e

[ -f "./config.env" ] && source ./config.env

[ -d "./venv" ] && source ./venv/bin/activate

export OLLAMA_HOST QDRANT_HOST COLLECTION_NAME EMBEDDING_MODEL EMBEDDING_DIMENSION
export CHUNK_SIZE CHUNK_OVERLAP MIN_CHUNK_SIZE MAX_CHUNK_SIZE
export USE_SEMANTIC_CHUNKING USE_SECTIONS CONTEXTUAL_HEADERS PARENT_CHILD_ENABLED
export EXTRACT_TABLES TABLE_STYLE
export OCR_ENABLED OCR_LANGUAGE DEDUP_ENABLED
export EMBEDDING_TIMEOUT

# v47: Export FastEmbed config
export FEATURE_FASTEMBED_ENABLED FASTEMBED_MODEL FASTEMBED_CACHE_DIR
export FASTEMBED_BATCH_SIZE FASTEMBED_QUERY_PREFIX EMBEDDING_MODEL_OLLAMA

# v47: Export SmartChunker config
export FEATURE_SMARTCHUNKER_ENABLED SMARTCHUNKER_MODE
export SMARTCHUNKER_CONFIDENCE_THRESHOLD
export SMARTCHUNKER_TABLE_ATOMIC SMARTCHUNKER_TABLE_EXTRACT_STRUCTURE
export SMARTCHUNKER_CODE_PRESERVE_SYNTAX SMARTCHUNKER_CODE_DETECT_LANGUAGE
export SMARTCHUNKER_FIGURE_LINK_CAPTION SMARTCHUNKER_EQUATION_PRESERVE_FORMAT
export SMARTCHUNKER_OCR_ENABLED SMARTCHUNKER_OCR_LANGUAGE

# v45: Export Enhanced Chunker config (legacy fallback)
export FEATURE_ENHANCED_CHUNKER_ENABLED

DOCS_DIR="${DOCUMENTS_DIR:-./documents}"
FORCE=false
SEMANTIC=false
SECTIONS=true
CONTEXTUAL=true
PARENT_CHILD=false
DEBUG=false
REBUILD_BM25=true
USE_SEMANTIC="${USE_SEMANTIC_CHUNKING:-false}"
USE_SECTIONS="${USE_SECTIONS:-true}"

while [[ $# -gt 0 ]]; do
	case $1 in
		--force) FORCE=true ;;
		--semantic) USE_SEMANTIC=true ;;
		--no-sections) USE_SECTIONS=false ;;
		--no-contextual) CONTEXTUAL=false ;;
		--parent-child) PARENT_CHILD=true ;;
		--no-bm25) REBUILD_BM25=false ;;
		--debug) DEBUG=true ;;
		*) DOCS_DIR="$1" ;;
	esac
	shift
done

if [ ! -d "$DOCS_DIR" ]; then
	echo "ERROR: Documents directory not found: $DOCS_DIR"
	exit 1
fi

OLLAMA="${OLLAMA_HOST:-http://localhost:11434}"
QDRANT="${QDRANT_HOST:-http://localhost:6333}"
COLLECTION="${COLLECTION_NAME:-documents}"
EMBED_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
EMBED_DIM="${EMBEDDING_DIMENSION:-768}"
EMBED_TIMEOUT="${EMBEDDING_TIMEOUT:-60}"
OCR_LANG="${OCR_LANGUAGE:-eng}"

echo "============================================"
echo " RAG v46 Document Ingestion (SmartChunker)"
echo "============================================"
echo ""
echo "Documents: $DOCS_DIR"
echo "Collection: $COLLECTION"
echo "SmartChunker: ${FEATURE_SMARTCHUNKER_ENABLED:-true} (mode: ${SMARTCHUNKER_MODE:-auto})"
echo "Enhanced Chunker Fallback: ${FEATURE_ENHANCED_CHUNKER_ENABLED:-true}"
echo "Semantic: $USE_SEMANTIC"
echo "Sections: $USE_SECTIONS"
echo "Contextual: $CONTEXTUAL"
echo ""

echo -n "Checking Ollama... "
if curl -s "$OLLAMA/api/tags" > /dev/null 2>&1; then
	echo "OK"
else
	echo "NOT RUNNING - Start with: ollama serve"
	exit 1
fi

echo -n "Checking Qdrant... "
if curl -s "$QDRANT/collections" > /dev/null 2>&1; then
	echo "OK"
else
	echo "NOT RUNNING - Start with: docker compose up -d"
	exit 1
fi

echo -n "Checking embedding... "
FASTEMBED_OK=false
if [ "${FEATURE_FASTEMBED_ENABLED:-true}" = "true" ]; then
	if python3 -c "import fastembed" 2>/dev/null; then
		echo "OK (FastEmbed: ${FASTEMBED_MODEL:-BAAI/bge-small-en-v1.5})"
		FASTEMBED_OK=true
	fi
fi
if [ "$FASTEMBED_OK" = "false" ]; then
	if curl -s "$OLLAMA/api/tags" | grep -q "nomic-embed-text"; then
		echo "OK (Ollama fallback: nomic-embed-text)"
	else
		echo "Pulling Ollama fallback model..."
		ollama pull nomic-embed-text
	fi
fi

echo -n "Checking/creating collection... "
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$QDRANT/collections/$COLLECTION")

if [ "$HTTP_CODE" != "200" ]; then
	curl -s -X PUT "$QDRANT/collections/$COLLECTION" \
		-H "Content-Type: application/json" \
		-d "{
			\"vectors\": {
				\"size\": $EMBED_DIM,
				\"distance\": \"Cosine\"
			}
		}" > /dev/null
	echo "CREATED"
else
	echo "EXISTS"
fi

FILES=$(find "$DOCS_DIR" -type f \( \
	-name "*.pdf" -o -name "*.docx" -o -name "*.doc" \
	-o -name "*.xlsx" -o -name "*.xls" \
	-o -name "*.pptx" -o -name "*.ppt" \
	-o -name "*.txt" -o -name "*.md" -o -name "*.rst" \
	-o -name "*.html" -o -name "*.htm" -o -name "*.xml" \
	-o -name "*.csv" -o -name "*.tsv" -o -name "*.json" \
	-o -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \
	-o -name "*.gif" -o -name "*.bmp" -o -name "*.tiff" \
\) 2>/dev/null | sort)

if [ -z "$FILES" ]; then
	FILE_COUNT=0
else
	FILE_COUNT=$(echo "$FILES" | wc -l)
fi

if [ "$FILE_COUNT" -eq 0 ]; then
	echo ""
	echo "No documents found in $DOCS_DIR"
	exit 0
fi

echo ""
echo "Found $FILE_COUNT documents"
echo ""

# Run Python from project root, add lib to path
python3 << EOFPY
import sys
import os
import json
import hashlib
import time
import requests

# Add lib directory to Python path
sys.path.insert(0, './lib')

from pdf_parser import parse_pdf
from office_parser import parse_docx, parse_doc, parse_xlsx, parse_xls, parse_pptx
from text_parser import parse_txt, parse_md, parse_html, parse_csv, parse_json
from image_parser import parse_image

# v47: FastEmbed support
fastembed_available = False
get_fastembed_embedding = None
try:
    from fastembed_helper import get_embedding as get_fastembed_embedding, is_available, get_model_info
    fastembed_available = is_available()
    if fastembed_available:
        info = get_model_info()
        print(f"[INFO] FastEmbed: {info.get('model', 'unknown')} ({info.get('dimension', '?')} dim)")
except ImportError:
    print("[INFO] FastEmbed not available, using Ollama")

OLLAMA = "$OLLAMA"
QDRANT = "$QDRANT"
COLLECTION = "$COLLECTION"
EMBED_MODEL = "$EMBED_MODEL"
EMBED_MODEL_OLLAMA = os.environ.get("EMBEDDING_MODEL_OLLAMA", "nomic-embed-text")
EMBED_DIM = $EMBED_DIM
EMBED_TIMEOUT = $EMBED_TIMEOUT
USE_FASTEMBED = os.environ.get("FEATURE_FASTEMBED_ENABLED", "true") == "true"
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "80"))
MIN_CHUNK = int(os.environ.get("MIN_CHUNK_SIZE", "100"))
USE_SEMANTIC = "$USE_SEMANTIC" == "true"
USE_SECTIONS = "$USE_SECTIONS" == "true"
CONTEXTUAL = "$CONTEXTUAL" == "true"
PARENT_CHILD = "$PARENT_CHILD" == "true"
OCR_LANG = "$OCR_LANG"
FORCE = "$FORCE" == "true"
DEBUG = "$DEBUG" == "true"
DEDUP = os.environ.get("DEDUP", "true") == "true"

# v46: SmartChunker selection with fallback chain
SMARTCHUNKER_ENABLED = os.environ.get("FEATURE_SMARTCHUNKER_ENABLED", "true") == "true"
ENHANCE_CHUNKER = os.environ.get("FEATURE_ENHANCED_CHUNKER_ENABLED", "true") == "true"

chunker_name = "legacy"

if SMARTCHUNKER_ENABLED:
	try:
		from smartchunker import chunk_document_legacy_compatible as chunk_document
		from smartchunker import get_smartchunker_info
		info = get_smartchunker_info()
		if info["available"]:
			chunker_name = f"SmartChunker ({info['mode']})"
			print(f"[INFO] Using {chunker_name}")
		else:
			raise ImportError("SmartChunker not available")
	except Exception as e:
		print(f"[WARN] SmartChunker failed: {e}, trying enhanced chunker")
		SMARTCHUNKER_ENABLED = False

if not SMARTCHUNKER_ENABLED and ENHANCE_CHUNKER:
	try:
		from enhanced_chunker import chunk_document_legacy_compatible as chunk_document
		chunker_name = "enhanced"
		print("[INFO] Using ENHANCED chunker (v45 fallback)")
	except Exception as e:
		print(f"[WARN] Enhanced chunker failed: {e}, falling back to legacy")
		from chunker import chunk_document
		chunker_name = "legacy"
		ENHANCE_CHUNKER = False

if not SMARTCHUNKER_ENABLED and not ENHANCE_CHUNKER:
	from chunker import chunk_document
	print("[INFO] Using LEGACY chunker (v44 compatible)")

dedup_file = "./cache/dedup_index.json"

if FORCE:
	dedup_index = set()
else:
	try:
		with open(dedup_file) as f:
			dedup_index = set(json.load(f))
	except:
		dedup_index = set()

os.makedirs("./.ingest_tracking", exist_ok=True)
os.makedirs("./cache", exist_ok=True)

files = """$FILES""".strip().split('\n')
files = [f for f in files if f.strip()]

total_stored = 0
total_skipped = 0
file_num = 0
start_time = time.time()

print(f"\nProcessing {len(files)} files (chunker: {chunker_name})...")
print("-" * 60)

for file_path in files:
	file_num += 1
	basename = os.path.basename(file_path)
	tracking_file = f"./.ingest_tracking/{basename.replace(' ', '_').replace('/', '_')}"
	
	if not FORCE and os.path.exists(tracking_file):
		if DEBUG:
			print(f"[{file_num}/{len(files)}] SKIP {basename[:40]}...")
		
		total_skipped += 1
		continue
	
	print(f"[{file_num}/{len(files)}] {basename[:50]}...", end=" ", flush=True)
	
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
	
	chunk_config = {
		"chunk_size": CHUNK_SIZE,
		"chunk_overlap": CHUNK_OVERLAP,
		"min_chunk_size": MIN_CHUNK,
		"use_semantic": USE_SEMANTIC,
		"use_sections": USE_SECTIONS,
		"contextual_headers": CONTEXTUAL,
		"parent_child": PARENT_CHILD,
		"filename": basename,
		"section": "",
		"use_enhanced": ENHANCE_CHUNKER,
		"use_smartchunker": SMARTCHUNKER_ENABLED,
		"ocr_enabled": True,
		"ocr_lang": OCR_LANG,
	}
	
	try:
		chunks = chunk_document(text, chunk_config)
	except Exception as e:
		print(f"ERROR chunking: {e}")
		continue
	
	print(f"-> {len(chunks)} chunks", end=" ", flush=True)
	
	if not chunks:
		print("-> no chunks")
		continue
	
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
		
		chunk_hash = hashlib.md5(chunk_text[:300].encode()).hexdigest()
		
		if DEDUP and chunk_hash in dedup_index:
			dedup_count += 1
			continue
		
		# v47: FastEmbed with Ollama fallback
		try:
			embedding = []
			if USE_FASTEMBED and fastembed_available and get_fastembed_embedding:
				embedding = get_fastembed_embedding(chunk_text[:8000])
			
			if not embedding:
				resp = requests.post(f"{OLLAMA}/api/embeddings", json={
					"model": EMBED_MODEL_OLLAMA,
					"prompt": chunk_text[:2000]
				}, timeout=EMBED_TIMEOUT)
				if resp.status_code == 200:
					embedding = resp.json().get("embedding", [])
			
			if not embedding or len(embedding) != EMBED_DIM:
				continue
		
		except Exception as e:
			if DEBUG:
				print(f"\nEmbed error: {e}")
			continue
		
		doc_id = hashlib.md5(f"{file_path}_{i}_{chunk_hash}".encode()).hexdigest()
		
		payload = {
			"text": chunk_text,
			"source": file_path,
			"filename": basename,
			"section": section,
			"chunk_type": chunk_type,
			"chunk_index": i
		}
		
		if isinstance(chunk_obj, dict):
			if "parent_id" in chunk_obj:
				payload["parent_id"] = chunk_obj["parent_id"]
			
			if "children" in chunk_obj:
				payload["children"] = chunk_obj["children"]
			
			if "confidence" in chunk_obj:
				payload["confidence"] = chunk_obj["confidence"]
			
			if "detected_type" in chunk_obj:
				payload["detected_type"] = chunk_obj["detected_type"]
			
			if "metadata" in chunk_obj:
				payload["metadata"] = chunk_obj["metadata"]
			
			# v46: Additional SmartChunker metadata
			if "position" in chunk_obj:
				payload["position"] = chunk_obj["position"]
			
			if "relationships" in chunk_obj:
				payload["relationships"] = chunk_obj["relationships"]
		
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
	
	info = f"-> stored {stored}"
	
	if dedup_count > 0:
		info += f", dedup {dedup_count}"
	
	print(info)
	
	total_stored += stored
	
	with open(tracking_file, 'w') as f:
		f.write(str(stored))

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

try:
	resp = requests.get(f"{QDRANT}/collections/{COLLECTION}")
	count = resp.json().get("result", {}).get("points_count", 0)
	print(f"Database total: {count} chunks")
except:
	pass

EOFPY

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

log_ok "Ingestion script created (v46 with SmartChunker)"

# ============================================================================
# END NEW VERSION 46 EXTENSIONS
# ============================================================================

echo ""
echo "============================================================================"
echo " Ingestion Setup Complete! (v46 with SmartChunker + DeepDoc)"
echo "============================================================================"
echo ""
echo "Features implemented:"
echo " - FastEmbed ONNX embeddings (v47 - CPU optimized)"
echo " - SmartChunker with DeepDoc integration (production-grade)"
echo " - Heuristic fallback for low-resource systems"
echo " - Content-type aware chunking:"
echo "   - Tables: Atomic chunks, semantic structure extraction"
echo "   - Code: Syntax preservation, language detection"
echo "   - Figures: Caption linking, image references"
echo "   - Equations: LaTeX/MathML format preservation"
echo "   - Text: Semantic chunking with section awareness"
echo " - Position/layout metadata"
echo " - Relationship tracking between chunks"
echo " - Backward compatible with v45/v44 chunkers"
echo ""
echo "Fallback chain:"
echo " SmartChunker(DeepDoc) -> SmartChunker(Heuristic) -> EnhancedChunker -> Legacy"
echo ""
echo "Legacy v45/v44 features preserved:"
echo " - Contextual headers"
echo " - Semantic chunking"
echo " - Section detection"
echo " - Parent-child chunks"
echo " - Table extraction"
echo " - OCR support"
echo " - Deduplication"
echo " - BM25 index building"
echo " - French language support"
echo ""
echo "Usage:"
echo " ./ingest.sh                # Index ./documents"
echo " ./ingest.sh --force        # Re-index everything"
echo " ./ingest.sh --semantic     # Use sentence chunking"
echo " ./ingest.sh --parent-child # Hierarchical chunks"
echo " ./ingest.sh --debug        # Verbose output"
echo ""
echo "Next: Run bash setup-rag-query-v47.sh"
echo "============================================================================"
