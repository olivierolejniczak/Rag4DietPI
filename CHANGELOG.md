# Changelog

All notable changes to this project are documented in this file.

This project evolved through multiple development iterations, consolidating into this final release.

---

## Current Release

### Features

#### Document Processing
- **20+ File Format Support**: PDF, DOCX, DOC, XLSX, CSV, PPTX, HTML, XML, JSON, YAML, Markdown, and more
- **Unstructured.io Integration**: Robust document parsing with fallback handling
- **Smart Chunking**: Semantic-aware document splitting with configurable overlap
- **French OCR**: tesseract-ocr-fra for French document scanning
- **Legacy .doc Support**: antiword integration for Word 97-2003 files
- **CSV Dual Mode**: Structured + natural language representations for tabular data
- **Website Crawling**: Full web scraping with depth control

#### Search & Retrieval
- **Hybrid Search**: Dense (semantic) + Sparse (BM25/Splade) vectors
- **Native RRF Fusion**: Reciprocal Rank Fusion directly in Qdrant
- **HyDE**: Hypothetical Document Embeddings for abstract queries
- **Multi-Pass Retrieval**: Query expansion, rewriting, and variants
- **StepBack Queries**: Broader context retrieval
- **Subquery Decomposition**: Complex query handling
- **FlashRank Reranking**: Neural relevance scoring

#### Generation & Quality
- **Map/Reduce Summarization**: Summarize documents of any length
- **Extraction Mode**: Structured data extraction from documents
- **Self-Reflection**: Answer verification for high-stakes queries
- **CRAG**: Corrective RAG with web search fallback
- **Citations**: Source attribution in responses
- **RAGAS Evaluation**: Built-in quality metrics

#### System Optimization
- **CPU Score Profiling**: Automatic hardware detection (cores × MHz)
- **Adaptive Embedding Selection**: Model choice based on available RAM
- **Qdrant Low-Memory Mode**: ON_DISK_PAYLOAD for systems <8GB
- **Tiered Performance**: Quick/Default/Deep query modes
- **2-Layer Caching**: Search results + LLM response caching
- **Swap Management**: Automatic configuration for low-RAM systems

#### Infrastructure
- **Native Ollama**: Direct installation (not containerized)
- **Qdrant Container**: Persistent vector storage
- **SearXNG Container**: Privacy-respecting web search
- **pyspellchecker**: French + English spellcheck with bundled dictionaries

---

## Development History

The system evolved through multiple iterations, each adding critical features:

### Retrieval Improvements
- FastEmbed ONNX embeddings for fast, local embedding generation
- Native Qdrant client with gRPC support for faster operations
- Hybrid search combining semantic and keyword matching
- Native RRF fusion eliminating external dependencies

### Quality Features
- Quality ledger for tracking query performance
- Abstention on low-confidence responses
- Multi-pass retrieval for comprehensive results
- CRAG web fallback when local context insufficient

### French Language Support
- Document deduplication with French normalization
- pyspellchecker integration with FR/EN dictionaries
- Query normalization for technical terms

### Evaluation & Testing
- RAGAS integration for automated quality metrics
- Auto-generated test questions from documents
- Batch evaluation with SLA thresholds

### Data Handling
- CSV natural language transformation
- Dual-mode spreadsheet ingestion (structured + readable)
- Extended format support (XML, YAML, PowerShell, etc.)

### Web Features
- Website crawling with configurable depth
- Web-only query mode bypassing local documents
- SearXNG integration for privacy-respecting search

### Performance Tiers
- Tiered query modes (quick/default/deep)
- 2-layer caching system
- Real-time progress tracking

### Advanced Generation
- Map/Reduce for long document summarization
- Extraction mode for structured data
- Self-reflection answer verification

---

## Architecture Decisions

### Why Native Ollama (Not Containerized)
- Direct GPU access without Docker GPU passthrough complexity
- Lower memory overhead
- Simpler model management with `ollama pull`

### Why Qdrant Over Alternatives
- Native hybrid search support
- RRF fusion built-in
- gRPC for fast operations
- Excellent low-memory options

### Why FastEmbed Over Ollama Embeddings
- 10x faster inference
- ONNX optimization
- CPU-efficient
- Multiple model options

### Why pyspellchecker Over Hunspell
- Bundled dictionaries (no system deps)
- Pure Python (portable)
- Simple API

---

## Migration Notes

### From Earlier Versions
If upgrading from development versions:

1. Backup existing data:
   ```bash
   ./backup.sh --full
   ```

2. Run new setup scripts:
   ```bash
   sudo bash setup-rag-core.sh
   sudo bash setup-rag-ingest.sh
   sudo bash setup-rag-query.sh
   ```

3. Re-ingest documents (recommended for new features):
   ```bash
   ./ingest.sh --force ./documents/
   ```

### Configuration Migration
The new `config.env` includes all settings. Key additions:
- `MAPREDUCE_*`: Summarization settings
- `EXTRACTION_*`: Data extraction settings
- `REFLECTION_*`: Answer verification settings
- `SYSTEM_CPU_SCORE`: Hardware profiling

---

## Known Limitations

1. **No REST API**: CLI-only (Web UI available separately)
2. **No Docker Compose**: Services managed individually
3. **No Streaming**: Full response generated before output
4. **Single Collection**: One vector collection per instance
5. **English-Optimized Models**: bge-* models primarily English

---

## Future Considerations

Potential improvements for future development:
- REST API with OpenAI-compatible endpoints
- Docker Compose orchestration
- Knowledge Graph integration (GraphRAG)
- Multi-modal support (images)
- Streaming responses
