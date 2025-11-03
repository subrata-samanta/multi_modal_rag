# Enhanced RAG System with Extraction Caching

## Overview
The RAG system has been enhanced to save expensive document extractions to files first, then ingest them into the vector database. This prevents costly re-extractions and provides better efficiency.

## Key Improvements

### 1. Extraction Caching System
- **Automatic file-based caching**: All extracted content is saved to JSON files
- **Image persistence**: Extracted page images are saved to disk
- **Hash-based validation**: Uses MD5 hash to detect file changes
- **Smart cache invalidation**: Automatically re-processes if source file changes

### 2. Enhanced File Support
- **EML (Email) files**: Extract email metadata, body, and attachment info
- **XLS/XLSX files**: Extract tabular data and embedded images
- **Image extraction from Excel**: Supports embedded images in spreadsheets

### 3. Directory Structure
```
document_extractions/
├── images/          # Saved page images (PNG format)
│   └── filename_hash_pageX_type.png
└── content/         # Cached extraction results (JSON format)
    └── filename_hash.json
```

### 4. New Features in Main Menu
1. **Ingest Single Document** - With cache support
2. **Ingest Folder** - Batch processing with caching
3. **Load from Saved Extractions** - Bulk load from cache
4. **View Extraction Cache Status** - Monitor cached files
5. **Clear Old Extractions** - Cleanup old cache files
6. **Force Reprocess Document** - Bypass cache when needed

### 5. JSON Extraction Format
```json
{
  "source_file": "path/to/document.pdf",
  "file_hash": "md5hash",
  "extraction_date": "2025-11-03T19:30:51.123456",
  "content": {
    "text": [
      {
        "content": "extracted text",
        "page": 1,
        "source": "path/to/document.pdf",
        "type": "text",
        "image_path": "path/to/saved/image.png"
      }
    ],
    "tables": [...],
    "visuals": [...]
  }
}
```

## Benefits

### 1. Cost Efficiency
- **Avoid re-extraction**: Expensive LLM calls only happen once per document
- **Fast re-ingestion**: Load from cache in seconds instead of minutes
- **Reduced API costs**: Significant savings on Google Vertex AI usage

### 2. Performance Improvements
- **Faster startup**: Quick loading from saved extractions
- **Batch processing**: Efficient folder ingestion with caching
- **Parallel processing**: Maintained for new extractions

### 3. Data Persistence
- **Extraction preservation**: Keep extracted content even if vector DB is cleared
- **Image references**: Maintain links to extracted page images
- **Audit trail**: Track when extractions were performed

### 4. Flexibility
- **Selective reprocessing**: Choose which files to re-extract
- **Cache management**: View, clean, and manage cached extractions
- **Multiple ingestion paths**: Fresh extraction or cache loading

## Usage Examples

### Basic Document Processing
```python
# Process a document (uses cache if available)
rag.ingest_document("document.pdf")

# Force reprocessing (ignore cache)
rag.ingest_document("document.pdf", force_reprocess=True)
```

### Batch Operations
```python
# Process folder with caching
rag.ingest_folder_with_caching("/path/to/documents")

# Load from all saved extractions
rag.ingest_from_saved_extractions()
```

### Cache Management
```python
# View cache status
rag.get_extraction_status()

# Clear old extractions (keep last 30 days)
rag.clear_old_extractions(30)
```

## File Types Supported
- **PDF**: Text, tables, and visual content
- **PPTX**: Slide content extraction
- **EML**: Email metadata, body, and attachment info
- **XLS/XLSX**: Tabular data and embedded images

## Technical Implementation

### Document Processor Enhancements
- `_get_file_hash()`: Generate MD5 hash for cache validation
- `_save_extraction_results()`: Save to JSON with metadata
- `_load_saved_extraction()`: Load and validate cached results
- `has_saved_extraction()`: Check cache availability

### Vector Store Enhancements
- `add_documents_from_extraction_file()`: Load single extraction file
- `add_documents_from_multiple_extractions()`: Bulk load from directory

### RAG Pipeline Enhancements
- `ingest_from_saved_extractions()`: High-level cache loading
- `get_extraction_status()`: Cache monitoring
- `force_reprocess_document()`: Cache bypass

## Migration Guide
The enhanced system is backward compatible. Existing vector databases continue to work, and new caching features are additive.

To migrate:
1. Update your code to use the new main.py interface
2. Existing documents will be cached on next processing
3. Use "View Extraction Cache Status" to monitor cache building

## Configuration
All existing configuration in `config.py` remains the same. The caching system uses default directories that are created automatically.

## Testing
Run the test script to verify functionality:
```bash
python test_extraction_caching.py
```

This will verify:
- Directory creation
- Hash generation consistency
- Cache validation
- Component initialization