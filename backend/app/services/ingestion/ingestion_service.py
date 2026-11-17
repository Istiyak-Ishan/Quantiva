import re
from typing import List, Dict, Any

from app.services.ingestion.parsers.pdf_parser import extract_text_blocks, extract_tables
from app.services.ingestion.parsers.excel_parser import parse_excel
from app.services.ingestion.parsers.csv_parser import parse_csv
from app.services.ingestion.parsers.docx_parser import parse_docx
from app.services.ingestion.chunker import chunk_text_blocks


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _table_cell_text(markdown: str) -> str:
    """Normalize a markdown table to its cell content for overlap checks."""
    cells: list[str] = []
    for line in markdown.splitlines():
        line = line.strip()
        if not line or set(line) <= set("|-: "):
            continue
        cells.extend(cell.strip() for cell in line.strip("|").split("|"))
    return _normalize(" ".join(cell for cell in cells if cell))


def deduplicate_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep table chunks and drop text blocks already represented by them.

    PDF text extraction can emit a plain-text copy of a table alongside the
    table-as-markdown block. Keeping both doubles embeddings and makes keyword
    retrieval return the same content twice. Exact duplicate blocks are also
    removed. Prose blocks that only *contain* a table are retained.
    """
    table_texts = [
        text
        for block in blocks
        if block.get("block_type") == "table"
        for text in [_table_cell_text(block.get("table_as_markdown", ""))]
        if text
    ]
    kept: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for block in blocks:
        raw = block.get("text") or block.get("table_as_markdown") or ""
        normalized = _normalize(raw)
        if not normalized or normalized in seen:
            continue
        if block.get("block_type") == "text" and any(
            len(table_text) >= 20
            and (normalized in table_text or table_text in normalized)
            for table_text in table_texts
        ):
            # The table block preserves the same content with structure.
            continue
        seen.add(normalized)
        kept.append(block)
    return kept


def process_document(doc_id: str, file_bytes: bytes, file_type: str) -> List[Dict[str, Any]]:
    """
    Main orchestration function for document ingestion.
    Routes the file to the correct parser based on file_type, extracts blocks,
    and returns chunked text ready for vector embeddings.
    """
    raw_blocks = []
    
    # 1. Route to Parser
    if file_type == "application/pdf":
        text_blocks = extract_text_blocks(file_bytes)
        table_blocks = extract_tables(file_bytes)
        # Preserve structured table chunks and drop duplicated plain-text copies.
        raw_blocks.extend(text_blocks)
        raw_blocks.extend(table_blocks)
        
    elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        raw_blocks = parse_excel(file_bytes)
        
    elif file_type in ["text/csv", "application/csv"]:
        raw_blocks = parse_csv(file_bytes)
        
    elif file_type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/docx",
        "application/msword"
    ]:
        raw_blocks = parse_docx(file_bytes)
        
    elif file_type in ["text/plain", "text/markdown", "text/x-markdown"]:
        # Simple fallback for text files.  The desktop ingestion pipeline
        # (app/ingestion) adds Markdown-aware heading/table/code extraction.
        raw_blocks = [{"text": file_bytes.decode('utf-8', errors='ignore'), "page": 1, "block_type": "text"}]
        
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # 2. Chunk the text. Chunk sizes are characters by design and documented
    #    in the chunker; tokenizer dependency is intentionally avoided here.
    chunks = chunk_text_blocks(deduplicate_blocks(raw_blocks), chunk_size=512, chunk_overlap=64)
    
    return chunks
