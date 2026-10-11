from typing import List, Dict, Any
from app.services.ingestion.parsers.pdf_parser import extract_text_blocks, extract_tables
from app.services.ingestion.parsers.excel_parser import parse_excel
from app.services.ingestion.parsers.csv_parser import parse_csv
from app.services.ingestion.chunker import chunk_text_blocks

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
        # For this MVP, we just combine them. Tables might duplicate text slightly 
        # depending on PyMuPDF block extraction, but it ensures we get everything.
        raw_blocks.extend(text_blocks)
        raw_blocks.extend(table_blocks)
        
    elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        raw_blocks = parse_excel(file_bytes)
        
    elif file_type in ["text/csv", "application/csv"]:
        raw_blocks = parse_csv(file_bytes)
        
    elif file_type == "text/plain":
        # Simple fallback for text files
        raw_blocks = [{"text": file_bytes.decode('utf-8', errors='ignore'), "page": 1, "block_type": "text"}]
        
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # 2. Chunk the text
    chunks = chunk_text_blocks(raw_blocks, chunk_size=512, chunk_overlap=64)
    
    return chunks
