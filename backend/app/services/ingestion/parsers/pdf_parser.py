import fitz  # PyMuPDF
from typing import List, Dict, Any

def extract_text_blocks(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extracts text blocks from a PDF using PyMuPDF.
    Returns a list of blocks, preserving page number and block structure.
    """
    blocks = []
    # Open the document from bytes
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    
    for page_num, page in enumerate(doc, start=1):
        # Extract blocks (dict format). Each block contains lines and spans of text.
        page_blocks = page.get_text("blocks")
        
        for block in page_blocks:
            text = block[4].strip()
            if text:
                blocks.append({
                    "text": text,
                    "page": page_num,
                    "block_type": "text"
                })
    return blocks

def extract_tables(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Experimental/Basic table extraction using PyMuPDF's new table features.
    Extracts tables as markdown.
    """
    tables = []
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    
    for page_num, page in enumerate(doc, start=1):
        page_tables = page.find_tables()
        for table in page_tables:
            # Convert to markdown
            md_table = table.to_markdown()
            if md_table:
                tables.append({
                    "table_as_markdown": md_table,
                    "page": page_num,
                    "block_type": "table"
                })
    return tables
