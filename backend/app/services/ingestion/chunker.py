from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

def chunk_text_blocks(blocks: List[Dict[str, Any]], chunk_size: int = 512, chunk_overlap: int = 64) -> List[Dict[str, Any]]:
    """
    Takes raw text blocks (e.g. from a PDF) and recursively chunks them.
    Preserves metadata like page number and section.
    
    Args:
        blocks: List of dicts containing 'text' and metadata (page, block_type)
        chunk_size: Maximum characters per chunk (512 is good for dense retrieval)
        chunk_overlap: Characters to overlap between chunks to maintain context
    """
    # Initialize the Langchain recursive text splitter
    # It tries to split on paragraphs (\n\n), then sentences (\n), then words, then chars
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    final_chunks = []
    
    for block in blocks:
        text = block.get("text") or block.get("table_as_markdown") or ""
        if not text:
            continue
            
        # If it's already a table, we often prefer NOT to split it randomly,
        # but for safety against massive tables, we'll let it chunk if necessary.
        splits = splitter.split_text(text)
        
        for i, split_text in enumerate(splits):
            final_chunks.append({
                "text": split_text,
                "metadata": {
                    "page": block.get("page", 1),
                    "sheet_name": block.get("sheet_name"),
                    "row_range": block.get("row_range"),
                    "chunk_type": block.get("block_type", "text"),
                    "char_count": len(split_text)
                }
            })
            
    return final_chunks
