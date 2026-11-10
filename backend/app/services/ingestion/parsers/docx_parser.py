import io
from typing import List, Dict, Any

def parse_docx(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extracts text paragraphs and structured tables from a Word (.docx) file.
    Returns a list of block dictionaries ready for chunking.
    """
    blocks: List[Dict[str, Any]] = []
    
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
    except Exception:
        # Fallback to basic text extraction via zip/xml if docx fails
        try:
            import zipfile
            import xml.etree.ElementTree as ET
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
                xml_content = z.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                texts = [elem.text for elem in tree.iter() if elem.text]
                combined = " ".join(texts).strip()
                if combined:
                    return [{"text": combined, "page": 1, "block_type": "text"}]
        except Exception:
            pass
        return blocks

    current_page = 1
    words_on_page = 0
    WORDS_PER_PAGE_ESTIMATE = 450

    # 1. Extract paragraphs
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
            
        word_count = len(text.split())
        words_on_page += word_count
        if words_on_page > WORDS_PER_PAGE_ESTIMATE:
            current_page += 1
            words_on_page = word_count

        blocks.append({
            "text": text,
            "page": current_page,
            "block_type": "text"
        })

    # 2. Extract tables
    for table_idx, table in enumerate(doc.tables, start=1):
        rows_data = []
        for row in table.rows:
            row_cells = [cell.text.replace("\n", " ").strip() for cell in row.cells]
            rows_data.append(row_cells)

        if not rows_data:
            continue

        header = rows_data[0]
        divider = ["---"] * len(header)
        body = rows_data[1:]

        md_lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(divider) + " |"
        ]
        for row in body:
            # Pad row if fewer cells than header
            padded = row + [""] * max(0, len(header) - len(row))
            md_lines.append("| " + " | ".join(padded[:len(header)]) + " |")

        md_table = "\n".join(md_lines)
        blocks.append({
            "text": md_table,
            "table_as_markdown": md_table,
            "page": current_page,
            "block_type": "table"
        })

    return blocks
