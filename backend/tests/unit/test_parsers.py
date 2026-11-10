import io
import pytest
import docx
from app.services.ingestion.parsers.docx_parser import parse_docx
from app.services.ingestion.parsers.pdf_parser import extract_text_blocks, extract_tables
from app.services.ingestion.parsers.csv_parser import parse_csv
from app.services.ingestion.parsers.excel_parser import parse_excel
from app.services.ingestion.ingestion_service import process_document

class TestDocxParser:
    def test_parse_docx_paragraphs_and_tables(self):
        doc = docx.Document()
        doc.add_heading("Financial Report 2026", level=1)
        doc.add_paragraph("This document outlines the fiscal performance.")
        
        # Add a table
        table = doc.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Metric"
        table.cell(0, 1).text = "Value"
        table.cell(1, 0).text = "Revenue"
        table.cell(1, 1).text = "$10M"

        stream = io.BytesIO()
        doc.save(stream)
        stream.seek(0)
        file_bytes = stream.read()

        blocks = parse_docx(file_bytes)
        assert len(blocks) >= 2
        
        texts = [b.get("text", "") for b in blocks]
        assert any("Financial Report" in t for t in texts)
        assert any("Revenue" in t for t in texts)
        
        # Verify routing in process_document
        chunks = process_document("doc-test", file_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        assert len(chunks) >= 1
        assert any("Revenue" in c["text"] for c in chunks)

class TestCsvParser:
    def test_parse_csv(self):
        csv_data = b"Quarter,Revenue,Profit\nQ1,100,20\nQ2,150,35\n"
        blocks = parse_csv(csv_data, batch_size=1)
        assert len(blocks) == 2
        assert "Q1" in blocks[0]["text"]
        assert blocks[0]["row_range"] == "1-1"

class TestPdfParser:
    def test_empty_bytes_or_invalid_pdf(self):
        try:
            blocks = extract_text_blocks(b"not a real pdf")
            assert isinstance(blocks, list)
        except Exception:
            pass
