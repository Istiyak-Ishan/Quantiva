import pytest
from app.services.ingestion.chunker import chunk_text_blocks

class TestChunker:
    def test_short_text_single_chunk(self):
        blocks = [{"text": "This is a short document.", "page": 1}]
        chunks = chunk_text_blocks(blocks)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "This is a short document."
        assert chunks[0]["metadata"]["page"] == 1

    def test_long_text_multiple_chunks(self):
        long_text = "The quick brown fox jumps over the lazy dog. " * 50
        blocks = [{"text": long_text, "page": 2}]
        chunks = chunk_text_blocks(blocks, chunk_size=200, chunk_overlap=30)
        assert len(chunks) > 1
        assert all(c["metadata"]["page"] == 2 for c in chunks)

    def test_empty_block_skipped(self):
        blocks = [{"text": "", "page": 1}, {"text": "Valid text", "page": 2}]
        chunks = chunk_text_blocks(blocks)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Valid text"

    def test_chunk_metadata_preserved(self):
        blocks = [{
            "text": "Revenue figures for Q3.",
            "page": 5,
            "sheet_name": "Financials",
            "block_type": "table"
        }]
        chunks = chunk_text_blocks(blocks)
        assert len(chunks) == 1
        meta = chunks[0]["metadata"]
        assert meta["page"] == 5
        assert meta["sheet_name"] == "Financials"
        assert meta["chunk_type"] == "table"
        assert meta["char_count"] == len("Revenue figures for Q3.")
