import pytest
from app.services.ingestion.chunker import chunk_text_blocks
from app.services.ingestion.ingestion_service import deduplicate_blocks

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

    def test_table_as_markdown_is_chunked(self):
        blocks = [{
            "table_as_markdown": "| Col1 | Col2 |\n|---|---|\n| Val1 | Val2 |",
            "page": 3,
            "block_type": "table"
        }]
        chunks = chunk_text_blocks(blocks)
        assert len(chunks) == 1
        assert "| Col1 | Col2 |" in chunks[0]["text"]
        assert chunks[0]["metadata"]["chunk_type"] == "table"


class TestBlockDeduplication:
    def test_duplicate_text_block_is_removed(self):
        blocks = [
            {"text": "Same content", "page": 1, "block_type": "text"},
            {"text": "  same   content ", "page": 1, "block_type": "text"},
        ]
        assert len(deduplicate_blocks(blocks)) == 1

    def test_table_is_preserved_and_duplicate_text_copy_dropped(self):
        table = "| Metric | Value |\n|---|---|\n| Revenue | 10 |\n| Profit | 3 |"
        blocks = [
            {"text": "Introductory prose about the annual report.", "page": 1, "block_type": "text"},
            {"text": "Metric Value Revenue 10 Profit 3", "page": 1, "block_type": "text"},
            {"table_as_markdown": table, "page": 1, "block_type": "table"},
        ]
        kept = deduplicate_blocks(blocks)
        assert [block["block_type"] for block in kept] == ["text", "table"]
        assert kept[1]["table_as_markdown"] == table

    def test_prose_containing_table_words_is_kept(self):
        table = "| Metric | Value |\n|---|---|\n| Revenue | 10 |\n| Profit | 3 |"
        blocks = [
            {"text": "Revenue and profit are discussed in detail in the narrative section of this report.", "page": 1, "block_type": "text"},
            {"table_as_markdown": table, "page": 1, "block_type": "table"},
        ]
        assert len(deduplicate_blocks(blocks)) == 2
