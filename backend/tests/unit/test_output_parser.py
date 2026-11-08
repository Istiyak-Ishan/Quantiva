import pytest
from app.services.llm.output_parser import parse_structured_output, ParseError
from app.schemas.agent import AgentAnswer

VALID_JSON = '''{
    "answer": "The Q3 revenue was $5M, up 15% from Q2.",
    "citations": [
        {
            "doc_id": "abc-123",
            "filename": "annual_report.pdf",
            "page_or_row": "12",
            "text_excerpt": "Q3 revenue totalled $5M",
            "relevance_score": 0.92
        }
    ],
    "confidence": 0.94,
    "tokens_used": 312
}'''

MARKDOWN_WRAPPED = f"```json\n{VALID_JSON}\n```"
MARKDOWN_NO_LANG = f"```\n{VALID_JSON}\n```"

class TestOutputParser:
    def test_clean_json_parsed(self):
        result = parse_structured_output(VALID_JSON, AgentAnswer)
        assert isinstance(result, AgentAnswer)
        assert "revenue" in result.answer.lower()
        assert result.confidence == 0.94
        assert len(result.citations) == 1

    def test_markdown_json_block_extracted(self):
        result = parse_structured_output(MARKDOWN_WRAPPED, AgentAnswer)
        assert isinstance(result, AgentAnswer)
        assert result.answer != ""

    def test_markdown_no_language_tag(self):
        result = parse_structured_output(MARKDOWN_NO_LANG, AgentAnswer)
        assert isinstance(result, AgentAnswer)

    def test_invalid_json_raises_parse_error(self):
        with pytest.raises(ParseError):
            parse_structured_output("This is not JSON at all!", AgentAnswer)

    def test_empty_string_raises_parse_error(self):
        with pytest.raises(ParseError):
            parse_structured_output("", AgentAnswer)

    def test_citation_fields_present(self):
        result = parse_structured_output(VALID_JSON, AgentAnswer)
        citation = result.citations[0]
        assert citation.doc_id == "abc-123"
        assert citation.page_or_row == "12"
