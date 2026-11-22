"""Deterministic tests for dual-mode agents, sandboxing and citations."""
from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
import os

import pytest

from app.agents.citations import CitationValidator
from app.agents.reasoning import AgentIterationLimitError, ReActAgent
from app.agents.tools import AgentTool, AgentToolRegistry
from app.providers.interfaces.exceptions import CapabilityNotSupportedError
from app.providers.interfaces.llm import LLMProvider
from app.providers.interfaces.types import (
    ChatMessage,
    ConnectionTestResult,
    GenerationConfig,
    LLMResponse,
    LLMStreamChunk,
    ProviderCapabilities,
    ToolCallRequest,
    ToolDefinition,
)
from app.retrieval.types import RetrievedChunk
from app.security.sandbox import LocalPythonSandbox


def capabilities(*, function_calling: bool) -> ProviderCapabilities:
    return ProviderCapabilities(
        supports_streaming=False,
        supports_function_calling=function_calling,
        supports_structured_output=False,
        supports_vision=False,
        supports_system_prompt=True,
        max_context_window=8192,
        is_local=not function_calling,
    )


class MockLLMProvider(LLMProvider):
    def __init__(
        self,
        responses: list[LLMResponse],
        *,
        function_calling: bool,
        reject_tools: bool = False,
    ) -> None:
        self.responses = list(responses)
        self._capabilities = capabilities(function_calling=function_calling)
        self.reject_tools = reject_tools
        self.calls: list[dict] = []

    def provider_name(self) -> str:
        return "mock"

    def model_name(self) -> str:
        return "mock-agent-model"

    def get_capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    async def generate(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {"messages": list(messages), "config": config, "tools": tools}
        )
        if tools and self.reject_tools:
            self.reject_tools = False
            raise CapabilityNotSupportedError(
                "function_calling",
                provider=self.provider_name(),
                model=self.model_name(),
            )
        return self.responses.pop(0)

    async def stream(
        self,
        messages: list[ChatMessage],
        config: GenerationConfig | None = None,
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        raise CapabilityNotSupportedError("streaming", provider="mock")
        yield  # pragma: no cover

    async def test_connection(self) -> ConnectionTestResult:
        return ConnectionTestResult(
            success=True,
            latency_ms=0.0,
            provider=self.provider_name(),
            model=self.model_name(),
            capabilities=self._capabilities,
        )


def lookup_registry(observations: list[str]) -> AgentToolRegistry:
    async def lookup(query: str) -> dict:
        observations.append(query)
        return {"evidence": "Revenue was 42 million"}

    return AgentToolRegistry(
        [
            AgentTool.create(
                "search_knowledge_base",
                "Search evidence",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                lookup,
            )
        ]
    )


async def test_react_agent_uses_native_tool_calling():
    observations: list[str] = []
    provider = MockLLMProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        id="call-1",
                        name="search_knowledge_base",
                        arguments={"query": "revenue"},
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content="Revenue was 42 million [1]."),
        ],
        function_calling=True,
    )
    agent = ReActAgent(provider, lookup_registry(observations))

    result = await agent.run("What was revenue?")

    assert result.answer == "Revenue was 42 million [1]."
    assert result.used_native_tool_calling is True
    assert result.tool_calls_made == 1
    assert result.iterations == 2
    assert observations == ["revenue"]
    assert provider.calls[0]["tools"][0].name == "search_knowledge_base"
    tool_message = provider.calls[1]["messages"][-1]
    assert tool_message.tool_result is not None
    assert "42 million" in tool_message.tool_result.content


async def test_react_text_fallback_handles_think_tags_and_markdown_action():
    observations: list[str] = []
    provider = MockLLMProvider(
        [
            LLMResponse(
                content=(
                    "<think>I should search privately.</think>\n"
                    "Thought: Need evidence\nAction:\n```json\n"
                    '{"tool":"search_knowledge_base",'
                    '"arguments":{"query":"revenue"}}\n```'
                )
            ),
            LLMResponse(
                content=(
                    "<think>The observation is enough.</think>\n"
                    "Final Answer: Revenue was 42 million [1]."
                )
            ),
        ],
        function_calling=False,
    )
    agent = ReActAgent(provider, lookup_registry(observations))

    result = await agent.run("What was revenue?")

    assert result.answer == "Revenue was 42 million [1]."
    assert "think" not in result.answer.lower()
    assert result.used_native_tool_calling is False
    assert result.tool_calls_made == 1
    assert observations == ["revenue"]
    assert provider.calls[0]["tools"] is None
    assert "Observation:" in provider.calls[1]["messages"][-1].content


async def test_runtime_capability_error_triggers_text_fallback():
    provider = MockLLMProvider(
        [LLMResponse(content="Final Answer: fallback worked")],
        function_calling=True,
        reject_tools=True,
    )
    agent = ReActAgent(provider, lookup_registry([]))

    result = await agent.run("question")

    assert result.answer == "fallback worked"
    assert result.used_native_tool_calling is False
    assert provider.calls[0]["tools"] is not None
    assert provider.calls[1]["tools"] is None


async def test_max_iterations_stops_tool_loop():
    responses = [
        LLMResponse(
            content=(
                'Thought: repeat\nAction: {"tool":"search_knowledge_base",'
                '"arguments":{"query":"same"}}'
            )
        )
        for _ in range(2)
    ]
    agent = ReActAgent(
        MockLLMProvider(responses, function_calling=False),
        lookup_registry([]),
        max_iterations=2,
    )

    with pytest.raises(AgentIterationLimitError, match="maximum of 2"):
        await agent.run("loop")


async def test_sandbox_blocks_dangerous_import_and_external_open(tmp_path: Path):
    sandbox = LocalPythonSandbox(data_dir=tmp_path, timeout_seconds=1)

    imported = await sandbox.execute("import subprocess\nprint('bad')")
    target_path = "C:/Windows/win.ini" if os.name == "nt" else "/etc/passwd"
    opened = await sandbox.execute(f"print(open('{target_path}').read())")

    assert imported.success is False
    assert "not allowed" in (imported.error or "")
    assert opened.success is False
    assert "scratch directory" in (opened.error or "")


async def test_sandbox_terminates_infinite_loop(tmp_path: Path):
    sandbox = LocalPythonSandbox(data_dir=tmp_path, timeout_seconds=0.2)

    result = await sandbox.execute("while True:\n    pass")

    assert result.success is False
    assert result.timed_out is True
    assert "TimeoutError" in (result.error or "")


async def test_sandbox_runs_pandas_numpy_aggregation(tmp_path: Path):
    sandbox = LocalPythonSandbox(data_dir=tmp_path, timeout_seconds=15)
    code = """
import pandas as pd
import numpy as np
frame = pd.DataFrame({"revenue": [10, 20, 30], "cost": [4, 5, 6]})
profit = np.sum(frame["revenue"].to_numpy() - frame["cost"].to_numpy())
print(int(profit))
"""

    result = await sandbox.execute(code)

    assert result.success is True, result.error
    assert result.output == "45"


def citation_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="chunk-1",
            content="The annual report states that revenue increased to 42 million in 2025.",
            metadata={
                "document_id": "doc-1",
                "source_file": "annual-report.pdf",
                "page_number": 7,
                "chunk_index": 3,
            },
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            content="Operating expenses declined by five percent.",
            metadata={
                "document_id": "doc-2",
                "source_file": "operations.pdf",
                "page_number": 2,
                "chunk_index": 1,
            },
        ),
    ]


def test_citation_validator_attributes_exact_chunk_and_verbatim_quote():
    answer = 'Revenue "increased to 42 million" in 2025 [1].'

    result = CitationValidator().validate_citations(answer, citation_chunks())

    assert result.valid is True
    assert len(result.citations) == 1
    citation = result.citations[0]
    assert citation.marker == "[1]"
    assert citation.chunk_id == "chunk-1"
    assert citation.document_id == "doc-1"
    assert citation.source_file == "annual-report.pdf"
    assert citation.page_number == 7
    assert citation.quote_verified is True
    assert citation.quote_snippet in citation_chunks()[0].content


def test_citation_validator_flags_and_strips_hallucinated_citations():
    answer = "Revenue increased to 42 million [1], but satellites doubled [99]."

    result = CitationValidator().validate_citations(
        answer,
        citation_chunks(),
        strip_invalid=True,
    )

    assert result.valid is False
    assert result.phantom_citations == [99]
    assert result.hallucinated_citations == [99]
    assert "[99]" not in result.sanitized_answer
    assert "[1]" in result.sanitized_answer


def test_citation_validator_flags_unsupported_real_index():
    answer = "The company launched a spacecraft to Mars [2]."

    result = CitationValidator(minimum_overlap=0.2).validate_citations(
        answer,
        citation_chunks(),
    )

    assert result.phantom_citations == []
    assert result.unsupported_citations == [2]
    assert result.citations[0].valid is False
