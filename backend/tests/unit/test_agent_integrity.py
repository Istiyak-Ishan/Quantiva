from unittest.mock import AsyncMock, patch

import pytest
from google.genai import types

from app.services.agents.base_agent import BaseAgent, AgentResult
from app.services.agents.reasoning_agent import ReasoningAgent
from app.services.llm.errors import AgentStepLimitError, LLMUnavailableError, ModelResponseError
from app.services.tools.registry import ToolRegistry
from app.schemas.agent import Citation


async def test_agent_propagates_provider_failure():
    agent = BaseAgent("system")
    with patch.object(agent.client, "generate", new_callable=AsyncMock, side_effect=LLMUnavailableError()):
        with pytest.raises(LLMUnavailableError):
            await agent.run_loop("question")


async def test_malformed_json_never_gets_random_confidence_or_fake_tokens():
    agent = ReasoningAgent([], [], user_id="00000000-0000-0000-0000-000000000001")
    with patch.object(BaseAgent, "run_loop", new_callable=AsyncMock, return_value=AgentResult(answer="not JSON", total_tokens=57)):
        with pytest.raises(ModelResponseError):
            await agent.run("question")


async def test_retrieved_citations_remain_validated_objects_not_raw_dicts():
    agent = ReasoningAgent(["doc"], [], user_id="owner")
    agent.citations = [{"doc_id": "doc", "filename": "source.pdf", "page_or_row": 1, "text_excerpt": "evidence", "relevance_score": .9}]
    with patch.object(BaseAgent, "run_loop", new_callable=AsyncMock, return_value=AgentResult(answer='{"answer":"Evidence based answer","confidence":0.9}', total_tokens=57)):
        result = await agent.run("question")
    assert isinstance(result.answer.citations[0], Citation)
    assert result.answer.citations[0].model_dump()["doc_id"] == "doc"
    assert result.answer.tokens_used == 57


async def test_tool_turn_preserves_thought_signature():
    registry = ToolRegistry()
    registry.register("lookup", AsyncMock(return_value="evidence"), {})
    agent = BaseAgent("system", registry)
    content = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="lookup", args={}), thought_signature=b"signature")])
    tool_response = types.GenerateContentResponse(candidates=[types.Candidate(content=content)])
    final_response = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part.from_text(text="done")]))])

    async def generate(contents, config, *, usage):
        if len(contents) == 1:
            return tool_response
        assert contents[1].parts[0].thought_signature == b"signature"
        usage.total_tokens = 42
        return final_response

    with patch.object(agent.client, "generate", side_effect=generate):
        result = await agent.run_loop("question")
    assert result.answer == "done"
    assert result.total_tokens == 42
    assert result.tool_calls_made == 1


async def test_step_exhaustion_has_an_actionable_error_not_a_generic_one():
    registry = ToolRegistry()
    registry.register("lookup", lambda: "evidence", {"description": "lookup"})
    agent = BaseAgent("system", registry)
    content = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="lookup", args={}))])
    tool_response = types.GenerateContentResponse(candidates=[types.Candidate(content=content)])

    with patch.object(agent.client, "generate", new_callable=AsyncMock, return_value=tool_response):
        with pytest.raises(AgentStepLimitError, match="allowed number of steps"):
            await agent.run_loop("question")


async def test_invalid_response_is_retried_once_when_nothing_was_streamed():
    agent = ReasoningAgent([], [], user_id="owner")
    calls = 0

    async def run_loop(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ModelResponseError()
        return AgentResult(answer='{"answer":"ok","confidence":0.9}', total_tokens=5)

    with patch.object(BaseAgent, "run_loop", side_effect=run_loop):
        result = await agent.run("question")

    assert calls == 2
    assert result.answer.answer == "ok"


async def test_visible_partial_answer_is_never_duplicated_by_a_retry():
    agent = ReasoningAgent([], [], user_id="owner")
    calls = 0
    emitted = []

    async def run_loop(*args, on_text=None, **kwargs):
        nonlocal calls
        calls += 1
        await on_text('{"answer":"partial')
        raise ModelResponseError()

    async def collect(delta):
        emitted.append(delta)

    with patch.object(BaseAgent, "run_loop", side_effect=run_loop):
        with pytest.raises(ModelResponseError):
            await agent.run("question", on_token=collect)

    assert calls == 1
    assert "".join(emitted) == "partial"


def test_lookup_questions_use_reasoning_and_explicit_math_uses_data_analyst():
    from app.services.agents.orchestrator import AgentOrchestrator

    router = AgentOrchestrator()
    assert router.route("What were Apple's total assets as of March 29, 2025?") == "reasoning"
    assert router.route("How much revenue did Apple's Services segment generate?") == "reasoning"
    assert router.route("Calculate the year over year growth rate") == "data_analyst"
    assert router.route("Plot revenue by quarter") == "data_analyst"


async def test_provider_outage_is_not_relabelled_as_invalid_output():
    agent = ReasoningAgent([], [], user_id="owner")
    calls = 0

    async def run_loop(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise LLMUnavailableError()

    with patch.object(BaseAgent, "run_loop", side_effect=run_loop):
        with pytest.raises(LLMUnavailableError, match="unavailable"):
            await agent.run("question")

    assert calls == 1


class _FakeRetriever:
    def __init__(self, repeated=False):
        self.calls = 0
        self.repeated = repeated

    async def search(self, **kwargs):
        self.calls += 1
        text = "same evidence" if self.repeated else f"evidence {self.calls}"
        return [{"payload": {"doc_id": "doc", "filename": "source.pdf", "page": 1, "text": text}, "score": 1.0}]


async def test_retrieval_calls_are_capped_and_repeats_use_the_cache():
    retriever = _FakeRetriever()
    agent = ReasoningAgent(["doc"], [], user_id="owner", retriever=retriever)

    await agent.retrieve_chunks("first question")
    cached = await agent.retrieve_chunks("First   Question")
    await agent.retrieve_chunks("second question")
    await agent.retrieve_chunks("third question")
    capped = await agent.retrieve_chunks("fourth question")

    assert "already returned" in cached
    assert "retrieval limit" in capped
    assert retriever.calls == 3
    assert len(agent.citations) == 3


async def test_duplicate_chunks_are_not_appended_to_the_prompt_twice():
    retriever = _FakeRetriever(repeated=True)
    agent = ReasoningAgent(["doc"], [], user_id="owner", retriever=retriever)

    first = await agent.retrieve_chunks("first question")
    second = await agent.retrieve_chunks("second question")

    assert "same evidence" in first
    assert second == "No new relevant chunks found. Use the evidence already provided."
    assert len(agent.citations) == 1
