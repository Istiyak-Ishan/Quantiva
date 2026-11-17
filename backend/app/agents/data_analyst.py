"""Sandboxed spreadsheet/table analysis agent."""
from __future__ import annotations

from collections.abc import Sequence

from app.agents.reasoning import ReActAgent
from app.agents.tools import AgentTool, AgentToolRegistry, ControlledAgentTools
from app.providers.interfaces.llm import LLMProvider
from app.providers.interfaces.types import GenerationConfig
from app.retrieval.pipeline import RetrievalPipeline
from app.security.sandbox import LocalPythonSandbox


class DataAnalystAgent(ReActAgent):
    """ReAct agent specialized in tables and exact numerical synthesis.

    The agent never executes Python in-process.  Its calculation tool always
    delegates to :class:`LocalPythonSandbox`.
    """

    SYSTEM_PROMPT = (
        "You are Quantiva Data Analyst. Inspect spreadsheet and document table "
        "evidence before calculating. Use extract_document_tables for structured "
        "table data and run_python_analysis for every non-trivial calculation. "
        "Use compare_documents for cross-document comparisons. Never guess a "
        "number, never execute code outside the provided sandbox, and cite the "
        "source chunks supporting each conclusion."
    )

    def __init__(
        self,
        llm_provider: LLMProvider,
        *,
        retrieval_pipeline: RetrievalPipeline | None = None,
        sandbox: LocalPythonSandbox | None = None,
        table_extractor=None,
        knowledge_base_id: str | None = None,
        tools: AgentToolRegistry | Sequence[AgentTool] | None = None,
        max_iterations: int = 5,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self.controlled_tools = ControlledAgentTools(
            retrieval_pipeline=retrieval_pipeline,
            sandbox=sandbox,
            table_extractor=table_extractor,
            knowledge_base_id=knowledge_base_id,
        )
        registry = tools or self.controlled_tools.registry()
        super().__init__(
            llm_provider,
            registry,
            system_prompt=self.SYSTEM_PROMPT,
            max_iterations=max_iterations,
            generation_config=generation_config,
        )


__all__ = ["DataAnalystAgent"]
