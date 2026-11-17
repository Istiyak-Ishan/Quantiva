"""Provider-agnostic agent tool registry and controlled Quantiva tools."""
from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, cast

from app.providers.interfaces.types import ToolDefinition
from app.retrieval.pipeline import RetrievalPipeline
from app.security.sandbox import LocalPythonSandbox

ToolHandler = Callable[..., Any | Awaitable[Any]]


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(cast(Any, value))
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def serialize_tool_result(value: Any) -> str:
    """Serialize tool output deterministically for an LLM observation."""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=_json_default, ensure_ascii=False, sort_keys=True)


@dataclass(slots=True)
class AgentTool:
    """A provider-facing schema paired with its local implementation."""

    definition: ToolDefinition
    handler: ToolHandler

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: ToolHandler,
    ) -> AgentTool:
        return cls(
            definition=ToolDefinition(
                name=name,
                description=description,
                parameters_json_schema=parameters,
            ),
            handler=handler,
        )


class AgentToolRegistry:
    """Small async registry used by both native and text ReAct execution."""

    def __init__(
        self,
        tools: Sequence[AgentTool] | Mapping[str, ToolHandler | AgentTool] | None = None,
    ) -> None:
        self._tools: dict[str, AgentTool] = {}
        if isinstance(tools, Mapping):
            for name, value in tools.items():
                if isinstance(value, AgentTool):
                    self.register(value)
                else:
                    self.register(
                        AgentTool.create(
                            name,
                            f"Execute the {name} tool.",
                            {"type": "object", "properties": {}},
                            value,
                        )
                    )
        else:
            for tool in tools or ():
                self.register(tool)

    def register(self, tool: AgentTool) -> None:
        if not tool.definition.name:
            raise ValueError("Tool name must be non-empty")
        self._tools[tool.definition.name] = tool

    @property
    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition for tool in self._tools.values()]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    async def execute(self, name: str, arguments: Mapping[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return serialize_tool_result(
                {"success": False, "error": f"Unknown tool: {name}"}
            )
        try:
            result = tool.handler(**dict(arguments))
            if inspect.isawaitable(result):
                result = await result
            return serialize_tool_result(result)
        except Exception as exc:  # tool errors become observations, not loop crashes
            return serialize_tool_result(
                {
                    "success": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                }
            )


class ControlledAgentTools:
    """Factory for Quantiva's controlled retrieval and analysis tools.

    All persistence/model dependencies are injected.  ``table_extractor`` may
    be a sync or async callback accepting ``document_id`` and optional
    ``table_indices`` keyword arguments.
    """

    def __init__(
        self,
        *,
        retrieval_pipeline: RetrievalPipeline | None = None,
        sandbox: LocalPythonSandbox | None = None,
        table_extractor: ToolHandler | None = None,
        knowledge_base_id: str | None = None,
    ) -> None:
        self.retrieval_pipeline = retrieval_pipeline
        self.sandbox = sandbox or LocalPythonSandbox()
        self.table_extractor = table_extractor
        self.knowledge_base_id = knowledge_base_id

    async def search_knowledge_base(
        self,
        query: str,
        knowledge_base_id: str | None = None,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        if self.retrieval_pipeline is None:
            return {"success": False, "error": "Retrieval pipeline is unavailable"}
        filters: dict[str, Any] = {}
        if document_ids:
            filters["document_ids"] = [str(value) for value in document_ids]
        result = await self.retrieval_pipeline.retrieve(
            query,
            knowledge_base_id=knowledge_base_id or self.knowledge_base_id,
            filters=filters or None,
        )
        chunks = result.chunks[:top_k] if top_k else result.chunks
        return {
            "success": True,
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "score": chunk.score,
                    "metadata": dict(chunk.metadata),
                }
                for chunk in chunks
            ],
        }

    async def compare_documents(
        self,
        query: str,
        document_ids: list[str],
        knowledge_base_id: str | None = None,
        top_k_per_document: int = 4,
    ) -> dict[str, Any]:
        if len(document_ids) < 2:
            return {
                "success": False,
                "error": "compare_documents requires at least two document_ids",
            }
        comparisons: dict[str, Any] = {}
        for document_id in document_ids:
            result = await self.search_knowledge_base(
                query,
                knowledge_base_id=knowledge_base_id,
                document_ids=[document_id],
                top_k=top_k_per_document,
            )
            comparisons[str(document_id)] = result.get("chunks", [])
        return {"success": True, "documents": comparisons}

    async def extract_document_tables(
        self,
        document_id: str,
        table_indices: list[int] | None = None,
    ) -> Any:
        if self.table_extractor is None:
            return {"success": False, "error": "Table extractor is unavailable"}
        result = self.table_extractor(
            document_id=document_id,
            table_indices=table_indices,
        )
        if inspect.isawaitable(result):
            result = await result
        return result

    async def run_python_analysis(self, code: str) -> dict[str, Any]:
        return (await self.sandbox.execute(code)).to_dict()

    def registry(self) -> AgentToolRegistry:
        object_schema = {"type": "object", "additionalProperties": False}
        tools = [
            AgentTool.create(
                "search_knowledge_base",
                "Search indexed document chunks for evidence relevant to a query.",
                {
                    **object_schema,
                    "properties": {
                        "query": {"type": "string"},
                        "knowledge_base_id": {"type": "string"},
                        "document_ids": {"type": "array", "items": {"type": "string"}},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 25},
                    },
                    "required": ["query"],
                },
                self.search_knowledge_base,
            ),
            AgentTool.create(
                "compare_documents",
                "Retrieve comparable evidence from two or more documents.",
                {
                    **object_schema,
                    "properties": {
                        "query": {"type": "string"},
                        "document_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                        },
                        "knowledge_base_id": {"type": "string"},
                        "top_k_per_document": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["query", "document_ids"],
                },
                self.compare_documents,
            ),
            AgentTool.create(
                "extract_document_tables",
                "Extract structured tables from a document for numerical analysis.",
                {
                    **object_schema,
                    "properties": {
                        "document_id": {"type": "string"},
                        "table_indices": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["document_id"],
                },
                self.extract_document_tables,
            ),
            AgentTool.create(
                "run_python_analysis",
                "Run deterministic Python calculations in the local restricted sandbox.",
                {
                    **object_schema,
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
                self.run_python_analysis,
            ),
        ]
        return AgentToolRegistry(tools)


__all__ = [
    "AgentTool",
    "AgentToolRegistry",
    "ControlledAgentTools",
    "serialize_tool_result",
]
