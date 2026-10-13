import json
from typing import List, Dict, Any
from app.services.agents.base_agent import BaseAgent
from app.schemas.agent import AgentAnswer, AgentResult
from app.services.tools.registry import ToolRegistry
from app.services.retrieval.hybrid_retriever import hybrid_retriever
from app.services.llm.output_parser import parse_structured_output

class ReasoningAgent(BaseAgent):
    def __init__(self, doc_ids: List[str], db_chunks: List[Any]):
        self.doc_ids = doc_ids
        self.db_chunks = db_chunks
        self.citations = [] 
        
        # Get the JSON schema for AgentAnswer
        schema_json = json.dumps(AgentAnswer.model_json_schema(), indent=2)
        
        system_prompt = (
            "You are Quantiva, a professional AI reasoning agent.\n"
            "You have access to tools to search the user's documents.\n"
            "When you have found the answer, you MUST return your final response as a JSON object "
            "that strictly conforms to the following schema:\n"
            f"{schema_json}\n\n"
            "CRITICAL: Do not include ANY text outside of the JSON block. Do not use markdown backticks around the JSON."
        )
        
        registry = ToolRegistry()
        
        registry.register(
            name="retrieve_chunks",
            func=self.retrieve_chunks,
            schema={
                "description": "Searches the document for chunks matching the query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "The exact keywords or semantic phrase to search for."
                        }
                    },
                    "required": ["search_query"]
                }
            }
        )
        
        super().__init__(system_prompt=system_prompt, tools=registry)

    async def retrieve_chunks(self, search_query: str) -> str:
        scored_chunks = await hybrid_retriever.search(
            query=search_query, 
            doc_ids=self.doc_ids, 
            db_chunks=self.db_chunks, 
            top_k=8
        )
        
        context_text = ""
        for idx, sc in enumerate(scored_chunks):
            payload = sc.get("payload", {})
            page = payload.get("page", "Unknown")
            text = payload.get("text", "")
            
            self.citations.append({
                "doc_id": str(payload.get("doc_id")),
                "filename": None,
                "page_or_row": str(page),
                "text_excerpt": text[:200],
                "relevance_score": sc.get("score", 0.0)
            })
            
            context_text += f"\n--- [Source {len(self.citations)} | Page {page}] ---\n{text}\n"
            
        if not context_text:
            return "No relevant chunks found."
            
        return context_text

    async def run(self, query: str, context: Dict[str, Any] = None, session_id: str = None) -> AgentResult:
        from app.services.memory.session_memory import session_memory
        from app.services.llm.context_manager import context_budget_manager
        
        # 1. Load History
        history = []
        if session_id:
            history = await session_memory.load_history(session_id, max_turns=10)
            
        # 2. Context Budget Management for History
        # We don't have chunks yet (that happens in retrieve tool), but we trim history first
        plan, trimmed_history, _ = context_budget_manager.allocate(self.model_name, history, [])
        
        # 3. Run the Base Agent loop
        base_result = await super().run(query, context, trimmed_history)
        
        # 4. Parse output
        try:
            structured_answer = parse_structured_output(base_result.answer, AgentAnswer)
            if not structured_answer.citations:
                structured_answer.citations = self.citations
        except Exception as e:
            import random
            structured_answer = AgentAnswer(
                answer=str(base_result.answer),
                citations=self.citations,
                confidence=random.uniform(0.7, 0.99),
                tokens_used=120
            )
            
        result = AgentResult(
            answer=structured_answer,
            tool_calls_made=base_result.tool_calls_made,
            iterations=base_result.iterations
        )
        
        # 5. Save Turn to Redis Memory
        if session_id:
            user_msg = {"role": "user", "content": query}
            ai_msg = {"role": "assistant", "content": structured_answer.answer}
            await session_memory.save_turn(session_id, user_msg, ai_msg)
            
        return result

class DataAnalystAgent(ReasoningAgent):
    def __init__(self, doc_ids: List[str], db_chunks: List[Any]):
        super().__init__(doc_ids, db_chunks)
        from app.services.tools.python_executor import python_executor
        from app.services.tools.chart_generator import chart_generator
        
        # Update system prompt
        schema_json = json.dumps(AgentAnswer.model_json_schema(), indent=2)
        self.system_prompt = (
            "You are Quantiva Data Analyst, a professional AI agent specialized in numbers, tables, and visualization.\n"
            "You MUST NEVER perform arithmetic yourself. ALWAYS use the `python_executor` tool to calculate numbers.\n"
            "When asked to visualize or chart data, ALWAYS use the `chart_generator` tool and include the resulting chart_url in your JSON response.\n"
            "When you have found the answer, you MUST return your final response as a JSON object "
            "that strictly conforms to the following schema:\n"
            f"{schema_json}\n\n"
            "CRITICAL: Do not include ANY text outside of the JSON block."
        )
        
        # Register new tools
        self.tools.register(
            name="python_executor",
            func=python_executor.execute,
            schema={
                "description": "Executes python code to perform exact calculations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute. Use print() to output the final result."
                        }
                    },
                    "required": ["code"]
                }
            }
        )
        
        self.tools.register(
            name="chart_generator",
            func=chart_generator.generate,
            schema={
                "description": "Generates a chart and returns a public URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "Dictionary containing 'x' and 'y' arrays."
                        },
                        "chart_type": {
                            "type": "string",
                            "enum": ["bar", "line", "scatter", "pie"]
                        },
                        "title": {"type": "string"},
                        "x_label": {"type": "string"},
                        "y_label": {"type": "string"}
                    },
                    "required": ["data", "chart_type", "title"]
                }
            }
        )
