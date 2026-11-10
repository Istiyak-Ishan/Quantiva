import json
import logging
from typing import List, Dict, Any
from app.services.agents.base_agent import BaseAgent
from app.schemas.agent import AgentAnswer, AgentResult, Citation
from app.services.tools.registry import ToolRegistry
from app.services.retrieval.hybrid_retriever import hybrid_retriever
from app.services.llm.output_parser import parse_structured_output
from app.services.llm.errors import AgentStepLimitError, ModelResponseError, ProviderError
from app.services.llm.gemini_client import gemini_client
from app.services.llm.output_parser import answer_string_prefix

logger = logging.getLogger(__name__)

class ReasoningAgent(BaseAgent):
    MAX_RETRIEVAL_CALLS = 3

    def __init__(self, doc_ids: List[str], db_chunks: List[Any], *, user_id: str, document_names=None, retriever=None):
        self.user_id = user_id
        self.document_names = document_names or {}
        self.chart_urls: set[str] = set()
        self.retriever = retriever if retriever is not None else hybrid_retriever
        self.doc_ids = doc_ids
        self.db_chunks = db_chunks
        self.citations: List[Dict[str, Any]] = []
        # Bound the tool loop: repeated retrieval calls grow the prompt until the
        # agent exhausts its step budget. Cache repeats and never append evidence
        # that was already sent to the model.
        self._retrieval_calls = 0
        self._retrieval_cache: Dict[str, str] = {}
        self._seen_cites: set[tuple] = set()
        
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
        normalized = " ".join(search_query.lower().split())
        if normalized in self._retrieval_cache:
            return (
                "Those results were already returned above. "
                "Use the existing evidence and produce the final JSON answer now."
            )
        if self._retrieval_calls >= self.MAX_RETRIEVAL_CALLS:
            return (
                "The retrieval limit for this question has been reached. "
                "Use the evidence already returned and produce the final JSON answer now."
            )
        self._retrieval_calls += 1

        # 1. Lookup filenames for active doc_ids to enrich citations and match targets
        filename_map = self.document_names
        id_by_filename = {name.lower(): doc_id for doc_id, name in filename_map.items()}

        q_lower = search_query.lower()
        search_doc_ids = list(self.doc_ids)

        # 2. Check if the user query explicitly targets a specific document
        matched_target_ids = []
        for fname_lower, d_id in id_by_filename.items():
            base_clean = fname_lower.replace(".pdf", "").replace(".docx", "").replace(".csv", "").replace(".xlsx", "")
            tokens = [t for t in base_clean.replace("-", " ").replace("_", " ").split() if len(t) >= 4]
            if any(t in q_lower for t in tokens) or base_clean in q_lower or fname_lower in q_lower:
                matched_target_ids.append(d_id)

        if matched_target_ids:
            search_doc_ids = matched_target_ids

        is_summary_query = any(k in q_lower for k in ["summarize", "summary", "overview", "what is this paper", "abstract", "key findings"])
        is_all_docs_query = any(k in q_lower for k in ["all the uploaded", "all uploaded", "all pdf", "all documents", "all files", "every document", "everything"])

        # 3. Perform hybrid search
        scored_chunks = await self.retriever.search(
            query=search_query, 
            doc_ids=search_doc_ids, 
            db_chunks=self.db_chunks, 
            top_k=8
            , user_id=self.user_id
        )
        
        # 4. If summary query, also seed with leading overview/abstract chunks from DB
        combined_payloads: List[Dict[str, Any]] = []
        seen_texts = set()

        if (is_all_docs_query or is_summary_query) and self.db_chunks:
            # Group db_chunks by doc_id and select substantive introductory and abstract chunks
            target_ids_set = set(search_doc_ids)
            for c in self.db_chunks:
                c_doc_id = str(getattr(c, "doc_id", ""))
                c_idx = getattr(c, "chunk_index", 999)
                txt = getattr(c, "text", getattr(c, "text_preview", "")).strip()
                if c_doc_id in target_ids_set and c_idx < 15 and len(txt) > 60:
                    if txt[:80] not in seen_texts:
                        seen_texts.add(txt[:80])
                        combined_payloads.append({
                            "payload": {
                                "doc_id": c_doc_id,
                                "filename": filename_map.get(c_doc_id, "Document"),
                                "page": getattr(c, "page_number", 1),
                                "text": txt
                            },
                            "score": 0.95
                        })

        for sc in scored_chunks:
            payload = sc.get("payload", {})
            txt = payload.get("text", "").strip()
            if txt and txt[:80] not in seen_texts:
                seen_texts.add(txt[:80])
                combined_payloads.append(sc)

        context_text = ""
        seen_cites = self._seen_cites

        for idx, sc in enumerate(combined_payloads):
            payload = sc.get("payload", {})
            page = payload.get("page", "Unknown")
            text = payload.get("text", "").strip()
            d_id = str(payload.get("doc_id", ""))
            fname = filename_map.get(d_id, payload.get("filename", "Document"))
            
            cite_key = (fname, str(page), text[:60])
            if cite_key in seen_cites:
                continue
            seen_cites.add(cite_key)
            self.citations.append({
                "doc_id": d_id,
                "filename": fname,
                "page_or_row": str(page),
                "text_excerpt": text[:200],
                "relevance_score": sc.get("score", 0.0)
            })
            
            context_text += f"\n--- [Source {len(self.citations)} | File: {fname} | Page {page}] ---\n{text}\n"
            
        if not context_text:
            self._retrieval_cache[normalized] = ""
            return "No new relevant chunks found. Use the evidence already provided."

        self._retrieval_cache[normalized] = context_text
        return context_text

    async def run(self, query: str, context: Dict[str, Any] = None, session_id: str = None, history=None, on_token=None) -> AgentResult:
        from app.services.memory.session_memory import session_memory
        from app.services.llm.context_manager import context_budget_manager
        from app.services.memory.memory_pager import memory_pager
        
        # 1. Load History
        if history is None and session_id:
            history = await session_memory.load_history(session_id, max_turns=10)
        history = history or []
            
        # 2. Context Budget Management for History
        # We don't have chunks yet (that happens in retrieve tool), but we trim history first.
        plan, trimmed_history, _ = context_budget_manager.allocate(self.model_name, history, [])
        if session_id and len(trimmed_history) < len(history):
            omitted = history[:len(history) - len(trimmed_history)]
            try:
                # Summarization is optional. If it fails, the deterministic
                # truncated history is still sent; no fabricated summary.
                trimmed_history = [await memory_pager.summarize(session_id, omitted), *trimmed_history]
            except Exception:
                logger.warning("History summarization unavailable; using bounded history", exc_info=False)
        
        # 3. Run the Base Agent loop. A malformed or empty model response is
        # retried once when nothing was shown to the user yet; visible partial
        # answers are never duplicated by a blind retry.
        emitted = ""

        async def emit_prefix(raw):
            nonlocal emitted
            prefix = answer_string_prefix(raw)
            if prefix.startswith(emitted) and len(prefix) > len(emitted):
                delta, emitted = prefix[len(emitted):], prefix
                await on_token(delta)

        stream_callback = emit_prefix if on_token is not None else None
        base_result = None
        structured_answer = None
        for attempt in range(2):
            emitted = ""
            try:
                base_result = await super().run_loop(query, context, trimmed_history, on_text=stream_callback)
                # 4. Parse output
                structured_answer = parse_structured_output(base_result.answer, AgentAnswer)
                break
            except AgentStepLimitError:
                raise
            except ModelResponseError:
                if attempt == 0 and not emitted:
                    logger.warning("Invalid model response with no visible answer; retrying once")
                    continue
                raise
            except ProviderError:
                # Provider outages already carry an accurate, user-safe message.
                # Never relabel them as malformed output or retry them blind.
                raise
            except Exception as exc:
                if attempt == 0 and not emitted:
                    logger.warning("Structured output parse failed with no visible answer; retrying once: %s", exc)
                    continue
                raise ModelResponseError() from exc

        if base_result is None or structured_answer is None:
            raise ModelResponseError()

        # Usage, model identity, and sources are server facts, never model guesses.
        structured_answer.citations = [Citation.model_validate(c) for c in self.citations]
        structured_answer.tokens_used = base_result.total_tokens
        structured_answer.input_tokens = base_result.input_tokens
        structured_answer.output_tokens = base_result.output_tokens
        structured_answer.cost_usd = gemini_client.calculate_cost(base_result.input_tokens, base_result.output_tokens)
        structured_answer.model_used = self.model_name
        if structured_answer.chart_url not in self.chart_urls:
            structured_answer.chart_url = None
            
        result = AgentResult(
            answer=structured_answer,
            tool_calls_made=base_result.tool_calls_made,
            iterations=base_result.iterations
        )
        
        return result

class DataAnalystAgent(ReasoningAgent):
    def __init__(self, doc_ids: List[str], db_chunks: List[Any], *, user_id: str, document_names=None, retriever=None):
        super().__init__(doc_ids, db_chunks, user_id=user_id, document_names=document_names, retriever=retriever)
        from app.services.tools.python_executor import python_executor
        
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
            func=self.generate_chart,
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

    async def generate_chart(self, **kwargs):
        from app.services.tools.chart_generator import chart_generator
        result = await chart_generator.generate(**kwargs, owner_id=self.user_id)
        if result.get("success") and result.get("chart_url"):
            self.chart_urls.add(result["chart_url"])
        return result
