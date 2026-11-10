from typing import List, Any
from app.services.agents.reasoning_agent import ReasoningAgent, DataAnalystAgent
from app.schemas.agent import AgentResult
import hashlib
import json
import re
from app.core.metrics import CACHE_HITS, CACHE_MISSES

class AgentOrchestrator:
    def __init__(self):
        pass

    def route(self, query: str) -> str:
        """
        Simple heuristic router. If the user asks for charts, math, numbers, or percentages,
        we route them to the DataAnalystAgent. Otherwise, ReasoningAgent.
        """
        query_lower = query.lower()
        # Only route on explicit computation/visualisation intent. Broad terms
        # such as "total" or "revenue" describe lookups, not arithmetic, and
        # sending them to the data agent added needless tool steps.
        data_keywords = ["%", "percent", "calculate", "compare", "chart", "graph", "plot", "math", "sum", "average", "ratio"]
        
        for kw in data_keywords:
            if kw in query_lower:
                return "data_analyst"
                
        return "reasoning"

    async def run(self, query: str, session_id: str, doc_ids: List[str], db_chunks: List[Any], *, user_id: str, history=None, on_token=None, document_names=None, use_cache=True, retriever=None) -> AgentResult:
        from app.services.memory.semantic_cache import semantic_cache
        from app.services.guardrails.guardrails_service import guardrails_service
        
        # 1. Check Input Guardrails
        await guardrails_service.check_input(query)
        
        # 2. Check Cache
        corpus_version = hashlib.sha256("|".join(sorted(str(c.id) for c in db_chunks)).encode()).hexdigest()
        context_fingerprint = hashlib.sha256(json.dumps(history or [], sort_keys=True).encode()).hexdigest()
        cache_scope = {"user_id": user_id, "corpus_version": corpus_version, "context_fingerprint": context_fingerprint}
        cached_answer = await semantic_cache.get(query, doc_ids, **cache_scope) if use_cache else None
        if cached_answer:
            CACHE_HITS.labels(cache_type="semantic").inc()
            if on_token is not None:
                await on_token(cached_answer.answer)
            return AgentResult(answer=cached_answer, tool_calls_made=0, iterations=0)
        if use_cache:
            CACHE_MISSES.labels(cache_type="semantic").inc()
            
        
        # 3. Route & Execute
        agent_type = self.route(query)
        agent: ReasoningAgent
        
        if agent_type == "data_analyst":
            agent = DataAnalystAgent(doc_ids=doc_ids, db_chunks=db_chunks, user_id=user_id, document_names=document_names, retriever=retriever)
        else:
            agent = ReasoningAgent(doc_ids=doc_ids, db_chunks=db_chunks, user_id=user_id, document_names=document_names, retriever=retriever)
            
        # Provider deltas are buffered to sentence/newline boundaries so an email
        # or identifier split across model chunks cannot leak before redaction.
        pending = ""

        async def safe_emit(text, *, flush=False):
            nonlocal pending
            from app.services.guardrails.pii_detector import pii_detector
            from app.services.guardrails.content_filter import content_filter
            from app.services.guardrails.guardrails_service import GuardrailBlockError
            pending += text
            while pending:
                boundary = re.search(r"\n|(?<=[.!?])\s", pending)
                if boundary is None and not flush:
                    break
                end = boundary.end() if boundary else len(pending)
                sentence, pending = pending[:end], pending[end:]
                content = await content_filter.check(sentence)
                if not content["is_safe"]:
                    raise GuardrailBlockError("Generated content was blocked", content["categories"])
                redacted = await pii_detector.scan_and_redact(sentence)
                await on_token(redacted["redacted_text"])

        result = await agent.run(query=query, session_id=session_id, history=history, on_token=safe_emit if on_token is not None else None)
        if on_token is not None:
            await safe_emit("", flush=True)
        
        # 4. Check Output Guardrails (ground against cited chunks or top candidates, not thousands of DB rows)
        source_chunks = [getattr(c, "text_excerpt", "") for c in result.answer.citations if getattr(c, "text_excerpt", "")]
        if not source_chunks and db_chunks:
            source_chunks = [getattr(c, "text", getattr(c, "text_preview", "")) for c in db_chunks[:5]]
        guard_result = await guardrails_service.check_output(result.answer.answer, source_chunks)
        
        # Update answer with redacted text and flags
        result.answer.answer = guard_result["answer"]
        result.answer.guardrail_flags = guard_result["flags"]
        # This is an embedding-grounding heuristic, not calibrated probability.
        result.answer.confidence = guard_result["hallucination_score"]
            
        # 5. Store in Cache
        if use_cache:
            await semantic_cache.set(query, doc_ids, result.answer, **cache_scope)
        
        return result

orchestrator = AgentOrchestrator()
