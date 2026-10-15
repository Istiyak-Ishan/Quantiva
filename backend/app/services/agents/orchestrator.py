from typing import List, Any
from app.services.agents.reasoning_agent import ReasoningAgent, DataAnalystAgent
from app.schemas.agent import AgentResult

class AgentOrchestrator:
    def __init__(self):
        pass

    def route(self, query: str) -> str:
        """
        Simple heuristic router. If the user asks for charts, math, numbers, or percentages,
        we route them to the DataAnalystAgent. Otherwise, ReasoningAgent.
        """
        query_lower = query.lower()
        data_keywords = ["%", "percent", "calculate", "compare", "chart", "graph", "plot", "revenue", "math", "+", "-", "sum", "average", "total"]
        
        for kw in data_keywords:
            if kw in query_lower:
                return "data_analyst"
                
        return "reasoning"

    async def run(self, query: str, session_id: str, doc_ids: List[str], db_chunks: List[Any]) -> AgentResult:
        from app.services.memory.semantic_cache import semantic_cache
        from app.services.guardrails.guardrails_service import guardrails_service
        
        # 1. Check Input Guardrails
        await guardrails_service.check_input(query)
        
        # 2. Check Cache
        cached_answer = await semantic_cache.get(query, doc_ids)
        if cached_answer:
            print("[Orchestrator] Cache HIT!")
            return AgentResult(answer=cached_answer, tool_calls_made=0, iterations=0)
            
        print("[Orchestrator] Cache MISS. Routing query...")
        
        # 3. Route & Execute
        agent_type = self.route(query)
        print(f"[Orchestrator] Routing query to: {agent_type.upper()} AGENT")
        
        if agent_type == "data_analyst":
            agent = DataAnalystAgent(doc_ids=doc_ids, db_chunks=db_chunks)
        else:
            agent = ReasoningAgent(doc_ids=doc_ids, db_chunks=db_chunks)
            
        result = await agent.run(query=query, session_id=session_id)
        
        # 4. Check Output Guardrails
        source_chunks = [c.text for c in db_chunks] # passing actual chunk texts
        guard_result = await guardrails_service.check_output(result.answer.answer, source_chunks)
        
        # Update answer with redacted text and flags
        result.answer.answer = guard_result["answer"]
        result.answer.guardrail_flags = guard_result["flags"]
        if guard_result["hallucination_score"] < 0.6:
            result.answer.confidence = guard_result["hallucination_score"]
            
        # 5. Store in Cache
        await semantic_cache.set(query, doc_ids, result.answer)
        
        return result

orchestrator = AgentOrchestrator()
