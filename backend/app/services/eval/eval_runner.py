import uuid
from typing import List, Dict, Any
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import asyncio

from app.services.eval.eval_dataset import QAPair
from app.services.eval.llm_judge import llm_judge
from app.services.agents.orchestrator import orchestrator
from app.models.chunk import Chunk

class EvalReport(BaseModel):
    avg_faithfulness: float
    avg_relevance: float
    avg_completeness: float
    per_question_scores: List[Dict[str, Any]]

class EvalRunner:
    async def run(self, dataset: List[QAPair], doc_ids: List[str], db: AsyncSession) -> EvalReport:
        """
        Runs the full evaluation pipeline over a golden dataset.
        For each question:
        1. Query the orchestrator
        2. Pass the orchestrator's answer and retrieved context to the LLM judge
        3. Collect scores
        """
        
        chunk_res = await db.execute(select(Chunk).where(Chunk.doc_id.in_(doc_ids)))
        db_chunks = chunk_res.scalars().all()
        
        all_scores = []
        
        for pair in dataset:
            # 1. Run the agent
            # We generate a unique session ID for each query to avoid history contamination
            fake_session_id = str(uuid.uuid4())
            agent_result = await orchestrator.run(
                query=pair.question,
                session_id=fake_session_id,
                doc_ids=doc_ids,
                db_chunks=db_chunks
            )
            
            answer_text = agent_result.answer.answer
            
            # Combine retrieved contexts into one string
            retrieved_context = "\n".join([c.text for c in db_chunks])
            
            # 2. Judge the output
            score = llm_judge.score(
                question=pair.question,
                retrieved_context=retrieved_context,
                answer=answer_text
            )
            
            all_scores.append({
                "question": pair.question,
                "expected": pair.expected_answer,
                "agent_answer": answer_text,
                "faithfulness": score.faithfulness,
                "relevance": score.relevance,
                "completeness": score.completeness,
                "reasoning": score.reasoning
            })
            
        if not all_scores:
            return EvalReport(avg_faithfulness=0.0, avg_relevance=0.0, avg_completeness=0.0, per_question_scores=[])
            
        avg_f = sum(s["faithfulness"] for s in all_scores) / len(all_scores)
        avg_r = sum(s["relevance"] for s in all_scores) / len(all_scores)
        avg_c = sum(s["completeness"] for s in all_scores) / len(all_scores)
        
        return EvalReport(
            avg_faithfulness=avg_f,
            avg_relevance=avg_r,
            avg_completeness=avg_c,
            per_question_scores=all_scores
        )

eval_runner = EvalRunner()
