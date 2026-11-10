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
from app.models.document import Document
from app.core.ownership import owned_documents

class EvalReport(BaseModel):
    avg_faithfulness: float
    avg_relevance: float
    avg_completeness: float
    per_question_scores: List[Dict[str, Any]]

class EvalRunner:
    async def run(self, dataset: List[QAPair], doc_ids: List[str], db: AsyncSession, *, user_id: str, retriever=None) -> EvalReport:
        """
        Runs the full evaluation pipeline over a golden dataset.
        For each question:
        1. Query the orchestrator
        2. Pass the orchestrator's answer and retrieved context to the LLM judge
        3. Collect scores
        """
        
        ids = [uuid.UUID(d) for d in doc_ids]
        docs = await owned_documents(db, ids, uuid.UUID(user_id))
        chunk_res = await db.execute(select(Chunk).join(Document, Chunk.doc_id == Document.id).where(Chunk.doc_id.in_(ids), Document.user_id == uuid.UUID(user_id)))
        db_chunks = list(chunk_res.scalars().all())
        
        all_scores: List[Dict[str, Any]] = []
        
        for pair in dataset:
            # 1. Run the agent
            # We generate a unique session ID for each query to avoid history contamination
            fake_session_id = str(uuid.uuid4())
            agent_result = await orchestrator.run(
                query=pair.question,
                session_id=fake_session_id,
                doc_ids=doc_ids,
                db_chunks=db_chunks
                , user_id=user_id, history=[], use_cache=False, document_names={str(d.id): d.filename for d in docs}, retriever=retriever
            )
            
            answer_text = agent_result.answer.answer
            
            # Combine retrieved contexts into one string
            retrieved_context = "\n".join(c.text_excerpt for c in agent_result.answer.citations)
            
            # 2. Judge the output
            score = await asyncio.to_thread(llm_judge.score,
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
