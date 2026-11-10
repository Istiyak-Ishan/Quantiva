from pydantic import BaseModel, Field
from typing import List
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.config import settings
from app.models.chunk import Chunk
from app.core.ownership import owned_documents
import uuid
import asyncio
from app.services.llm.gemini_client import gemini_client
from app.services.llm.errors import ModelResponseError

class QAPair(BaseModel):
    question: str = Field(description="A complex question that can be answered using the source chunk.")
    expected_answer: str = Field(description="The correct answer based solely on the chunk.")
    source_chunk: str = Field(description="The exact text of the chunk.")

class QADataset(BaseModel):
    pairs: List[QAPair]

class EvalDatasetGenerator:
    def __init__(self):
        self.client = gemini_client
        self.model_name = settings.GEMINI_CHAT_MODEL
        
    async def generate_qa_pairs(self, doc_id: str, db: AsyncSession, n: int = 20, *, user_id: str) -> List[QAPair]:
        """
        Generates synthetic Q&A pairs directly from the document's chunks.
        This provides a 'Golden Dataset' without requiring humans to read the document.
        """
        # Fetch chunks for this document
        await owned_documents(db, [uuid.UUID(doc_id)], uuid.UUID(user_id))
        result = await db.execute(select(Chunk).where(Chunk.doc_id == uuid.UUID(doc_id)).limit(n))
        chunks = result.scalars().all()
        
        if not chunks:
            return []
            
        pairs = []
        for chunk in chunks:
            prompt = (
                "You are an expert dataset creator. Generate ONE complex question that can be "
                f"answered using ONLY this text: \n\n{chunk.text}\n\n"
                "Then provide the exact correct answer."
            )
            
            raw = await asyncio.to_thread(self.client.complete, [{"role": "user", "content": prompt}], "Output JSON matching the schema exactly." + json.dumps(QADataset.model_json_schema()))
            try:
                dataset = QADataset.model_validate_json(raw)
                if dataset.pairs:
                    # Inject the source chunk
                    pair = dataset.pairs[0]
                    pair.source_chunk = chunk.text
                    pairs.append(pair)
            except ValueError as exc:
                raise ModelResponseError() from exc
                
        return pairs

eval_dataset_generator = EvalDatasetGenerator()
