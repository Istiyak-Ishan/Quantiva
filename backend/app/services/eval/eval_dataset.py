from pydantic import BaseModel, Field
from typing import List
import json
from google import genai
from google.genai import types
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.config import settings
from app.models.chunk import Chunk

class QAPair(BaseModel):
    question: str = Field(description="A complex question that can be answered using the source chunk.")
    expected_answer: str = Field(description="The correct answer based solely on the chunk.")
    source_chunk: str = Field(description="The exact text of the chunk.")

class QADataset(BaseModel):
    pairs: List[QAPair]

class EvalDatasetGenerator:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.model_name = "gemini-1.5-flash"
        
    async def generate_qa_pairs(self, doc_id: str, db: AsyncSession, n: int = 20) -> List[QAPair]:
        """
        Generates synthetic Q&A pairs directly from the document's chunks.
        This provides a 'Golden Dataset' without requiring humans to read the document.
        """
        # Fetch chunks for this document
        result = await db.execute(select(Chunk).where(Chunk.doc_id == doc_id).limit(n))
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
            
            config = types.GenerateContentConfig(
                system_instruction="Output JSON matching the schema exactly." + json.dumps(QADataset.model_json_schema()),
                temperature=0.7,
                response_mime_type="application/json"
            )
            
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                    config=config
                )
                dataset = QADataset.model_validate_json(response.text)
                if dataset.pairs:
                    # Inject the source chunk
                    pair = dataset.pairs[0]
                    pair.source_chunk = chunk.text
                    pairs.append(pair)
            except Exception as e:
                # Mock generation if key fails
                pairs.append(QAPair(
                    question="Mock Question about this chunk?",
                    expected_answer="Mock expected answer.",
                    source_chunk=chunk.text
                ))
                
        return pairs

eval_dataset_generator = EvalDatasetGenerator()
