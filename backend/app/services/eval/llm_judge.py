from pydantic import BaseModel, Field
import json
from app.config import settings
from app.services.llm.gemini_client import gemini_client
from app.services.llm.errors import ModelResponseError

class EvalScore(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0, description="Is the answer derived purely from the context?")
    relevance: float = Field(ge=0.0, le=1.0, description="Does the answer directly address the question?")
    completeness: float = Field(ge=0.0, le=1.0, description="Is the answer complete?")
    reasoning: str = Field(description="Brief explanation for the scores given.")

class LLMJudge:
    def __init__(self):
        self.client = gemini_client
        self.model_name = settings.GEMINI_CHAT_MODEL
        
        self.JUDGE_PROMPT = """
        You are an expert AI evaluator assessing an Agent's response.
        You will be provided with:
        1. A QUESTION asked by the user.
        2. The RETRIEVED CONTEXT provided to the Agent.
        3. The AGENT'S ANSWER.
        
        Your task is to score the Agent's answer on three dimensions from 0.0 to 1.0:
        
        - FAITHFULNESS: (0-1) Does the answer ONLY contain information present in the retrieved context? If it hallucinates outside facts, score lower.
        - RELEVANCE: (0-1) Does the answer directly address the user's question? If it rambles or misses the point, score lower.
        - COMPLETENESS: (0-1) Did it answer all parts of the question?
        
        You must output JSON exactly matching this schema:
        """

    def score(self, question: str, retrieved_context: str, answer: str) -> EvalScore:
        prompt = (
            f"QUESTION:\n{question}\n\n"
            f"RETRIEVED CONTEXT:\n{retrieved_context}\n\n"
            f"AGENT'S ANSWER:\n{answer}"
        )
        
        raw = self.client.complete([{"role": "user", "content": prompt}], self.JUDGE_PROMPT + json.dumps(EvalScore.model_json_schema(), indent=2))
        try:
            return EvalScore.model_validate_json(raw)
        except ValueError as exc:
            raise ModelResponseError() from exc

llm_judge = LLMJudge()
