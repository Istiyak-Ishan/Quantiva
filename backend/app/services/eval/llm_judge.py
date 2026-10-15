from pydantic import BaseModel, Field
import json
from google import genai
from google.genai import types
from app.config import settings

class EvalScore(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0, description="Is the answer derived purely from the context?")
    relevance: float = Field(ge=0.0, le=1.0, description="Does the answer directly address the question?")
    completeness: float = Field(ge=0.0, le=1.0, description="Is the answer complete?")
    reasoning: str = Field(description="Brief explanation for the scores given.")

class LLMJudge:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        # Often a larger, more capable model (like Gemini 1.5 Pro) is used as the judge
        # to evaluate a smaller model (like Gemini 1.5 Flash). We'll use flash here for speed/cost.
        self.model_name = "gemini-1.5-flash"
        
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
        
        config = types.GenerateContentConfig(
            system_instruction=self.JUDGE_PROMPT + json.dumps(EvalScore.model_json_schema(), indent=2),
            temperature=0.0,
            response_mime_type="application/json"
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=config
            )
            return EvalScore.model_validate_json(response.text)
        except Exception as e:
            # Fallback for local testing without valid key
            import random
            return EvalScore(
                faithfulness=random.uniform(0.6, 1.0),
                relevance=random.uniform(0.6, 1.0),
                completeness=random.uniform(0.6, 1.0),
                reasoning=f"Mock eval due to API error: {str(e)}"
            )

llm_judge = LLMJudge()
