import asyncio
from typing import List, Dict, Any

from app.services.guardrails.pii_detector import pii_detector
from app.services.guardrails.hallucination_checker import hallucination_checker
from app.services.guardrails.content_filter import content_filter

class GuardrailBlockError(Exception):
    def __init__(self, message: str, categories: List[str]):
        self.message = message
        self.categories = categories
        super().__init__(self.message)

class GuardrailsService:
    async def check_input(self, user_query: str):
        """
        Runs input-side guardrails before the Agent even sees the prompt.
        """
        content_res = await content_filter.check(user_query)
        
        if not content_res["is_safe"]:
            raise GuardrailBlockError(
                message="Your prompt was blocked by our safety filters.",
                categories=content_res["categories"]
            )

    async def check_output(self, answer: str, source_chunks: List[str]) -> Dict[str, Any]:
        """
        Runs output-side guardrails in parallel.
        1. PII Redaction
        2. Hallucination / Groundedness check
        3. Output toxicity/safety check
        """
        
        # We run these heavy tasks concurrently
        results = await asyncio.gather(
            pii_detector.scan_and_redact(answer),
            hallucination_checker.check(answer, source_chunks),
            content_filter.check(answer)
        )
        
        pii_res, hal_res, content_res = results
        
        # If output itself is toxic/unsafe, we block it entirely
        if not content_res["is_safe"]:
            raise GuardrailBlockError(
                message="The agent generated unsafe content.",
                categories=content_res["categories"]
            )
            
        flags = []
        if pii_res["has_pii"]:
            flags.append(f"PII Redacted: {', '.join(pii_res['entities'])}")
            
        if hal_res["is_hallucinated"]:
            flags.append(f"Low Groundedness Score ({hal_res['score']:.2f}). Agent may be hallucinating.")
            
        return {
            "answer": pii_res["redacted_text"],
            "flags": flags,
            "blocked": False,
            "hallucination_score": hal_res["score"]
        }

guardrails_service = GuardrailsService()
