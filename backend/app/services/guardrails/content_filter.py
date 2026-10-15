from typing import Dict, Any

class ContentFilter:
    def __init__(self):
        # Basic heuristic blocklist for prompt injection and toxic content
        self.injection_keywords = [
            "ignore previous instructions",
            "you are now",
            "system prompt",
            "forget all rules",
            "act as a",
            "bypassing"
        ]
        self.unsafe_keywords = [
            "hack", "attack", "exploit", "kill", "murder", "bomb"
        ]

    async def check(self, text: str) -> Dict[str, Any]:
        """
        Scans text for prompt injection attempts and unsafe content.
        """
        text_lower = text.lower()
        categories = []
        is_safe = True
        
        for kw in self.injection_keywords:
            if kw in text_lower:
                is_safe = False
                categories.append("prompt_injection")
                break
                
        for kw in self.unsafe_keywords:
            if kw in text_lower:
                is_safe = False
                categories.append("unsafe_content")
                break
                
        return {
            "is_safe": is_safe,
            "categories": categories,
            "confidence": 0.95 if not is_safe else 1.0
        }

content_filter = ContentFilter()
