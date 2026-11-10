"""Narrow prompt-override detection, not a general safety classifier.

Benign security, finance, and educational content is not blocked for mentioning
words like 'attack', 'kill', 'system prompt', or 'act as an analyst'.
"""
import re


class ContentFilter:
    OVERRIDE = re.compile(r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|system)\s+(?:instructions|rules)\b", re.I)

    async def check(self, text: str) -> dict:
        blocked = bool(self.OVERRIDE.search(text))
        return {"is_safe": not blocked, "categories": ["prompt_injection"] if blocked else []}


content_filter = ContentFilter()
