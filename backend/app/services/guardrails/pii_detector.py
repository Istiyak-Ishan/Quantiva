"""Deterministic PII protection, with explicit optional Presidio NLP mode.

Pattern mode does not claim to detect names/addresses. Presidio mode fails closed
if its packages or provisioned language model are missing; it never downloads at
application import and never silently disables redaction.
"""
import asyncio
import importlib.util
import re
from threading import Lock

from app.config import settings
from app.services.llm.errors import ProviderError


class PIIDetector:
    PATTERNS = {
        "EMAIL_ADDRESS": re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
        "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "PHONE_NUMBER": re.compile(r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?!\w)"),
        "CREDIT_CARD": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    }

    def __init__(self):
        self.analyzer = None
        self.anonymizer = None
        self._lock = Lock()

    def _presidio(self, text):
        with self._lock:
            if self.analyzer is None:
                if importlib.util.find_spec(settings.SPACY_MODEL) is None:
                    raise ProviderError("PII protection unavailable: provision the configured spaCy model")
                from presidio_analyzer import AnalyzerEngine
                from presidio_analyzer.nlp_engine import NlpEngineProvider
                from presidio_anonymizer import AnonymizerEngine
                nlp = NlpEngineProvider(nlp_configuration={"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": settings.SPACY_MODEL}]}).create_engine()
                self.analyzer = AnalyzerEngine(nlp_engine=nlp)
                self.anonymizer = AnonymizerEngine()
            results = self.analyzer.analyze(text=text, language="en", entities=[*self.PATTERNS, "PERSON"])
            redacted = self.anonymizer.anonymize(text=text, analyzer_results=results).text
        return {"has_pii": bool(results), "redacted_text": redacted, "entities": sorted({r.entity_type for r in results}), "mode": "presidio"}

    async def scan_and_redact(self, text: str) -> dict:
        if settings.PII_ENGINE == "presidio":
            return await asyncio.to_thread(self._presidio, text)
        if settings.PII_ENGINE != "patterns":
            raise ProviderError("Unknown PII protection mode")
        entities = []
        for entity, pattern in self.PATTERNS.items():
            if pattern.search(text):
                entities.append(entity)
                text = pattern.sub(f"[REDACTED_{entity}]", text)
        return {"has_pii": bool(entities), "redacted_text": text, "entities": entities, "mode": "patterns"}


pii_detector = PIIDetector()
