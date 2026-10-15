import asyncio
from typing import List, Dict, Any, Tuple
try:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
except ImportError:
    pass

class PIIDetector:
    def __init__(self):
        try:
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        except:
            self.analyzer = None
            self.anonymizer = None

    async def scan_and_redact(self, text: str) -> Dict[str, Any]:
        """
        Scans the text for PII using Microsoft Presidio and redacts it.
        Returns the modified text and whether PII was found.
        """
        if not self.analyzer:
            return {"has_pii": False, "redacted_text": text, "entities": []}

        # Offload CPU bound NLP task to threadpool
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, 
            lambda: self.analyzer.analyze(
                text=text,
                entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "US_SSN", "PERSON"],
                language='en'
            )
        )
        
        has_pii = len(results) > 0
        entities = [res.entity_type for res in results]
        
        if has_pii:
            # We configure the anonymizer to replace with [REDACTED_<ENTITY>]
            operators = {
                entity: OperatorConfig("replace", {"new_value": f"[REDACTED_{entity}]"})
                for entity in set(entities)
            }
            
            anonymized_result = await loop.run_in_executor(
                None,
                lambda: self.anonymizer.anonymize(
                    text=text,
                    analyzer_results=results,
                    operators=operators
                )
            )
            redacted_text = anonymized_result.text
        else:
            redacted_text = text
            
        return {
            "has_pii": has_pii,
            "entities": list(set(entities)),
            "redacted_text": redacted_text
        }

pii_detector = PIIDetector()
