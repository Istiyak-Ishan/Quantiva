"""Opt-in, bounded live provider smoke test; never touches documents/databases."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.genai import types
from app.services.llm.gemini_client import gemini_client, TokenUsage
from app.services.retrieval.embedding_service import embedding_service
from app.services.llm.errors import ProviderError


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize one synthetic generation and one embedding request")
    args = parser.parse_args()
    if not args.live:
        parser.error("Use --live to authorize the two small provider requests")
    result = {}
    usage = TokenUsage()
    try:
        response = await gemini_client.generate([types.Content(role="user", parts=[types.Part.from_text(text="Reply with exactly: Quantiva provider check")])], types.GenerateContentConfig(temperature=0, max_output_tokens=64), usage=usage)
        if not response.text or "Quantiva" not in response.text:
            raise ValueError("Unexpected smoke response")
        result["generation"] = {"status": "passed", "model": gemini_client.model_name, "total_tokens": usage.total_tokens}
    except ProviderError as error:
        result["generation"] = {"status": "failed", "code": error.code}
    try:
        vector = await asyncio.to_thread(embedding_service.embed_text, "Synthetic Quantiva provider check")
        result["embedding"] = {"status": "passed", "model": embedding_service.model_name, "dimensions": len(vector)}
    except ProviderError as error:
        result["embedding"] = {"status": "failed", "code": error.code}
    print(json.dumps(result, indent=2))
    return 0 if all(v["status"] == "passed" for v in result.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
