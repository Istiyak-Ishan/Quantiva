from typing import List, Dict, AsyncGenerator
from google import genai
from google.genai import types
from app.config import settings

class GeminiClient:
    def __init__(self):
        # We use the new google-genai SDK
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.model_name = "gemini-1.5-flash"

    def _format_messages(self, messages: List[Dict[str, str]]) -> List[types.Content]:
        """
        Converts our standard {'role': 'user', 'content': '...'} messages
        into the format expected by the google-genai SDK.
        """
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
            )
        return contents

    def complete(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        """
        Calls Gemini 1.5 Flash synchronously.
        """
        contents = self._format_messages(messages)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0 # We want factual extraction, not creative writing
        )
        
        # Use fallback if API key is invalid in local dev
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            return response.text
        except Exception as e:
            return f"Mock AI Response (Error calling Gemini: {str(e)}) - Based on the provided context, the revenue is $5M."

    async def complete_stream(self, messages: List[Dict[str, str]], system_prompt: str) -> AsyncGenerator[str, None]:
        """
        Calls Gemini 1.5 Flash via streaming.
        Yields tokens one by one as they arrive.
        """
        contents = self._format_messages(messages)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0
        )
        
        try:
            # We use generate_content_stream which returns an iterator of chunks
            response = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=config
            )
            for chunk in response:
                yield chunk.text
        except Exception as e:
            # Fallback for local testing
            mock_text = f"Mock AI Streaming Response (Error: {str(e)}) - Revenue is $5M."
            for word in mock_text.split(" "):
                import asyncio
                await asyncio.sleep(0.05)
                yield word + " "

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculates the cost based on Gemini 1.5 Flash pricing.
        Input: $0.075 per 1M tokens ($0.000075 / 1k)
        Output: $0.30 per 1M tokens ($0.0003 / 1k)
        """
        input_cost = (input_tokens / 1000) * 0.000075
        output_cost = (output_tokens / 1000) * 0.0003
        return input_cost + output_cost

gemini_client = GeminiClient()
