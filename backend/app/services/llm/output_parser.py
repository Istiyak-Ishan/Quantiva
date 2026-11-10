import json
import re
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def answer_string_prefix(raw_text: str) -> str:
    """Decode the complete prefix of a streamed JSON answer string.

    Do not emit JSON framing, incomplete escapes, or half a surrogate pair.
    The final complete object is still validated with Pydantic before success.
    """
    match = re.search(r'"answer"\s*:\s*"', raw_text)
    if match is None:
        return ""
    start, end = match.end(), match.end()
    while end < len(raw_text):
        character = raw_text[end]
        if character == '"':
            break
        if character == "\\":
            if end + 1 >= len(raw_text):
                break
            if raw_text[end + 1] == "u":
                if end + 6 > len(raw_text):
                    break
                end += 6
                continue
            end += 2
            continue
        end += 1
    try:
        prefix = json.loads('"' + raw_text[start:end] + '"')
        if prefix and 0xD800 <= ord(prefix[-1]) <= 0xDBFF:
            prefix = prefix[:-1]
        return prefix
    except ValueError:
        return ""

class ParseError(Exception):
    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text

def parse_structured_output(raw_text: str, schema: Type[T]) -> T:
    """
    Parses a raw LLM string into a validated Pydantic model.
    Handles plain JSON, markdown code blocks (```json ... ```), and partial text.
    """
    text = raw_text.strip()
    
    # 1. Extract JSON from markdown blocks if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
        
    # 2. Attempt to parse JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ParseError(f"Failed to decode JSON: {str(e)}", raw_text)
        
    # 3. Validate against Pydantic schema
    try:
        validated_data = schema(**data)
        return validated_data
    except ValidationError as e:
        raise ParseError(f"Schema validation failed: {str(e)}", raw_text)

