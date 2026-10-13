import json
import re
from typing import Type, TypeVar, Any
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

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

def validate_and_retry(llm_client: Any, messages: list, schema: Type[T], max_retries: int = 3) -> T:
    """
    Calls the LLM. If the output fails validation, appends the error and retries up to max_retries.
    """
    # Create a local copy of messages to append errors to
    current_messages = list(messages)
    
    for attempt in range(max_retries):
        # Using a default system prompt for the raw text generation if not provided,
        # but in our Agent logic, the system prompt is configured on the client.
        # Assuming llm_client.complete handles (messages, system_prompt).
        # Wait, for the Agent we pass system_prompt separately. 
        # This function assumes llm_client is our GeminiClient.
        raw_text = llm_client.complete(current_messages, "Return ONLY valid JSON matching the schema.")
        
        try:
            return parse_structured_output(raw_text, schema)
        except ParseError as e:
            if attempt == max_retries - 1:
                # If we exhausted retries, raise the exception to the caller (API will return 500)
                raise Exception(f"Failed to generate structured output after {max_retries} attempts. Last error: {e}")
                
            # Otherwise, append the error and retry
            retry_prompt = (
                f"Your previous response failed validation. "
                f"Error: {str(e)}\n"
                f"Please fix the formatting and output valid JSON exactly matching the schema."
            )
            
            # The LLM's invalid response
            current_messages.append({"role": "assistant", "content": raw_text})
            # Our correction command
            current_messages.append({"role": "user", "content": retry_prompt})
            
    # Should not reach here due to the raise in the loop
    raise Exception("Unknown validation error")
