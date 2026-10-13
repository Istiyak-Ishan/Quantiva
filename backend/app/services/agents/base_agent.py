from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
from google import genai
from google.genai import types

from app.config import settings
from app.services.tools.registry import ToolRegistry

class AgentResult(BaseModel):
    answer: str
    citations: List[Dict[str, Any]] = []
    tool_calls_made: int = 0
    iterations: int = 0

class BaseAgent:
    def __init__(self, system_prompt: str, tools: ToolRegistry = None):
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.model_name = "gemini-1.5-flash"
        self.system_prompt = system_prompt
        self.tools = tools
        self.MAX_ITERATIONS = 8

    def _observe(self, messages: List[Dict[str, Any]], system_prompt: str) -> List[types.Content]:
        """
        OBSERVE: The agent looks at the conversation history so far.
        We format the messages for the Gemini SDK.
        """
        contents = []
        for msg in messages:
            if msg["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg["content"])]))
            elif msg["role"] == "assistant":
                # Check if it was a tool call or text response
                if "tool_calls" in msg:
                    parts = []
                    for tc in msg["tool_calls"]:
                        parts.append(types.Part.from_function_call(
                            name=tc["name"], 
                            args=tc["args"]
                        ))
                    contents.append(types.Content(role="model", parts=parts))
                else:
                    contents.append(types.Content(role="model", parts=[types.Part.from_text(text=msg["content"])]))
            elif msg["role"] == "tool":
                # The result of a tool execution
                contents.append(types.Content(role="user", parts=[
                    types.Part.from_function_response(
                        name=msg["name"], 
                        response={"result": msg["content"]}
                    )
                ]))
        return contents

    def _think(self, contents: List[types.Content]) -> Any:
        """
        THINK: The agent uses the LLM to decide what to do next.
        It can either reply with text (final answer) or output a function call.
        """
        # Pass the tool schemas to Gemini so it knows what it can call
        tools = None
        if self.tools:
            tools = [{"function_declarations": self.tools.get_schemas()}]

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=0.0,
            tools=tools
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            return response
        except Exception as e:
            # Fallback for local testing without valid API Key
            print(f"Gemini API Error: {e}")
            
            class MockFunctionCall:
                def __init__(self, name, args):
                    self.name = name
                    self.args = args
                    
            class MockResponse:
                def __init__(self, text, function_calls):
                    self.text = text
                    self.function_calls = function_calls
                    
            # If the LLM has already seen a tool response, give the final answer
            has_tool_response = any(m.get("role") == "tool" for m in contents if hasattr(m, "role") and getattr(m, "role") == "tool" or (isinstance(m, dict) and m.get("role") == "tool"))
            
            if len(contents) > 1 and "function_response" in str(contents[-1]):
                return MockResponse(f"Mock Final Answer (API Error) - After looking at the retrieved chunks, the revenue is $5M.", [])
            else:
                # Simulate LLM deciding to search
                import json
                return MockResponse("", [MockFunctionCall("retrieve_chunks", {"search_query": "revenue"})])

    async def _act(self, function_calls) -> List[Dict[str, Any]]:
        """
        ACT: The agent executes the tools it decided to call during the THINK phase.
        """
        results = []
        for fc in function_calls:
            name = fc.name
            args = fc.args
            
            # Execute the python function from the registry
            result_str = await self.tools.execute(name, args)
            
            # Format the result back into a message for the next iteration
            results.append({
                "role": "tool",
                "name": name,
                "content": result_str
            })
        return results

    async def run(self, query: str, context: Dict[str, Any] = None, history: List[Dict[str, Any]] = None) -> AgentResult:
        """
        The Core Agent Loop: Observe -> Think -> Act.
        """
        if history is None:
            history = []
            
        # Append the new user query to the history
        messages = list(history)
        messages.append({"role": "user", "content": query})
        
        for i in range(self.MAX_ITERATIONS):
            # 1. OBSERVE current state
            contents = self._observe(messages, self.system_prompt)
            
            # 2. THINK (Call LLM)
            response = self._think(contents)
            
            # If the LLM returned function calls, we must ACT
            if response.function_calls:
                tool_calls = [{"name": fc.name, "args": fc.args} for fc in response.function_calls]
                messages.append({"role": "assistant", "tool_calls": tool_calls})
                
                tool_results = await self._act(response.function_calls)
                messages.extend(tool_results)
            else:
                return AgentResult(answer=response.text, citations=[], tool_calls_made=i+1, iterations=i+1)
                
        return AgentResult(answer="Agent stopped after reaching maximum iterations.", citations=[], tool_calls_made=self.MAX_ITERATIONS, iterations=self.MAX_ITERATIONS)
