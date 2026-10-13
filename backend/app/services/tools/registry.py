import inspect
from typing import List, Dict, Any, Callable

class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name: str, func: Callable, schema: Dict[str, Any]):
        """Registers a Python function as an AI-callable tool."""
        self.tools[name] = {
            "func": func,
            "schema": schema
        }

    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        Returns all registered tool schemas in a format compatible with 
        OpenAI and Gemini function calling specs.
        """
        schemas = []
        for name, tool in self.tools.items():
            schemas.append({
                "name": name,
                "description": tool["schema"].get("description", ""),
                "parameters": tool["schema"].get("parameters", {"type": "object", "properties": {}})
            })
        return schemas

    async def execute(self, name: str, args: Dict[str, Any]) -> str:
        """Executes the Python function with the arguments provided by the LLM."""
        if name not in self.tools:
            return f"Error: Tool {name} not found."
            
        func = self.tools[name]["func"]
        
        try:
            # If it's an async function, await it
            if inspect.iscoroutinefunction(func):
                result = await func(**args)
            else:
                result = func(**args)
            return str(result)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"
