import sys
from io import StringIO
import concurrent.futures
from typing import Dict, Any

try:
    from RestrictedPython import compile_restricted
    from RestrictedPython import safe_globals
    from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getitem
    from RestrictedPython.Guards import safe_builtins, guarded_iter_unpack_sequence
except ImportError:
    pass

class PythonExecutorTool:
    def __init__(self, timeout_seconds: int = 10):
        self.timeout = timeout_seconds

    def _run_sandboxed(self, code: str) -> Dict[str, Any]:
        """
        Compiles and executes the code using RestrictedPython.
        Captures stdout and returns it.
        """
        # Capture stdout
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        
        try:
            # 1. Compile the code restrictively
            byte_code = compile_restricted(code, '<inline>', 'exec')
            
            # 2. Setup the restricted environment
            restricted_globals = safe_globals.copy()
            restricted_globals['__builtins__'] = safe_builtins.copy()
            # Explicitly block __import__ to prevent loading any dangerous modules (os, sys, subprocess)
            restricted_globals['__builtins__']['__import__'] = None 
            
            # Allow iteration and slicing (necessary for basic python operations)
            restricted_globals['_getiter_'] = default_guarded_getiter
            restricted_globals['_getitem_'] = default_guarded_getitem
            restricted_globals['_iter_unpack_sequence_'] = guarded_iter_unpack_sequence
            
            # 3. Execute
            exec(byte_code, restricted_globals)
            
            success = True
            error = None
        except Exception as e:
            success = False
            error = f"{type(e).__name__}: {str(e)}"
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            output_str = redirected_output.getvalue()
            
        return {
            "success": success,
            "output": output_str.strip(),
            "error": error
        }

    async def execute(self, code: str) -> Dict[str, Any]:
        """
        Runs the sandboxed execution inside a ThreadPoolExecutor 
        with a strict timeout to prevent infinite loops (e.g. while True).
        """
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Submit the task to the thread pool
                future = executor.submit(self._run_sandboxed, code)
                # Wait for completion, with timeout
                result = future.result(timeout=self.timeout)
                return result
        except concurrent.futures.TimeoutError:
            return {
                "success": False,
                "output": "",
                "error": f"TimeoutError: Execution exceeded {self.timeout} seconds."
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"ExecutionError: {str(e)}"
            }

python_executor = PythonExecutorTool()
