"""Killable RestrictedPython child process with isolated output and no secrets.

RestrictedPython is defense in depth, not an OS/container security boundary.
Deploy the worker in a resource-limited container without network privileges.
"""
import asyncio
import json
import os
from pathlib import Path
import sys


class PythonExecutorTool:
    def __init__(self, timeout_seconds: float = 10):
        self.timeout = timeout_seconds
        self._slots = asyncio.Semaphore(4)

    async def execute(self, code: str) -> dict:
        if len(code.encode()) > 16000:
            return {"success": False, "output": "", "error": "Code exceeds limit"}
        # Never inherit GOOGLE_API_KEY, database credentials, or process environment.
        env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP") if key in os.environ}
        async with self._slots:
            process = await asyncio.create_subprocess_exec(sys.executable, "-I", str(Path(__file__).with_name("python_sandbox.py")), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env)
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(json.dumps({"code": code}).encode()), timeout=self.timeout)
                if process.returncode != 0:
                    return {"success": False, "output": "", "error": "Execution process failed or exceeded resource limits"}
                return json.loads(stdout)
            except asyncio.TimeoutError:
                return {"success": False, "output": "", "error": f"TimeoutError: Execution exceeded {self.timeout} seconds"}
            except (ValueError, OSError):
                return {"success": False, "output": "", "error": "Invalid execution result"}
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()


python_executor = PythonExecutorTool()
