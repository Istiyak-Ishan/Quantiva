"""AST-validated Python execution in a killable, clean-environment subprocess.

This is a strong application-level boundary for local calculations, not a
replacement for an operating-system sandbox.  Code is validated before spawn,
executed with restricted builtins/imports, receives no parent environment, and
is terminated after a strict wall-clock timeout.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SAFE_MODULES = frozenset(
    {
        "math",
        "statistics",
        "decimal",
        "fractions",
        "datetime",
        "json",
        "re",
        "collections",
        "itertools",
        "pandas",
        "numpy",
    }
)

BLOCKED_MODULES = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "socket",
        "http",
        "urllib",
        "requests",
        "httpx",
        "pathlib",
        "ctypes",
        "winreg",
        "multiprocessing",
        "threading",
        "importlib",
        "inspect",
        "builtins",
    }
)

BLOCKED_CALLS = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
    }
)

BLOCKED_ATTRIBUTES = frozenset(
    {
        "system",
        "popen",
        "spawn",
        "fork",
        "kill",
        "remove",
        "unlink",
        "rmdir",
        "rmtree",
        "chmod",
        "chown",
        "connect",
        "request",
        "urlopen",
        "read_csv",
        "read_excel",
        "read_pickle",
        "read_parquet",
        "read_feather",
        "read_fwf",
        "read_html",
        "read_json",
        "read_orc",
        "read_sas",
        "read_spss",
        "read_sql",
        "read_stata",
        "read_table",
        "read_xml",
        "read_clipboard",
        "to_csv",
        "to_excel",
        "to_pickle",
        "to_parquet",
        "to_feather",
        "tofile",
        "load",
        "loadtxt",
        "save",
        "savetxt",
        "genfromtxt",
        "fromfile",
        "memmap",
        "ctypeslib",
        "load_library",
        "excelfile",
    }
)


class SandboxValidationError(ValueError):
    """Raised when code violates the sandbox AST policy."""


@dataclass(frozen=True, slots=True)
class SandboxResult:
    success: bool
    output: str = ""
    error: str | None = None
    timed_out: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "timed_out": self.timed_out,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


class _SandboxASTValidator(ast.NodeVisitor):
    def __init__(self, scratch_dir: Path) -> None:
        self.scratch_dir = scratch_dir.resolve()

    def fail(self, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", "?")
        raise SandboxValidationError(f"{message} (line {line})")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root in BLOCKED_MODULES or root not in SAFE_MODULES:
                self.fail(node, f"Import of '{root}' is not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".", 1)[0]
        if node.level or root in BLOCKED_MODULES or root not in SAFE_MODULES:
            self.fail(node, f"Import from '{root or 'relative module'}' is not allowed")
        if any(alias.name == "*" for alias in node.names):
            self.fail(node, "Wildcard imports are not allowed")
        for alias in node.names:
            if alias.name.lower() in BLOCKED_ATTRIBUTES:
                self.fail(node, f"Import of '{alias.name}' is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__") or node.id in BLOCKED_CALLS:
            self.fail(node, f"Name '{node.id}' is not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attribute = node.attr.lower()
        if attribute.startswith("_") or attribute in BLOCKED_ATTRIBUTES:
            self.fail(node, f"Attribute '{node.attr}' is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_CALLS:
                self.fail(node, f"Call to '{node.func.id}' is not allowed")
            if node.func.id == "open":
                self._validate_open(node)
        self.generic_visit(node)

    def _validate_open(self, node: ast.Call) -> None:
        if not node.args:
            self.fail(node, "open() requires a scratch-directory path")
        path_node = node.args[0]
        if not isinstance(path_node, ast.Constant) or not isinstance(path_node.value, str):
            # Dynamic paths are checked again by the worker's safe_open.
            return
        candidate = Path(path_node.value)
        if not candidate.is_absolute():
            candidate = self.scratch_dir / candidate
        try:
            candidate.resolve().relative_to(self.scratch_dir)
        except ValueError:
            self.fail(node, "open() may only access the sandbox scratch directory")


def validate_python_code(code: str, scratch_dir: Path) -> None:
    """Parse and validate code without executing it."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxValidationError(f"Invalid Python syntax: {exc.msg}") from exc
    _SandboxASTValidator(scratch_dir).visit(tree)


class LocalPythonSandbox:
    """Execute bounded calculations under Quantiva's local scratch directory."""

    def __init__(
        self,
        *,
        data_dir: Path | str | None = None,
        timeout_seconds: float = 15.0,
        max_code_bytes: int = 32_000,
        max_output_bytes: int = 65_536,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if data_dir is None:
            from app.config import settings

            data_dir = settings.QUANTIVA_DATA_DIR
        self.scratch_dir = Path(data_dir).resolve() / "sandbox_scratch"
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = float(timeout_seconds)
        self.max_code_bytes = int(max_code_bytes)
        self.max_output_bytes = int(max_output_bytes)
        self._worker = Path(__file__).with_name("_sandbox_worker.py")

    def validate(self, code: str) -> None:
        if not isinstance(code, str) or not code.strip():
            raise SandboxValidationError("Code must be a non-empty string")
        if len(code.encode("utf-8")) > self.max_code_bytes:
            raise SandboxValidationError("Code exceeds the sandbox size limit")
        validate_python_code(code, self.scratch_dir)

    async def execute(self, code: str) -> SandboxResult:
        try:
            self.validate(code)
        except SandboxValidationError as exc:
            return SandboxResult(False, error=f"SandboxValidationError: {exc}")

        payload = json.dumps(
            {
                "code": code,
                "scratch_dir": str(self.scratch_dir),
                "max_output_bytes": self.max_output_bytes,
            }
        ).encode("utf-8")
        # Do not inherit API keys, credentials, proxy variables, PATH, HOME, or
        # any other parent process environment.  Python is invoked by absolute
        # path and receives only deterministic interpreter controls.
        clean_env = {
            "PYTHONIOENCODING": "utf-8",
            "PYTHONHASHSEED": "0",
        }
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(asyncio.subprocess, "CREATE_NO_WINDOW", 0)
            if not creationflags:
                creationflags = 0x08000000
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                str(self._worker),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.scratch_dir),
                env=clean_env,
                creationflags=creationflags,
            )
        except OSError as exc:
            return SandboxResult(False, error=f"Sandbox process failed: {exc}")

        try:
            stdout, _stderr = await asyncio.wait_for(
                process.communicate(payload),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return SandboxResult(
                False,
                error=(
                    "TimeoutError: execution exceeded "
                    f"{self.timeout_seconds:g} seconds"
                ),
                timed_out=True,
            )
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

        if len(stdout) > self.max_output_bytes * 2:
            return SandboxResult(False, error="Sandbox output exceeded the size limit")
        try:
            result = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return SandboxResult(False, error="Sandbox returned an invalid result")
        return SandboxResult(
            success=bool(result.get("success")),
            output=str(result.get("output", "")),
            error=result.get("error"),
            timed_out=False,
        )

    async def run(self, code: str) -> SandboxResult:
        """Alias for :meth:`execute`."""
        return await self.execute(code)


__all__ = [
    "BLOCKED_MODULES",
    "LocalPythonSandbox",
    "SAFE_MODULES",
    "SandboxResult",
    "SandboxValidationError",
    "validate_python_code",
]
