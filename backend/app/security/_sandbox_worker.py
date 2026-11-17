"""Isolated child process for :mod:`app.security.sandbox` (no app imports)."""
from __future__ import annotations

import builtins
import contextlib
import io
import json
import sys
from pathlib import Path

SAFE_MODULES = {
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


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name).split(".", 1)[0]
    if level or root not in SAFE_MODULES:
        raise ImportError(f"Import of '{root}' is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _safe_open_factory(scratch_dir: Path):
    scratch_dir = scratch_dir.resolve()

    def safe_open(file, mode="r", *args, **kwargs):
        candidate = Path(file)
        if not candidate.is_absolute():
            candidate = scratch_dir / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(scratch_dir)
        except ValueError as exc:
            raise PermissionError(
                "open() may only access the sandbox scratch directory"
            ) from exc
        if any(flag in mode for flag in ("+", "a", "w", "x")):
            candidate.parent.mkdir(parents=True, exist_ok=True)
        return builtins.open(candidate, mode, *args, **kwargs)

    return safe_open


def _safe_builtins(scratch_dir: Path):
    allowed_names = (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "pow",
        "print",
        "range",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "Exception",
        "ValueError",
        "TypeError",
        "ArithmeticError",
    )
    values = {name: getattr(builtins, name) for name in allowed_names}
    values["__import__"] = _safe_import
    values["open"] = _safe_open_factory(scratch_dir)
    return values


def main() -> None:
    try:
        payload = json.loads(sys.stdin.buffer.read(100_000).decode("utf-8"))
        code = str(payload["code"])
        scratch_dir = Path(payload["scratch_dir"]).resolve()
        max_output = int(payload.get("max_output_bytes", 65_536))
        namespace = {"__builtins__": _safe_builtins(scratch_dir)}
        collector = io.StringIO()
        with contextlib.redirect_stdout(collector):
            exec(compile(code, "<quantiva-sandbox>", "exec"), namespace, namespace)
        output = collector.getvalue()
        if len(output.encode("utf-8")) > max_output:
            raise ValueError("Output limit exceeded")
        result = {"success": True, "output": output.strip(), "error": None}
    except BaseException as exc:
        result = {
            "success": False,
            "output": "",
            "error": f"{type(exc).__name__}: {str(exc)[:500]}",
        }
    sys.stdout.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
