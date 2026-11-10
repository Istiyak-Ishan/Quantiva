"""Isolated child entry point. Do not import application config or credentials."""
import json
import sys

from RestrictedPython import compile_restricted, safe_globals
from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getitem
from RestrictedPython.Guards import guarded_iter_unpack_sequence, full_write_guard
from RestrictedPython.PrintCollector import PrintCollector


class BoundedPrintCollector(PrintCollector):
    def __init__(self, _getattr_=None):
        super().__init__(_getattr_)
        self.size = 0

    def write(self, text):
        self.size += len(text)
        if self.size > 65536:
            raise ValueError("Output limit exceeded")
        super().write(text)


def main():
    try:
        # Unix resource limits complement the parent's wall-clock kill.
        try:
            import resource
        except ImportError:
            resource = None
        if resource is not None:
            resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        code = json.loads(sys.stdin.read(20000))["code"]
        bytecode = compile_restricted(code, "<calculation>", "exec")
        namespace = safe_globals.copy()
        namespace["__builtins__"] = {**namespace["__builtins__"], "sum": sum, "min": min, "max": max, "round": round, "sorted": sorted, "enumerate": enumerate, "all": all, "any": any}
        namespace.update(_getiter_=default_guarded_getiter, _getitem_=default_guarded_getitem, _iter_unpack_sequence_=guarded_iter_unpack_sequence, _write_=full_write_guard, _print_=BoundedPrintCollector)
        exec(bytecode, namespace)
        output = namespace["_print"]() if "_print" in namespace else ""
        result = {"success": True, "output": output.strip(), "error": None}
    except Exception as error:
        result = {"success": False, "output": "", "error": f"{type(error).__name__}: {str(error)[:300]}"}
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
