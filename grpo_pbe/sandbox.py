import signal
import re
import datetime
import _strptime  # noqa: F401 — pre-import so datetime.strptime works without __import__
from dataclasses import dataclass
from typing import Any


@dataclass
class ExecutionResult:
    success: bool
    output: Any = None
    error: str | None = None


def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")


def execute_code(code: str, input_val: Any, timeout_seconds: int = 2) -> ExecutionResult:
    """Execute a Python expression in a restricted sandbox.

    The expression can use `x` to refer to the input value.
    Only builtins, `re`, and `datetime` are available.
    """
    _real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    _ALLOWED_IMPORTS = frozenset({"re", "datetime", "_strptime", "time", "math"})

    def _safe_import(name, *args, **kwargs):
        if name in _ALLOWED_IMPORTS:
            return _real_import(name, *args, **kwargs)
        raise ImportError(f"Import of '{name}' is not allowed")

    safe_builtins = {
        k: v for k, v in __builtins__.items()
        if k not in ("exec", "eval", "compile", "open",
                      "breakpoint", "exit", "quit", "input", "print")
    } if isinstance(__builtins__, dict) else {
        k: getattr(__builtins__, k) for k in dir(__builtins__)
        if k not in ("exec", "eval", "compile", "open",
                      "breakpoint", "exit", "quit", "input", "print")
        and not k.startswith("_")
    }
    safe_builtins["__import__"] = _safe_import

    restricted_globals = {
        "__builtins__": safe_builtins,
        "re": re,
        "datetime": datetime,
    }

    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)

    try:
        fn = eval(f"lambda x: {code}", restricted_globals)
        result = fn(input_val)
        return ExecutionResult(success=True, output=result)
    except TimeoutError:
        return ExecutionResult(success=False, error="timeout")
    except Exception as e:
        return ExecutionResult(success=False, error=f"{type(e).__name__}: {e}")
    finally:
        if has_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
