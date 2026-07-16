"""Phase 2: tools.

Tools are the only way the executor touches the world. Built-ins are
deliberately safe: an AST-restricted calculator, a note-taker, and a
root-confined file reader. Sandbox/browser/terminal surfaces are future
tools behind the same protocol.
"""

from __future__ import annotations

import ast
import operator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str

    def run(self, **args: Any) -> Any: ...


class ToolSpec:
    """Concrete, constructible tool container.

    ``Tool`` above is a Protocol (structural typing only) and cannot be
    instantiated. ``ToolSpec`` is the real registration object handed to the
    ``ToolRouter``: it carries a name, description, and a callable ``run``.
    """

    def __init__(self, name: str, description: str, run: Any) -> None:
        self.name = name
        self.description = description
        self.run = run


_BIN_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Any] = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")


class CalculatorTool:
    """Arithmetic over numbers only — no names, calls, or attributes."""

    name = "calculator"
    description = 'Evaluate an arithmetic expression. Args: {"expression": "2 + 2 * 10"}'

    def run(self, *, expression: str) -> float:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree)
        # Return ints as ints so results compare cleanly against expectations.
        if isinstance(result, float) and result.is_integer():
            return int(result)
        return result


class NoteTool:
    """Record an intermediate note; returns the text so it can be the output."""

    name = "note"
    description = 'Record a note. Args: {"text": "..."}'

    def __init__(self) -> None:
        self.notes: list[str] = []

    def run(self, *, text: str) -> str:
        self.notes.append(text)
        return text


class TextTool:
    """Deterministic text transforms — a second behavioral domain alongside
    arithmetic, so cross-domain transfer is measurable."""

    name = "text"
    description = 'Transform text. Args: {"op": "length" | "reverse" | "upper", "value": "..."}'

    _OPS = {
        "length": len,
        "reverse": lambda value: value[::-1],
        "upper": str.upper,
    }

    def run(self, *, op: str, value: str):
        try:
            operation = self._OPS[op]
        except KeyError:
            raise ValueError(f"unsupported text op: {op!r}") from None
        return operation(value)


class JsonTool:
    """Deterministic JSON inspection — a third behavioral domain."""

    name = "json"
    description = (
        'Inspect a JSON document. Args: {"op": "get" | "keys" | "length",'
        ' "document": "<json string>", "path": "dotted.path (for get)"}'
    )

    def run(self, *, op: str, document: str, path: str = ""):
        import json as _json

        data = _json.loads(document)
        if op == "get":
            value = data
            for part in [p for p in path.split(".") if p]:
                if isinstance(value, list):
                    value = value[int(part)]
                else:
                    value = value[part]
            return value
        if op == "keys":
            if not isinstance(data, dict):
                raise ValueError("keys requires a JSON object")
            return sorted(data)
        if op == "length":
            return len(data)
        raise ValueError(f"unsupported json op: {op!r}")


class PythonTool:
    """OPT-IN sandbox seed — runs a Python snippet in a subprocess with a
    hard timeout. Deliberately NOT in default_tools: code execution joins
    the default surface only behind a real sandbox."""

    name = "python"
    description = 'Run a short Python snippet; stdout is the result. Args: {"code": "print(6*7)"}'

    def __init__(self, timeout_s: float = 5.0) -> None:
        self.timeout_s = timeout_s

    def run(self, *, code: str) -> str:
        import subprocess
        import sys

        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"python exited {completed.returncode}: {completed.stderr.strip()}")
        return completed.stdout.strip()


class ReadFileTool:
    """Read a UTF-8 text file confined to a root directory."""

    name = "read_file"
    description = 'Read a text file under the workspace root. Args: {"path": "relative/path"}'

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def run(self, *, path: str) -> str:
        target = (self.root / path).resolve()
        if not target.is_relative_to(self.root):
            raise PermissionError(f"path escapes tool root: {path}")
        return target.read_text(encoding="utf-8")


def default_tools(root: Path | None = None) -> list[Tool]:
    tools: list[Tool] = [CalculatorTool(), NoteTool(), TextTool(), JsonTool()]
    if root is not None:
        tools.append(ReadFileTool(root))
    return tools
