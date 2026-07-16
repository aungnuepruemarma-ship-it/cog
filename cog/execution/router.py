"""Phase 2: ToolRouter — name → tool dispatch.

The planner can only reference registered tools; dispatching an unknown name
raises, which the executor records as evidence rather than hiding.
"""

from __future__ import annotations

from typing import Any

from cog.execution.tools import Tool


class ToolNotFound(KeyError):
    pass


class ToolRouter:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFound(f"no tool registered under {name!r}") from None

    def dispatch(self, name: str, args: dict[str, Any], context: Any = None) -> Any:
        tool = self.get(name)
        import inspect

        sig = inspect.signature(tool.run)
        params = sig.parameters
        forwards_context = ("context" in params) or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if forwards_context:
            return tool.run(**args, context=context)
        return tool.run(**args)

    def specs(self) -> list[dict[str, str]]:
        """Name + description for every tool — what the planner may use."""
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]
