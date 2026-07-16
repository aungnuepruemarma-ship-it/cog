"""Phase 0: model-independent runtime — sessions, hooks, adapters, the loop."""

from typing import Any

from cog.runtime.adapter import ModelAdapter, ScriptedAdapter
from cog.runtime.hooks import HookBus
from cog.runtime.model_adapters import AnthropicAdapter, CallableAdapter, OpenAIAdapter
from cog.runtime.session import Session
from cog.runtime.task import Budget, Task

__all__ = [
    "AnthropicAdapter",
    "Budget",
    "CallableAdapter",
    "CogRuntime",
    "HookBus",
    "ModelAdapter",
    "OpenAIAdapter",
    "ScriptedAdapter",
    "Session",
    "Task",
]


def __getattr__(name: str) -> Any:
    # CogRuntime is loaded lazily: core.py imports the execution/memory layers,
    # which import runtime.adapter — an eager import here would be circular.
    if name == "CogRuntime":
        from cog.runtime.core import CogRuntime

        return CogRuntime
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
