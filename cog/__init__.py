"""Cog — an Intelligence Runtime with a scientific learning engine.

Every task is an experiment. Every experiment produces evidence.
Every verified pattern compresses into a better reasoning architecture.
"""

from cog.experience.record import Experience
from cog.runtime.adapter import ModelAdapter, ScriptedAdapter
from cog.runtime.core import CogRuntime
from cog.runtime.model_adapters import AnthropicAdapter, CallableAdapter, OpenAIAdapter
from cog.runtime.task import Budget, Task

__version__ = "0.1.0"

__all__ = [
    "AnthropicAdapter",
    "Budget",
    "CallableAdapter",
    "CogRuntime",
    "Experience",
    "ModelAdapter",
    "OpenAIAdapter",
    "ScriptedAdapter",
    "Task",
    "__version__",
]
