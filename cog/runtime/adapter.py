"""Phase 0: model independence.

The runtime talks to language models exclusively through ``ModelAdapter``.
Anything that maps a prompt string to a completion string can drive Cog —
a hosted LLM, a local model, or the deterministic ``ScriptedAdapter`` used
by tests, the demo, and the benchmark suite.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelAdapter(Protocol):
    name: str

    def complete(self, prompt: str) -> str:
        """Return the model's completion for ``prompt``."""
        ...


class ScriptedAdapter:
    """Deterministic adapter for tests and benchmarks.

    ``script`` maps a trigger substring to a canned completion; the first
    trigger found in the prompt wins (longest triggers checked first, so the
    most specific script entry takes precedence). Falls back to ``default``.
    """

    name = "scripted"

    def __init__(self, script: dict[str, str] | None = None, default: str = "") -> None:
        self.script = dict(script or {})
        self.default = default
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        for trigger in sorted(self.script, key=len, reverse=True):
            if trigger in prompt:
                return self.script[trigger]
        return self.default
