"""Phase 3: the ExecutionContext — explicit state flowing through execution.

This is the missing abstraction between planner and tools. Previously the
executor dispatched ``router.dispatch(tool, args)`` with no shared state, which
forced benchmark authors to fake dependencies with module-level flags or
closures. That is a benchmark artifact, not a runtime capability.

With an ExecutionContext, tools answer the honest question:

    "Given this environment/state, can I execute?"

not "Did the planner choose the correct strategy?". The context carries the
workspace, task id, a mutable metadata bag, and the executed-step history. This
same object later supports causal graphs, replay, debugging, and safety checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cog.workspace.workspace import TaskWorkspace


@dataclass
class ExecutionContext:
    task_id: str
    workspace: TaskWorkspace | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    executed_steps: list[str] = field(default_factory=list)

    def record_step(self, name: str) -> None:
        self.executed_steps.append(name)

    def set(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.metadata
