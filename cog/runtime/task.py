"""Task and Budget: what enters the runtime.

Purpose, constraints, and success criteria are first-class from the start
(Phase 14) so no downstream engine has to reconstruct intent from traces.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cog._util import new_id


@dataclass
class Budget:
    """Hard ceilings for one task's execution (Phase 20 raw material)."""

    max_actions: int = 16
    max_seconds: float = 60.0
    # Extra planning attempts after a failed verification. Each retry replans
    # with the failure hypothesis appended to the workspace.
    max_retries: int = 0


@dataclass
class Task:
    goal: str
    purpose: str = ""
    constraints: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    success_criteria: list[str] = field(default_factory=list)
    # Optional expectation for the verification layer: a literal value the
    # final output must equal, or a predicate ``f(output) -> bool``.
    expected_output: Any | Callable[[Any], bool] | None = None
    budget: Budget = field(default_factory=Budget)
    id: str = field(default_factory=lambda: new_id("task"))
    # Optional metadata used by the evidence layer (replay + analytics).
    domain: str = "unspecified"
    difficulty: str = "unspecified"
    seed: int | None = None
    version: str = "v1"
