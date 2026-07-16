"""Phase 1: the adaptive workspace.

Exists only for the current task; snapshotted into the Experience record at
task end, then discarded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cog.runtime.task import Budget


@dataclass
class TaskWorkspace:
    task_id: str
    goal: str
    purpose: str = ""
    constraints: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    memories: list[dict[str, Any]] = field(default_factory=list)
    skills: list[dict[str, Any]] = field(default_factory=list)
    research: list[dict[str, Any]] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    plan: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    budget: Budget = field(default_factory=Budget)

    def snapshot(self) -> dict[str, Any]:
        """Plain-data copy for the Experience record."""
        return asdict(self)

    def persist_snapshot(self) -> dict[str, Any]:
        """Plain-data copy for durable storage.

        Drops the *retrieved* context (``memories``/``skills``/``research``):
        those are planning inputs pulled from the memory router for THIS task,
        not outputs of the experience. Persisting them nests prior experiences'
        full workspaces inside this one (each retrieved experience carries its
        own nested context), ballooning a single record to tens of MB and making
        every later retrieval O(blob). The context is reconstructable from the
        knowledge-graph edges (similar_goal, produced_fact, used_skill).

        ``hypotheses`` is KEPT: it is a flat list of short strings (preemptive
        corrections), carries no nested records, and tests rely on it surviving
        in the persisted experience.
        """
        data = asdict(self)
        for key in ("memories", "skills", "research"):
            data.pop(key, None)
        return data
