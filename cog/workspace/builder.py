"""Phase 1: WorkspaceBuilder.

Assembles the per-task workspace, pulling relevant memories and skills from
the Memory Router so past verified evidence shapes the workspace before
planning begins.
"""

from __future__ import annotations

from cog.memory.router import MemoryRouter
from cog.runtime.task import Task
from cog.workspace.workspace import TaskWorkspace


class WorkspaceBuilder:
    def __init__(self, memory: MemoryRouter, memory_limit: int = 5) -> None:
        self.memory = memory
        self.memory_limit = memory_limit

    def build(self, task: Task) -> TaskWorkspace:
        memories = [
            {"id": r.id, "kind": r.kind, "content": r.content, "confidence": r.confidence}
            for r in self.memory.retrieve(
                task.goal, kinds=("fact", "experience", "concept"), limit=self.memory_limit
            )
        ]
        skills = [
            {"id": r.id, "kind": r.kind, "content": r.content, "confidence": r.confidence}
            for r in self.memory.retrieve(task.goal, kinds=("skill",), limit=self.memory_limit)
        ]
        # Causal credit assignment: preemptively inject corrections learned from
        # past failure→success retries so the first attempt avoids known mistakes.
        # Imported lazily — cog.learning pulls in the execution layer, and this
        # module is imported during that layer's own initialization.
        from cog.learning.corrections import corrections_for_goal

        hypotheses = corrections_for_goal(self.memory, task.goal)
        return TaskWorkspace(
            task_id=task.id,
            goal=task.goal,
            purpose=task.purpose,
            constraints=list(task.constraints),
            context=dict(task.context),
            memories=memories,
            skills=skills,
            hypotheses=hypotheses,
            budget=task.budget,
        )
