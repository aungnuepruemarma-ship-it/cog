"""Phase 2: the planner.

Turns a workspace into an executable plan via the model adapter. The wire
format is deliberately trivial — any model that can emit lines can drive Cog:

    step: <tool_name> <json-args> -- <description>

Lines that don't parse are kept as evidence in ``Plan.rejected`` instead of
being silently dropped.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from cog.execution.router import ToolRouter
from cog.runtime.adapter import ModelAdapter
from cog.workspace.workspace import TaskWorkspace

_STEP_RE = re.compile(
    r"^\s*step:\s*(?P<tool>[\w.-]+)\s*(?P<args>\{.*\})?\s*"
    r"(?:deps:\s*(?P<deps>[\w,.-]+)\s*)?"
    r"(?:--\s*(?P<desc>.*))?$"
)


@dataclass
class PlanStep:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    deps: list[str] = field(default_factory=list)  # step-ids this step depends on (ordering)
    id: str | None = None  # logical plan step id (e.g. "s0"), set by the planner

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "args": self.args,
                "description": self.description, "deps": self.deps, "id": self.id}


@dataclass
class Plan:
    steps: list[PlanStep]
    prompt: str
    raw: str
    rejected: list[str] = field(default_factory=list)


class Planner:
    def __init__(self, adapter: ModelAdapter, router: ToolRouter) -> None:
        self.adapter = adapter
        self.router = router

    def build_prompt(self, workspace: TaskWorkspace) -> str:
        tool_lines = "\n".join(
            f"- {spec['name']}: {spec['description']}" for spec in self.router.specs()
        )
        memory_lines = (
            "\n".join(
                f"- [{m['kind']}] {json.dumps(m['content'])[:200]}" for m in workspace.memories
            )
            or "- none"
        )
        skill_lines = (
            "\n".join(f"- {json.dumps(s['content'])[:200]}" for s in workspace.skills) or "- none"
        )
        constraint_lines = "\n".join(f"- {c}" for c in workspace.constraints) or "- none"
        hypothesis_lines = "\n".join(f"- {h}" for h in workspace.hypotheses) or "- none"
        return (
            "You are the planner inside the Cog intelligence runtime.\n"
            f"Goal: {workspace.goal}\n"
            f"Purpose: {workspace.purpose or 'unspecified'}\n"
            f"Constraints:\n{constraint_lines}\n"
            f"Available tools:\n{tool_lines}\n"
            f"Relevant memories:\n{memory_lines}\n"
            f"Relevant skills:\n{skill_lines}\n"
            f"Hypotheses:\n{hypothesis_lines}\n"
            f"Budget: at most {workspace.budget.max_actions} actions.\n"
            "Respond with one line per step, in execution order, formatted exactly as:\n"
            'step: <tool_name> {"arg": "value"} -- <short description>\n'
        )

    def parse(self, raw: str) -> tuple[list[PlanStep], list[str]]:
        steps: list[PlanStep] = []
        rejected: list[str] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            match = _STEP_RE.match(line)
            if not match:
                if line.strip().lower().startswith("step:"):
                    rejected.append(line.strip())
                continue
            args_text = match.group("args")
            try:
                args = json.loads(args_text) if args_text else {}
                if not isinstance(args, dict):
                    raise ValueError("args must be a JSON object")
            except ValueError:
                rejected.append(line.strip())
                continue
            steps.append(
                PlanStep(
                    tool=match.group("tool"),
                    args=args,
                    description=match.group("desc") or "",
                    deps=match.group("deps").split(",") if match.group("deps") else [],
                    id=f"s{len(steps)}",
                )
            )
        return steps, rejected

    def plan(self, workspace: TaskWorkspace) -> Plan:
        prompt = self.build_prompt(workspace)
        raw = self.adapter.complete(prompt)
        steps, rejected = self.parse(raw)
        workspace.plan = [s.to_dict() for s in steps]
        return Plan(steps=steps, prompt=prompt, raw=raw, rejected=rejected)
