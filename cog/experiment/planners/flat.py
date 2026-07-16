"""Phase A baseline planner for Exp3 (frozen, auditable control group).

FlatPlanner is a DISTNCT class from the runtime's generic ``Planner``
(cog.execution.planner.Planner). It is deliberately NOT a subclass: a subclass
would silently inherit any future dependency reasoning or decomposition added to
the generic Planner, invalidating Exp3's control group. Here we reuse ONLY the
shared line-parsing regex (``_STEP_RE``) — a pure lexical utility with no
planning semantics — so the parser can improve without contaminating the
baseline's *planning* contract.

Contract (guaranteed, frozen):
  - produces LINEAR plans only (ordered list of steps, no structure)
  - NO hierarchy / decomposition
  - NO dependency graph / lookahead
  - NO recovery policy
  - deterministic given identical adapter output

Any code path that would add the above belongs in HTNPlanner, never here.
"""

from __future__ import annotations

import hashlib
import json

# Reuse the lexical parser ONLY. Importing the class is forbidden by design.
from cog.execution.planner import Plan, PlanStep, _STEP_RE
from cog.workspace.workspace import TaskWorkspace


class FlatPlanner:
    """Baseline planner: simple linear planning, no dependency reasoning.

    Guarantees:
      - produces linear plans only
      - no dependency reasoning
      - no decomposition
      - deterministic output
    """

    # Capability flags — explicit, machine-checkable proof of "no dependency".
    CAPABILITIES = {
        "hierarchy": False,
        "decomposition": False,
        "dependency_graph": False,
        "lookahead": False,
        "recovery_policy": False,
        "linear_only": True,
    }

    def __init__(self, adapter, router) -> None:
        # adapter/model and router are accepted for interface parity with the
        # runtime Planner, but FlatPlanner uses NEITHER for reasoning: it only
        # calls adapter.complete() to get raw text, then lexes it flat.
        self.adapter = adapter
        self.router = router

    def build_prompt(self, workspace: TaskWorkspace) -> str:
        """Flat prompt: tools + goal + budget, no decomposition scaffolding.

        Intentionally omits hierarchy/dependency cues so the model is not
        steered toward structured output.
        """
        tool_lines = "\n".join(
            f"- {spec['name']}: {spec['description']}"
            for spec in self.router.specs()
        )
        return (
            "You are a flat planner inside the Cog runtime.\n"
            f"Goal: {workspace.goal}\n"
            f"Purpose: {workspace.purpose or 'unspecified'}\n"
            f"Available tools:\n{tool_lines}\n"
            f"Budget: at most {workspace.budget.max_actions} actions.\n"
            "Respond with one line per step, in execution order, formatted exactly as:\n"
            "'step: <tool_name> {\"arg\": \"value\"} -- <short description>'\n"
        )

    @staticmethod
    def _linear_parse(raw: str) -> tuple[list[PlanStep], list[str]]:
        """Lex raw model output into an ORDERED, FLAT list of steps.

        No grouping, no dependency edges, no reordering. Order is exactly the
        order the model emitted. This is the ONLY planning operation FlatPlanner
        performs.
        """
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
                )
            )
        return steps, rejected

    def plan(self, workspace: TaskWorkspace) -> Plan:
        prompt = self.build_prompt(workspace)
        raw = self.adapter.complete(prompt)
        steps, rejected = self._linear_parse(raw)
        workspace.plan = [s.to_dict() for s in steps]
        return Plan(steps=steps, prompt=prompt, raw=raw, rejected=rejected)

    # --- Frozen-behavior proof -------------------------------------------
    # A stable hash over this class's contract. If anyone edits the planning
    # logic (adds capability, reorders, etc.), this hash changes and the
    # preregistration's "FlatPlanner behavior frozen" check fails loudly
    # instead of silently contaminating Exp3.
    @classmethod
    def behavior_hash(cls) -> str:
        contract = json.dumps(
            {
                "capabilities": cls.CAPABILITIES,
                "parse": "linear_ordered_no_rewrite",
                "prompt_style": "flat_no_hierarchy",
                "methods": sorted(
                    ["build_prompt", "_linear_parse", "plan", "behavior_hash"]
                ),
            },
            sort_keys=True,
        )
        return hashlib.sha256(contract.encode()).hexdigest()[:16]
