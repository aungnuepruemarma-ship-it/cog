"""Phase 3 → Runtime: policy injection (the missing integration layer).

This is the seam that turns the cognitive loop into a SELF-IMPROVING runtime:

    Experience -> Belief -> Policy -> [THIS LAYER] -> Planner -> Execution -> New Experience

Design rule (from the review): a policy is a CONSTRAINT / HINT to the planner,
NEVER an authority over the executor. The planner still chooses and emits the
final plan; the policy only shapes it (e.g. prepends a required pre-check
step). The executor runs whatever the plan says.

Components:
  * PolicyContext  -- carries the active policies + their justifications to the
                      planner, so a human/agent can audit WHY a plan looks the
                      way it does.
  * PolicyAwareAdapter -- wraps the base ModelAdapter; when active policies
                      match, it prepends the policy's concrete pre-step(s) to
                      the produced plan. Deterministic, no LLM, no authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cog.execution.planner import Plan, PlanStep
from cog.runtime.adapter import ModelAdapter
from cog.learning.policy.model import Policy


# Map a policy action string to a concrete tool step the planner can emit.
def _action_to_step(action: str) -> PlanStep | None:
    a = action.lower()
    if "inspection" in a or "preflight" in a or "dependency scan" in a:
        return PlanStep(tool="dep_preflight", args={}, description="policy: dependency preflight")
    if "auto-install" in a or "install missing" in a:
        return PlanStep(tool="dep_install", args={}, description="policy: auto-install deps")
    if "manifest" in a:
        return PlanStep(tool="dep_manifest", args={}, description="policy: validate manifest")
    if "pre-check" in a:
        return PlanStep(tool="dep_preflight", args={}, description="policy: pre-check")
    return None


@dataclass
class PolicyContext:
    """What the planner receives: the matched active policies + why."""
    policies: list[Policy] = field(default_factory=list)
    justifications: list[list[str]] = field(default_factory=list)

    @property
    def policy_ids(self) -> list[str]:
        return [p.id for p in self.policies]

    def preflight_steps(self) -> list[PlanStep]:
        """Concrete planning steps implied by the active policies (deduped, ordered)."""
        steps: list[PlanStep] = []
        seen: set[str] = set()
        for p in self.policies:
            step = _action_to_step(p.action)
            if step is None or step.tool in seen:
                continue
            seen.add(step.tool)
            steps.append(step)
        return steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_ids": self.policy_ids,
            "justifications": self.justifications,
            "preflight_steps": [s.tool for s in self.preflight_steps()],
        }


class PolicyAwareAdapter:
    """Wraps a base ModelAdapter so active policies shape the produced plan.

    The base adapter produces the plan as usual; this wrapper prepends the
    policy-implied pre-steps. The planner (and thus the executor) still see a
    single coherent plan — the policy is a hint, not a separate authority.
    """

    name = "policy_aware"

    def __init__(self, base: ModelAdapter, context: PolicyContext) -> None:
        self.base = base
        self.context = context

    def complete(self, prompt: str) -> str:
        raw = self.base.complete(prompt)
        pre = self.context.preflight_steps()
        if not pre:
            return raw
        pre_lines = "\n".join(f"step: {s.tool} {{}} -- {s.description}" for s in pre)
        # Prepend policy steps so they execute BEFORE the base plan's steps.
        return pre_lines + "\n" + raw


def get_active_policies(task: Any, policy_store) -> PolicyContext:
    """Select active policies for a task and build the planner context.

    ``task`` may be a full Task or a lightweight context dict. Selection is the
    deterministic trigger match from selector.py.
    """
    from cog.learning.policy.selector import select_policies

    if isinstance(task, dict):
        ctx = task
    else:
        ctx = {
            "task_type": getattr(task, "domain", "software"),
            "tools": set(getattr(task, "tools", []) or []),
            "domain": getattr(task, "domain", "software"),
        }
    policies = select_policies(ctx, policy_store)
    return PolicyContext(
        policies=policies,
        justifications=[list(p.justification) for p in policies],
    )
