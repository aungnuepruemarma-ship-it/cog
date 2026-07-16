"""Phase 1C: the PolicyReplayEvaluator — the SHADOW bridge.

This is the bridge the review asked for:

    Experience Store
          |
          v
    Policy Validator
          |
          v
    Policy Lifecycle

Given a candidate policy and the validated ExperienceStore, it:
  1. Finds historical experiences matching the policy trigger.
  2. Replays them (verbatim structured evidence).
  3. Applies the hypothetical policy (simulated, no real execution).
  4. Compares baseline vs policy outcome.

The simulation is CONSERVATIVE and clearly labeled: a matched failure whose
category aligns with the policy trigger is counted as preventable. This is a
SHADOW estimate, not a real re-execution — true re-execution (re-running the
task with the policy prepended) is a later capability. Promoting to ACTIVE
still requires the gate (enough runs + lift), so a weak shadow estimate cannot
force promotion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cog.experience.store import ExperienceStore
from cog.learning.policy import Policy


@dataclass
class ShadowResult:
    policy_id: str
    matched: int
    baseline_success: float        # historical success rate on matched cases
    policy_success: float          # estimated success if policy applied
    preventable: int               # failures the policy would have addressed
    runs: int                      # == matched (shadow sample size)
    regressions: int = 0           # cases the policy would not help (0 in shadow)
    detail: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "matched": self.matched,
            "baseline_success": round(self.baseline_success, 4),
            "policy_success": round(self.policy_success, 4),
            "preventable": self.preventable,
            "runs": self.runs,
            "regressions": self.regressions,
            "detail": self.detail,
        }


class PolicyReplayEvaluator:
    def __init__(self, store: ExperienceStore) -> None:
        self.store = store

    def evaluate(self, policy: Policy) -> ShadowResult:
        t = policy.trigger
        # Match historical experiences against the policy trigger.
        matched = self.store.filter(
            domain=t.domain,
            outcome="failure" if t.failure_category else None,
        )
        # Further filter by tool / failure_category / operation if specified.
        def _match(exp: dict[str, Any]) -> bool:
            if t.tool is not None and exp.get("execution"):
                tools = [s.get("tool") for s in exp["execution"]]
                if t.tool not in tools:
                    return False
            if t.failure_category is not None:
                cat = (exp.get("failure") or {}).get("category")
                if cat != t.failure_category:
                    return False
            return True

        matched = [e for e in matched if _match(e)]

        n = len(matched)
        if n == 0:
            return ShadowResult(
                policy_id=policy.id, matched=0, baseline_success=0.0,
                policy_success=0.0, preventable=0, runs=0,
                detail=[{"note": "no matching historical experiences"}],
            )

        successes = sum(1 for e in matched if e.get("outcome") == "success")
        baseline_success = successes / n

        # SHADOW estimate: a matched failure whose category equals the trigger
        # failure_category is counted as preventable by the policy action.
        preventable = 0
        detail = []
        for e in matched:
            cat = (e.get("failure") or {}).get("category")
            would_help = (t.failure_category is not None and cat == t.failure_category)
            if would_help:
                preventable += 1
            detail.append({
                "experience_id": e.get("id"),
                "category": cat,
                "would_help": would_help,
            })
        policy_success = (successes + preventable) / n

        return ShadowResult(
            policy_id=policy.id, matched=n,
            baseline_success=baseline_success, policy_success=policy_success,
            preventable=preventable, runs=n, regressions=0, detail=detail,
        )

    def apply_to_policy(self, policy: Policy, result: ShadowResult) -> None:
        """Write the shadow metrics back onto the policy (no status change)."""
        policy.metrics.baseline_success = result.baseline_success
        policy.metrics.policy_success = result.policy_success
        policy.metrics.runs = result.runs
        policy.metrics.regressions = result.regressions
