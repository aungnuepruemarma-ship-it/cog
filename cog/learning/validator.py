"""Phase 1C: the PolicyValidator — deterministic promotion gates.

These are the rules that keep Cog from uncontrolled self-modification. No
transition into a higher-trust state is allowed unless the corresponding gate
passes. Every gate is a pure function over the Policy + its shadow metrics,
so promotion is auditable and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from cog.learning.policy import Policy, PolicyStatus


@dataclass
class GateResult:
    passed: bool
    reason: str

    def __bool__(self) -> bool:  # so `if gate:` works
        return self.passed


# Default thresholds. Kept as data so they can be tuned without code changes.
MIN_EVIDENCE_FOR_CANDIDATE = 3          # repeated pattern, not a one-off
MIN_EVIDENCE_FOR_SHADOW = 5
MIN_RUNS_FOR_ACTIVE = 30               # shadow must be evaluated on enough cases
MIN_LIFT_FOR_ACTIVE = 0.10             # policy_success - baseline >= 10 pts
MAX_REGRESSIONS_FOR_ACTIVE = 0


def can_become_candidate(policy: Policy,
                         min_evidence: int = MIN_EVIDENCE_FOR_CANDIDATE) -> GateResult:
    if len(policy.evidence) < min_evidence:
        return GateResult(False, f"need >= {min_evidence} evidence, have {len(policy.evidence)}")
    return GateResult(True, f"{len(policy.evidence)} evidence >= {min_evidence}")


def can_enter_shadow(policy: Policy,
                     min_evidence: int = MIN_EVIDENCE_FOR_SHADOW) -> GateResult:
    if len(policy.evidence) < min_evidence:
        return GateResult(False, f"need >= {min_evidence} evidence for shadow, have {len(policy.evidence)}")
    return GateResult(True, "evidence sufficient for shadow evaluation")


def can_become_active(policy: Policy,
                     min_runs: int = MIN_RUNS_FOR_ACTIVE,
                     min_lift: float = MIN_LIFT_FOR_ACTIVE,
                     max_regressions: int = MAX_REGRESSIONS_FOR_ACTIVE) -> GateResult:
    m = policy.metrics
    if m.runs < min_runs:
        return GateResult(False, f"shadow runs {m.runs} < {min_runs}")
    lift = m.policy_success - m.baseline_success
    if lift < min_lift:
        return GateResult(False, f"lift {lift:.3f} < {min_lift}")
    if m.regressions > max_regressions:
        return GateResult(False, f"{m.regressions} regressions > {max_regressions}")
    if policy.status not in (PolicyStatus.SHADOW, PolicyStatus.TESTED):
        return GateResult(False, f"cannot go active from {policy.status.value}")
    return GateResult(True, f"lift {lift:.3f} over {m.runs} runs, no regressions")


def can_retire(policy: Policy, *,
               unused_runs: int = 0,
               better_replacement: str | None = None,
               repeated_failures: int = 0) -> GateResult:
    if better_replacement is not None:
        return GateResult(True, f"superseded by {better_replacement}")
    if repeated_failures > 0:
        return GateResult(True, f"{repeated_failures} repeated application failures")
    if unused_runs >= 50:
        return GateResult(True, f"unused for {unused_runs} runs")
    return GateResult(False, "no retirement trigger met")
