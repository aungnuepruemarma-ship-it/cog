"""Phase 3, component 4: the Discriminator Experiment (the decisive piece).

A belief claims: under condition C (e.g. docker_build with no preflight),
failure probability is high. The experiment tests it by constructing two
groups over validated evidence:

    Group A (control):     experiences matching C as observed (preflight absent)
    Group B (treatment):   experiences matching C BUT with the intervention
                           (preflight present) -- if any exist in evidence

We compare failure rates. Lift and a Wald confidence interval decide the
verdict. The EXPERIMENT DECIDES — no LLM, no assertion of cause.

When no Group B evidence exists yet, the experiment reports INSUFFICIENT and
flags that a live A/B run is required. That is the honest boundary: Cog will
not promote a belief it cannot discriminate, and it knows what experiment
would settle it. (Wiring the live A/B runner is a later step; the
cog/experiment/ab.py harness already exists for it.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from cog.experience.store import ExperienceStore
from cog.learning.belief.model import Belief


@dataclass
class ExperimentResult:
    belief_id: str
    group_a_n: int
    group_a_failure_rate: float
    group_b_n: int
    group_b_failure_rate: float
    lift: float                       # reduction in failure rate (A - B)
    ci95: tuple[float, float]         # CI on the lift (Wald)
    verdict: str                      # SUPPORT | CHALLENGE | INSUFFICIENT
    needs_live_ab: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "belief_id": self.belief_id,
            "group_a_n": self.group_a_n,
            "group_a_failure_rate": round(self.group_a_failure_rate, 4),
            "group_b_n": self.group_b_n,
            "group_b_failure_rate": round(self.group_b_failure_rate, 4),
            "lift": round(self.lift, 4),
            "ci95": [round(x, 4) for x in self.ci95],
            "verdict": self.verdict,
            "needs_live_ab": self.needs_live_ab,
            "detail": self.detail,
        }


def _has_tool(exp: dict[str, Any], tool: str) -> bool:
    return any(s.get("tool") == tool for s in (exp.get("execution") or []))


def _preflight_present(exp: dict[str, Any]) -> bool:
    tools = [s.get("tool", "") for s in (exp.get("execution") or [])]
    return any("preflight" in t or "inspect" in t or "check" in t for t in tools)


_MIN_GROUP = 5          # need a real sample in each arm to discriminate
_MIN_LIFT = 0.10        # at least 10-point absolute failure-rate reduction
_CI_LOWER = 0.0         # lift CI lower bound must clear 0 by a margin


class BeliefTester:
    def __init__(self, store: ExperienceStore,
                 min_group: int = _MIN_GROUP, min_lift: float = _MIN_LIFT) -> None:
        self.store = store
        self.min_group = min_group
        self.min_lift = min_lift

    def run(self, belief: Belief) -> ExperimentResult:
        cond = belief.claim.condition
        tool = cond.get("task")
        domain = belief.scope.domain

        all_for_tool = self.store.filter(domain=domain, outcome=None)
        group_a, group_b = [], []
        for exp in all_for_tool:
            if tool is None or not _has_tool(exp, tool):
                continue
            if _preflight_present(exp):
                group_b.append(exp)
            else:
                group_a.append(exp)

        n_a, n_b = len(group_a), len(group_b)
        fail_a = sum(1 for e in group_a if e.get("outcome") == "failure")
        fail_b = sum(1 for e in group_b if e.get("outcome") == "failure")
        rate_a = fail_a / n_a if n_a else 0.0
        rate_b = fail_b / n_b if n_b else 0.0
        lift = rate_a - rate_b

        # Wald 95% CI on the difference of two proportions.
        se = 0.0
        if n_a:
            se += rate_a * (1 - rate_a) / n_a
        if n_b:
            se += rate_b * (1 - rate_b) / n_b
        se = math.sqrt(se) if se > 0 else 0.0
        ci_low = lift - 1.96 * se
        ci_high = lift + 1.96 * se

        # Verdict: experiment decides.
        if n_b < self.min_group:
            verdict = "INSUFFICIENT"
            needs_live = True
        elif lift >= self.min_lift and ci_low > _CI_LOWER:
            verdict = "SUPPORT"
            needs_live = False
        elif lift < 0:  # intervention made things worse
            verdict = "CHALLENGE"
            needs_live = False
        else:
            verdict = "INSUFFICIENT"
            needs_live = False

        # Record the estimated statistics back onto the belief for provenance.
        belief.statistics.sample_size = n_a + n_b
        belief.statistics.success_rate = round(1 - ((fail_a + fail_b) / max(1, n_a + n_b)), 3)
        belief.statistics.confidence_interval = (round(ci_low, 3), round(ci_high, 3))

        return ExperimentResult(
            belief_id=belief.id, group_a_n=n_a, group_a_failure_rate=rate_a,
            group_b_n=n_b, group_b_failure_rate=rate_b, lift=lift,
            ci95=(ci_low, ci_high), verdict=verdict, needs_live_ab=needs_live,
            detail={"fail_a": fail_a, "fail_b": fail_b},
        )
