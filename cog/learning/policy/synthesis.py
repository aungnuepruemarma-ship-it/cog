"""Phase 3, Policy track: deterministic candidate-policy generation.

A belief asserts "under condition C, failure is likely." That is NOT a proof
of which intervention is best. So synthesis is a GENERATOR: from one ACTIVE
belief it emits several candidate policies, each a distinct candidate action
(dependency scan, auto-install, manifest check, ...). All cite the same
belief as justification; all start in CANDIDATE. The policy experiment (not
this function) decides which, if any, earns promotion.

This is the deliberate correction vs. a direct Belief->Action mapping: the
transformation is a reasoning step that admits multiple hypotheses, and only
experimentation resolves them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cog._util import new_id, utc_now
from cog.learning.belief.model import Belief, BeliefState
from cog.learning.belief.store import BeliefStore
from cog.learning.policy.model import Policy, PolicyEffect, PolicyState

# Candidate interventions considered for a "missing dependency" failure belief.
# Deterministic list; a later phase can learn these, but v0.1 enumerates them.
DEPENDENCY_ACTIONS = [
    ("run dependency inspection before build", "dependency_inspection",
     "docker_build_failure_rate", -0.40),
    ("auto-install missing packages before build", "auto_install",
     "docker_build_failure_rate", -0.45),
    ("validate package manifest before build", "manifest_check",
     "docker_build_failure_rate", -0.35),
]


@dataclass
class SynthesisResult:
    candidates: list[Policy]
    source_belief_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [p.to_dict() for p in self.candidates],
            "source_belief_id": self.source_belief_id,
        }


def _action_for_belief(belief: Belief) -> list[tuple[str, str, str, float]]:
    """Return candidate (action_text, action_key, metric, expected_delta) tuples.

    Currently specialized for dependency_failure beliefs; generalizes later.
    """
    cat = belief.claim.prediction.get("category") or belief.claim.condition.get("category")
    if cat == "dependency_failure":
        return DEPENDENCY_ACTIONS
    # Generic fallback: a single sensible pre-check.
    return [("add a pre-check before the failing operation", "precheck",
             f"{belief.claim.condition.get('task', 'op')}_failure_rate", -0.20)]


def synthesize_candidates(belief_store: BeliefStore) -> list[SynthesisResult]:
    """Generate candidate policies from SUPPORTED/ACTIVE beliefs."""
    active = belief_store.by_state(BeliefState.ACTIVE) + belief_store.by_state(BeliefState.SUPPORTED)
    results: list[SynthesisResult] = []
    for belief in active:
        tool = belief.claim.condition.get("task")
        domain = belief.scope.domain
        actions = _action_for_belief(belief)
        policies: list[Policy] = []
        for action_text, action_key, metric, delta in actions:
            pol = Policy(
                id=new_id("pol"),
                action=action_text,
                trigger={"task_type": belief.scope.task_type, "tool": tool, "domain": domain},
                justification=[belief.id],
                state=PolicyState.CANDIDATE,
                confidence=belief.statistics.success_rate,
                evidence_ids=list(belief.evidence_ids),
                expected_effect=PolicyEffect(metric=metric, direction="decrease", expected_delta=delta),
                validation_results={},
                created_at=utc_now(),
                last_validated=utc_now(),
            )
            policies.append(pol)
        if policies:
            # Validate against the belief store before emitting (superstition guard).
            policies = [p for p in policies if p.is_valid(belief_store)]
            results.append(SynthesisResult(candidates=policies, source_belief_id=belief.id))
    return results
