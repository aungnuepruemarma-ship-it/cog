"""Phase 3: contradiction DETECTION (v0.1 scope — detect, don't auto-resolve).

This is intentionally narrower than a full contradiction *engine*. v0.1 can
detect that new evidence conflicts with an existing SUPPORTED/ACTIVE belief;
it records the conflict (review_required flag + contradiction_count) but does
NOT auto-narrow scope or auto-retire. Auto-resolution is v0.2.

Detection rule (deterministic, no LLM):
  For an existing belief B predicting "under condition C, failure_probability
  is high", new evidence E is contradictory if E matches C (same tool/domain,
  preflight present/absent per B.condition) but E's OUTCOME is success where B
  predicted failure, OR E is a failure under the SAME condition but with the
  proposed intervention present and still failing (intervention ineffective).
We surface the conflict so a human/agent can decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cog.experience.store import ExperienceStore
from cog.experience.record import Experience
from cog.learning.belief.model import Belief, BeliefState


@dataclass
class ContradictionReport:
    belief_id: str
    conflicting_experience_ids: list[str]
    contradiction_count: int
    review_required: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "belief_id": self.belief_id,
            "conflicting_experience_ids": self.conflicting_experience_ids,
            "contradiction_count": self.contradiction_count,
            "review_required": self.review_required,
            "detail": self.detail,
        }


def _matches_condition(exp: Experience, belief: Belief) -> bool:
    """Does this experience fall under the belief's stated condition?

    Consumes the Experience domain model directly (no dict normalization).
    """
    cond = belief.claim.condition
    tool = cond.get("task")
    domain = cond.get("domain")
    if domain and exp.domain != domain:
        return False
    # tool match: does execution contain the tool under study?
    if tool:
        tools = [s.get("tool") for s in (exp.execution or [])]
        if tool not in tools:
            return False
    return True


def detect_contradiction(belief: Belief, store: ExperienceStore) -> ContradictionReport:
    """Flag new evidence that contradicts an existing belief's prediction.

    Contradiction = experiences matching the belief's condition whose OUTCOME
    is success (the belief predicted failure) -- i.e. the predicted failure
    did not occur under the stated condition.
    """
    if belief.state not in (BeliefState.SUPPORTED, BeliefState.ACTIVE):
        # Only established beliefs can be contradicted.
        return ContradictionReport(belief.id, [], 0, False)

    conflicts: list[str] = []
    predicted_failure = belief.claim.prediction.get("failure_probability", 1.0) >= 0.5
    for eid in belief.evidence_ids:
        exp = store.get(eid)
        if exp is None:
            continue
        if not _matches_condition(exp, belief):
            continue
        # If we predicted failure but observed success -> contradiction.
        if predicted_failure and exp.outcome == "success":
            conflicts.append(eid)
        # If we predicted success but observed failure -> contradiction.
        if (not predicted_failure) and exp.outcome == "failure":
            conflicts.append(eid)

    review_required = len(conflicts) > 0
    return ContradictionReport(
        belief_id=belief.id,
        conflicting_experience_ids=conflicts,
        contradiction_count=len(conflicts),
        review_required=review_required,
        detail={"predicted_failure": predicted_failure},
    )
