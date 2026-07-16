"""Phase 3, component 2: the BeliefLifecycle state machine.

PROPOSED -> TESTING -> SUPPORTED -> ACTIVE -> (CHALLENGED) -> RETIRED.

Transitions are explicit and gated by the quarantine verdict (see
quarantine.py). Nothing auto-promotes. ACTIVE is the only state that may
influence future behavior; reaching it requires a passed validation
experiment, not mere repetition.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cog._util import utc_now
from cog.learning.belief.model import Belief, BeliefState


_ALLOWED = {
    BeliefState.PROPOSED: {BeliefState.TESTING},
    BeliefState.TESTING: {BeliefState.SUPPORTED, BeliefState.PROPOSED},
    BeliefState.SUPPORTED: {BeliefState.ACTIVE, BeliefState.PENDING_REVIEW, BeliefState.RETIRED},
    BeliefState.ACTIVE: {BeliefState.CHALLENGED, BeliefState.PENDING_REVIEW, BeliefState.RETIRED},
    BeliefState.PENDING_REVIEW: {BeliefState.ACTIVE, BeliefState.RETIRED},
    BeliefState.CHALLENGED: {BeliefState.RETIRED, BeliefState.ACTIVE},
    BeliefState.RETIRED: set(),
}


@dataclass
class BeliefTransition:
    at: str
    fro: str
    to: str
    reason: str


class BeliefLifecycle:
    def __init__(self) -> None:
        self.transitions: list[BeliefTransition] = []

    def _move(self, belief: Belief, target: BeliefState, reason: str) -> BeliefTransition:
        if target not in _ALLOWED.get(belief.state, set()):
            raise ValueError(f"illegal belief transition {belief.state.value} -> {target.value}")
        fro = belief.state.value
        belief.state = target
        belief.last_reviewed = utc_now()
        t = BeliefTransition(at=utc_now(), fro=fro, to=target.value, reason=reason)
        self.transitions.append(t)
        return t

    def to_testing(self, belief: Belief, reason: str = "enter quarantine/validation") -> BeliefTransition:
        if not belief.evidence_ids:
            raise ValueError("cannot test a belief with no evidence")
        return self._move(belief, BeliefState.TESTING, reason)

    def to_supported(self, belief: Belief, reason: str = "validation passed") -> BeliefTransition:
        if belief.state != BeliefState.TESTING:
            raise ValueError(f"cannot support from {belief.state.value}")
        return self._move(belief, BeliefState.SUPPORTED, reason)

    def to_active(self, belief: Belief, reason: str = "promoted after support") -> BeliefTransition:
        if belief.state != BeliefState.SUPPORTED:
            raise ValueError(f"cannot activate from {belief.state.value}")
        return self._move(belief, BeliefState.ACTIVE, reason)

    def to_pending_review(self, belief: Belief, reason: str = "high-impact transition held for gate") -> BeliefTransition:
        """Hold a high-impact transition (broad rule / strong conflict / safety)."""
        return self._move(belief, BeliefState.PENDING_REVIEW, reason)

    def approve(self, belief: Belief, reason: str = "review approved") -> BeliefTransition:
        """Resolve a PENDING_REVIEW in favor of promotion."""
        if belief.state != BeliefState.PENDING_REVIEW:
            raise ValueError(f"approve requires PENDING_REVIEW, got {belief.state.value}")
        return self._move(belief, BeliefState.ACTIVE, reason)

    def reject(self, belief: Belief, reason: str = "review rejected") -> BeliefTransition:
        """Resolve a PENDING_REVIEW by retiring the candidate."""
        if belief.state != BeliefState.PENDING_REVIEW:
            raise ValueError(f"reject requires PENDING_REVIEW, got {belief.state.value}")
        return self._move(belief, BeliefState.RETIRED, reason)

    def to_challenged(self, belief: Belief, by_belief_id: str = "",
                      reason: str = "contradictory evidence") -> BeliefTransition:
        if by_belief_id:
            belief.contradicted_by.append(by_belief_id)
        return self._move(belief, BeliefState.CHALLENGED, reason)

    def retire(self, belief: Belief, reason: str = "retired",
               superseded_by: str | None = None) -> BeliefTransition:
        if superseded_by is not None:
            belief.superseded_by = superseded_by
        return self._move(belief, BeliefState.RETIRED, reason)

    def history(self) -> list[dict[str, Any]]:
        return [asdict(t) for t in self.transitions]
