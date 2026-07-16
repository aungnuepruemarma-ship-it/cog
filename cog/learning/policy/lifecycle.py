"""Phase 3, Policy track: the PolicyLifecycle state machine + belief dependency.

States (7): OBSERVED -> CANDIDATE -> EXPERIMENTAL -> VALIDATED -> ACTIVE
           -> CHALLENGED -> RETIRED, plus REVIEW_REQUIRED.

Transitions are explicit and gated:
  * promoting to ACTIVE re-validates justification against the belief store
    (a policy cannot become active on a now-invalid belief).
  * a CHALLENGED belief cascades its dependent ACTIVE policies to
    REVIEW_REQUIRED (not direct retirement) — a challenged belief may only
    need a narrower scope, so we flag for human/agent review rather than
    destroy the policy.

No LLM. No automatic activation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cog._util import utc_now
from cog.learning.belief.model import BeliefState
from cog.learning.belief.store import BeliefStore
from cog.learning.policy.model import Policy, PolicyState
from cog.learning.policy.store import PolicyStore


_ALLOWED = {
    PolicyState.OBSERVED: {PolicyState.CANDIDATE},
    PolicyState.CANDIDATE: {PolicyState.EXPERIMENTAL, PolicyState.RETIRED},
    PolicyState.EXPERIMENTAL: {PolicyState.VALIDATED, PolicyState.CANDIDATE, PolicyState.RETIRED},
    PolicyState.VALIDATED: {PolicyState.ACTIVE, PolicyState.RETIRED},
    PolicyState.ACTIVE: {PolicyState.CHALLENGED, PolicyState.RETIRED, PolicyState.REVIEW_REQUIRED},
    PolicyState.CHALLENGED: {PolicyState.RETIRED, PolicyState.ACTIVE, PolicyState.REVIEW_REQUIRED},
    PolicyState.REVIEW_REQUIRED: {PolicyState.ACTIVE, PolicyState.RETIRED, PolicyState.CHALLENGED},
    PolicyState.RETIRED: set(),
}


@dataclass
class PolicyTransition:
    at: str
    fro: str
    to: str
    reason: str


class PolicyLifecycle:
    def __init__(self, policies: PolicyStore, beliefs: BeliefStore) -> None:
        self.policies = policies
        self.beliefs = beliefs

    def _move(self, policy: Policy, target: PolicyState, reason: str) -> PolicyTransition:
        if target not in _ALLOWED.get(policy.state, set()):
            raise ValueError(f"illegal policy transition {policy.state.value} -> {target.value}")
        fro = policy.state.value
        policy.state = target
        policy.last_validated = utc_now()
        self.policies.save_state(policy, fro=fro, reason=reason)
        return PolicyTransition(at=utc_now(), fro=fro, to=target.value, reason=reason)

    def promote(self, policy: Policy, target: PolicyState, reason: str = "") -> PolicyTransition:
        # CRITICAL: re-validate justification before any promotion, especially ACTIVE.
        problems = policy.validate(self.beliefs)
        if problems:
            raise ValueError(f"policy {policy.id} fails justification: {problems}")
        return self._move(policy, target, reason or f"promote to {target.value}")

    def to_candidate(self, policy: Policy) -> PolicyTransition:
        return self._move(policy, PolicyState.CANDIDATE, "synthesized candidate")

    def to_experimental(self, policy: Policy, reason: str = "validation passed") -> PolicyTransition:
        return self.promote(policy, PolicyState.EXPERIMENTAL, reason)

    def to_validated(self, policy: Policy, reason: str = "experiment confirmed") -> PolicyTransition:
        return self.promote(policy, PolicyState.VALIDATED, reason)

    def to_active(self, policy: Policy, reason: str = "deployed") -> PolicyTransition:
        return self.promote(policy, PolicyState.ACTIVE, reason)

    def challenge(self, policy: Policy, reason: str = "belief challenged") -> PolicyTransition:
        return self._move(policy, PolicyState.CHALLENGED, reason)

    def retire(self, policy: Policy, reason: str = "retired") -> PolicyTransition:
        return self._move(policy, PolicyState.RETIRED, reason)

    # ---- belief dependency cascade ---- #
    def on_belief_challenged(self, belief_id: str) -> list[PolicyTransition]:
        """Flag dependent ACTIVE policies for review (do not silently retire)."""
        dependent = self.policies.by_belief(belief_id)
        moves: list[PolicyTransition] = []
        for policy in dependent:
            if policy.state == PolicyState.ACTIVE:
                moves.append(self._move(policy, PolicyState.REVIEW_REQUIRED,
                                        f"belief {belief_id} challenged"))
            elif policy.state == PolicyState.VALIDATED:
                moves.append(self._move(policy, PolicyState.REVIEW_REQUIRED,
                                        f"belief {belief_id} challenged"))
        return moves
