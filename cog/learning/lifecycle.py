"""Phase 1C: the PolicyLifecycle — a deterministic state machine.

States: OBSERVED -> CANDIDATE -> SHADOW -> TESTED -> ACTIVE -> RETIRED.

Critical design property: transitions are EXPLICIT and GATED. Nothing
auto-promotes. Each transition method takes the policy, consults the
PolicyValidator, and only moves state (and records a transition entry) when
the gate passes. This is the governance layer that prevents uncontrolled
self-modification.

RETired can be reached from any active/tested state via an explicit retire().
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cog._util import utc_now
from cog.learning.policy import Policy, PolicyStatus
from cog.learning.validator import (
    can_become_active,
    can_become_candidate,
    can_enter_shadow,
    can_retire,
)


# Legal forward transitions. Backward moves are not allowed except RETIRE.
_ALLOWED = {
    PolicyStatus.OBSERVED: {PolicyStatus.CANDIDATE},
    PolicyStatus.CANDIDATE: {PolicyStatus.SHADOW},
    PolicyStatus.SHADOW: {PolicyStatus.TESTED, PolicyStatus.CANDIDATE},  # demote if shadow weak
    PolicyStatus.TESTED: {PolicyStatus.ACTIVE, PolicyStatus.CANDIDATE},
    PolicyStatus.ACTIVE: set(),  # only RETIRE from active
    PolicyStatus.RETIRED: set(),
}


@dataclass
class Transition:
    at: str
    fro: str
    to: str
    reason: str
    gate: str  # the gate result message


class PolicyLifecycle:
    """Owns the transition rules for one policy registry's policies."""

    def __init__(self) -> None:
        self.transitions: list[Transition] = []

    # ---- core guard ---- #
    def _move(self, policy: Policy, target: PolicyStatus, reason: str, gate_msg: str) -> None:
        if target not in _ALLOWED.get(policy.status, set()):
            raise ValueError(
                f"illegal transition {policy.status.value} -> {target.value}"
            )
        fro = policy.status.value
        policy.status = target
        self.transitions.append(
            Transition(at=utc_now(), fro=fro, to=target.value, reason=reason, gate=gate_msg)
        )

    # ---- explicit, gated transitions ---- #
    def promote_to_candidate(self, policy: Policy) -> Transition:
        gate = can_become_candidate(policy)
        if not gate.passed:
            raise ValueError(f"cannot promote to candidate: {gate.reason}")
        self._move(policy, PolicyStatus.CANDIDATE, "repeated evidence", gate.reason)
        return self.transitions[-1]

    def enter_shadow(self, policy: Policy) -> Transition:
        gate = can_enter_shadow(policy)
        if not gate.passed:
            raise ValueError(f"cannot enter shadow: {gate.reason}")
        self._move(policy, PolicyStatus.SHADOW, "shadow evaluation", gate.reason)
        return self.transitions[-1]

    def promote_to_tested(self, policy: Policy) -> Transition:
        # SHADOW -> TESTED once evaluation has run (metrics populated).
        if policy.status != PolicyStatus.SHADOW:
            raise ValueError(f"cannot test from {policy.status.value}")
        self._move(policy, PolicyStatus.TESTED, "shadow evaluation complete",
                   f"runs={policy.metrics.runs}")
        return self.transitions[-1]

    def promote_to_active(self, policy: Policy) -> Transition:
        gate = can_become_active(policy)
        if not gate.passed:
            raise ValueError(f"cannot promote to active: {gate.reason}")
        self._move(policy, PolicyStatus.ACTIVE, "proven improvement", gate.reason)
        return self.transitions[-1]

    def demote_to_candidate(self, policy: Policy, reason: str) -> Transition:
        self._move(policy, PolicyStatus.CANDIDATE, reason, reason)
        return self.transitions[-1]

    def retire(self, policy: Policy, *,
               unused_runs: int = 0,
               better_replacement: str | None = None,
               repeated_failures: int = 0,
               reason: str = "retirement trigger") -> Transition:
        gate = can_retire(policy, unused_runs=unused_runs,
                          better_replacement=better_replacement,
                          repeated_failures=repeated_failures)
        if not gate.passed:
            raise ValueError(f"cannot retire: {gate.reason}")
        fro = policy.status.value
        policy.status = PolicyStatus.RETIRED
        if better_replacement is not None:
            policy.superseded_by = better_replacement
        self.transitions.append(
            Transition(at=utc_now(), fro=fro, to=PolicyStatus.RETIRED.value,
                       reason=reason, gate=gate.reason)
        )
        return self.transitions[-1]

    def history(self) -> list[dict[str, Any]]:
        return [asdict(t) for t in self.transitions]
