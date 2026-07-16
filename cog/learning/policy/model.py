"""Phase 3 (Epistemic Engine), Policy track, component 1: the Policy schema v0.1.

A Policy is "what should we do?" — distinct from a Belief ("what is true?").
A policy is NEVER valid without justification: it must reference one or more
SUPPORTED/ACTIVE beliefs. This is what prevents policies from becoming a bag
of tricks (superstition).

The lifecycle is 7 states (one more than the review's first sketch): a policy
can be tested once (EXPERIMENTAL), validated statistically (VALIDATED),
deployed (ACTIVE), and later disproven (CHALLENGED) without being silently
retired. CHALLENGED cascades dependent policies to REVIEW_REQUIRED rather
than deleting them — a challenged belief may only need a narrower scope.

No LLM. No fuzzy retrieval. No automatic activation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from cog.learning.belief.model import BeliefState


class PolicyState(str, Enum):
    OBSERVED = "observed"
    CANDIDATE = "candidate"
    EXPERIMENTAL = "experimental"
    VALIDATED = "validated"
    ACTIVE = "active"
    CHALLENGED = "challenged"
    RETIRED = "retired"
    REVIEW_REQUIRED = "review_required"


@dataclass
class PolicyEffect:
    """The explicit effect model: what this policy claims to change."""
    metric: str                                   # e.g. "docker_build_failure_rate"
    direction: str = "decrease"                   # decrease | increase
    expected_delta: float = 0.0                   # expected absolute change (e.g. -0.40)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PolicyEffect:
        return cls(metric=d.get("metric", ""), direction=d.get("direction", "decrease"),
                   expected_delta=d.get("expected_delta", 0.0))


@dataclass
class Policy:
    id: str
    action: str                                   # prescriptive action (human-readable)
    trigger: dict[str, Any] = field(default_factory=dict)
    justification: list[str] = field(default_factory=list)   # belief_ids (REQUIRED)
    state: PolicyState = PolicyState.OBSERVED
    confidence: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    expected_effect: PolicyEffect = field(default_factory=PolicyEffect)
    validation_results: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    last_validated: str | None = None

    # ---- serialization ---- #
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        d["expected_effect"] = self.expected_effect.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Policy:
        d = dict(data)
        d["state"] = PolicyState(d["state"])
        d["expected_effect"] = PolicyEffect.from_dict(d.get("expected_effect", {}))
        return cls(**d)

    # ---- the critical invariant: a policy must be justified ---- #
    def validate(self, belief_store) -> list[str]:
        problems: list[str] = []
        if not self.id:
            problems.append("policy id empty")
        if not self.action:
            problems.append("policy has no action")
        if not self.justification:
            problems.append("policy references no belief (superstition guard)")
        if not (0.0 <= self.confidence <= 1.0):
            problems.append(f"confidence out of [0,1]: {self.confidence}")
        # Each justification must resolve to a SUPPORTED/ACTIVE belief.
        for bid in self.justification:
            belief = belief_store.get(bid) if belief_store is not None else None
            if belief is None:
                problems.append(f"justification belief {bid} not found")
            elif belief.state not in (BeliefState.SUPPORTED, BeliefState.ACTIVE):
                problems.append(
                    f"policy cites belief {bid} in state {belief.state.value}; "
                    f"only SUPPORTED/ACTIVE beliefs may justify a policy"
                )
        return problems

    def is_valid(self, belief_store) -> bool:
        return not self.validate(belief_store)
