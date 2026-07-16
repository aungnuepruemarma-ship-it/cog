"""Governance-v0.2: the learning governance policy (pure, no side effects).

This module is the SCIENTIFIC OBJECT the design note specifies. It contains
only deterministic functions over belief evidence; it performs NO database
access, NO mutation, and has NO side effects. That makes every rule unit-
testable in isolation.

Components:
    PromotionScore   -- multi-signal score in [0,1]
    ConfidencePolicy -- maps a score (+ gate signals) to a lifecycle decision
    is_broad         -- the BroadRuleDetector (scope-expansion check)

The policy is the safety boundary that must exist before any autonomous
research loop is allowed to modify Cog itself (see COG_LEARNING_GOVERNANCE.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cog.learning.belief.model import Belief, BeliefState

# Approved default weights (COG_LEARNING_GOVERNANCE.md section 3).
DEFAULT_WEIGHTS: dict[str, float] = {
    "confidence": 0.25,
    "evidence_qty": 0.15,
    "evidence_diversity": 0.20,
    "cross_session": 0.15,
    "runtime_success": 0.20,
    "contradiction_penalty": 0.05,
}

# Score bands (COG_LEARNING_GOVERNANCE.md section 3C).
BAND_QUARANTINE = 0.4   # below -> held (TESTING), not promoted
BAND_CANDIDATE = 0.7    # below -> TESTING candidate
BAND_ACTIVE = 0.9       # below -> SUPPORTED; at/above -> ACTIVE unless gated

# Lexical markers that indicate a belief removes an important constraint
# (over-generalization risk).
CONSTRAINT_MARKERS = ("always", "never", "all")
# Scope strings that denote a global / unbounded scope.
BROAD_SCOPE_TOKENS = ("global", "any", "all", "*", "")


@dataclass
class PromotionSignals:
    """Normalized [0,1] inputs to the promotion score.

    `evidence_quantity` is already normalized by the caller (see
    PromotionScore.calculate, which normalizes raw counts).
    """

    confidence: float = 0.0
    evidence_quantity: float = 0.0
    evidence_diversity: float = 0.0
    cross_session_consistency: float = 0.0
    runtime_success: float = 0.0
    contradiction_penalty: float = 0.0

    def __post_init__(self) -> None:
        for name, val in self.__dict__.items():
            if name == "contradiction_penalty":
                # penalty is subtracted; allowed to be 0..1
                if not (0.0 <= val <= 1.0):
                    raise ValueError(f"{name} out of [0,1]: {val}")
            else:
                if not (0.0 <= val <= 1.0):
                    raise ValueError(f"{name} out of [0,1]: {val}")


class PromotionScore:
    """Pure computation of the promotion score from normalized signals."""

    # reference count at which evidence_quantity saturates to 1.0
    REFERENCE_EVIDENCE_N = 50

    @staticmethod
    def normalize_evidence_quantity(raw_count: int, reference_n: int | None = None) -> float:
        ref = reference_n or PromotionScore.REFERENCE_EVIDENCE_N
        if ref <= 0:
            raise ValueError("reference_n must be > 0")
        return max(0.0, min(1.0, raw_count / ref))

    @staticmethod
    def calculate(
        signals: PromotionSignals,
        weights: dict[str, float] | None = None,
    ) -> float:
        """Return the promotion score in [0,1]. No side effects."""
        w = weights or DEFAULT_WEIGHTS
        score = (
            w["confidence"] * signals.confidence
            + w["evidence_qty"] * signals.evidence_quantity
            + w["evidence_diversity"] * signals.evidence_diversity
            + w["cross_session"] * signals.cross_session_consistency
            + w["runtime_success"] * signals.runtime_success
            - w["contradiction_penalty"] * signals.contradiction_penalty
        )
        return max(0.0, min(1.0, score))


class ConfidencePolicy:
    """Maps a promotion score (+ gate signals) to a lifecycle decision.

    Pure function: given a score and the belief's gate status, returns the
    target BeliefState. It never mutates the belief and never touches the DB.
    """

    @staticmethod
    def decide_state(
        score: float,
        belief: Belief | None = None,
        *,
        broad_rule: bool = False,
        strong_conflict: bool = False,
        safety_change: bool = False,
    ) -> BeliefState:
        # The gate: high-impact transitions are held for review regardless of score.
        if broad_rule or strong_conflict or safety_change:
            return BeliefState.PENDING_REVIEW
        if belief is not None and is_broad(belief):
            return BeliefState.PENDING_REVIEW
        if score < BAND_QUARANTINE:
            return BeliefState.TESTING          # quarantined / held
        if score < BAND_CANDIDATE:
            return BeliefState.TESTING          # candidate, not yet promoted
        if score < BAND_ACTIVE:
            return BeliefState.SUPPORTED        # active candidate
        return BeliefState.ACTIVE               # promoted (unless gated above)


# --------------------------------------------------------------------------- #
# BroadRuleDetector
# --------------------------------------------------------------------------- #
def is_broad(belief: Belief) -> bool:
    """Scope-expansion check (COG_LEARNING_GOVERNANCE.md section 3 / 4b).

    A belief is BROAD (high blast radius -> PENDING_REVIEW) if ANY hold:
        condition_count > 3
        OR domain is a global/any/all token (domain_count > 1 interpretation)
        OR applies to multiple task types (scope.task_type is a broad token)
        OR scope == "global"
        OR removes important constraints ("always"/"never"/"all" in statement)

    It is NOT defined by domain alone -- a cybersecurity belief can be narrow
    or broad. Pure function over the belief's own fields + statement text.
    """
    cond = belief.claim.condition or {}
    pred = belief.claim.prediction or {}
    scope = belief.scope

    # 1) too many condition keys
    if len(cond) > 3:
        return True

    # 2) global / unbounded domain
    if (scope.domain or "").strip().lower() in BROAD_SCOPE_TOKENS:
        return True

    # 3) applies to multiple task types
    if (scope.task_type or "").strip().lower() in BROAD_SCOPE_TOKENS:
        return True
    # explicit multi-task marker in the condition
    if cond.get("task_types") or cond.get("applies_to") or cond.get("scope") == "global":
        return True

    # 4) lexical constraint-removal markers in the human-readable statement
    text = (belief.statement() or "").lower()
    if any(marker in text for marker in CONSTRAINT_MARKERS):
        return True
    # also scan prediction values for "always"/"never"/"all"
    for v in pred.values():
        if isinstance(v, str) and any(m in v.lower() for m in CONSTRAINT_MARKERS):
            return True

    return False
