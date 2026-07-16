"""Phase 3 (Epistemic Engine), component 1: the Belief schema (v0.1).

A Belief is a SCIENTIFIC HYPOTHESIS, not a prose claim:
    claim.condition  -> "in this situation ..."
    claim.prediction -> "... this outcome is expected"

This is observation-based and machine-testable. It deliberately avoids
causal interpretation ("developers forget dependencies") in favour of a
measurable conditional prediction ("when preflight is absent, failure
probability > X"). Causal mechanisms come later, once many validated
observations exist.

States: PROPOSED -> TESTING -> SUPPORTED -> ACTIVE -> CHALLENGED -> RETIRED.
No LLM is involved in the schema or its transitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class BeliefState(str, Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    ACTIVE = "active"
    PENDING_REVIEW = "pending_review"  # high-impact transition held for the gate
    CHALLENGED = "challenged"
    RETIRED = "retired"


@dataclass
class BeliefClaim:
    """A conditional, testable prediction."""
    condition: dict[str, Any] = field(default_factory=dict)
    prediction: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BeliefClaim:
        return cls(condition=d.get("condition", {}), prediction=d.get("prediction", {}))


@dataclass
class BeliefStatistics:
    sample_size: int = 0
    success_rate: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "success_rate": self.success_rate,
            "confidence_interval": list(self.confidence_interval),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BeliefStatistics:
        ci = d.get("confidence_interval", [0.0, 0.0])
        return cls(sample_size=d.get("sample_size", 0),
                   success_rate=d.get("success_rate", 0.0),
                   confidence_interval=tuple(ci) if isinstance(ci, list) else tuple(ci))


@dataclass
class BeliefScope:
    domain: str = "unspecified"
    task_type: str = "unspecified"
    environment: str = "unspecified"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BeliefScope:
        return cls(domain=d.get("domain", "unspecified"),
                   task_type=d.get("task_type", "unspecified"),
                   environment=d.get("environment", "unspecified"))


@dataclass
class Belief:
    id: str
    claim: BeliefClaim
    evidence_ids: list[str] = field(default_factory=list)
    statistics: BeliefStatistics = field(default_factory=BeliefStatistics)
    scope: BeliefScope = field(default_factory=BeliefScope)
    confidence: float = 0.0
    state: BeliefState = BeliefState.PROPOSED
    contradicted_by: list[str] = field(default_factory=list)
    superseded_by: str | None = None
    created_at: str | None = None
    last_reviewed: str | None = None
    # ---- governance / decay fields (governance-v0.2, all additive) ---- #
    last_used: str | None = None          # when the belief last influenced a decision
    last_confirmed: str | None = None     # when last confirmed by new evidence
    confirmation_count: int = 0           # number of confirming observations
    contradiction_count: int = 0          # number of contradicting observations

    # ---- serialization ---- #
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        d["claim"] = self.claim.to_dict()
        d["statistics"] = self.statistics.to_dict()
        d["scope"] = self.scope.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Belief:
        d = dict(data)
        d["state"] = BeliefState(d["state"])
        d["claim"] = BeliefClaim.from_dict(d.get("claim", {}))
        d["statistics"] = BeliefStatistics.from_dict(d.get("statistics", {}))
        d["scope"] = BeliefScope.from_dict(d.get("scope", {}))
        return cls(**d)

    # ---- deterministic validation (no LLM) ---- #
    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.id:
            problems.append("belief id is empty")
        if not self.claim.condition:
            problems.append("belief claim has no condition")
        if not self.claim.prediction:
            problems.append("belief claim has no prediction")
        if not (0.0 <= self.confidence <= 1.0):
            problems.append(f"confidence out of [0,1]: {self.confidence}")
        if self.state in (BeliefState.SUPPORTED, BeliefState.ACTIVE) and not self.evidence_ids:
            problems.append(f"{self.state.value} belief without evidence")
        return problems

    def is_valid(self) -> bool:
        return not self.validate()

    # ---- convenience: human-readable statement (derived, not stored) ---- #
    def statement(self) -> str:
        return f"IF {self.claim.condition} THEN predict {self.claim.prediction}"
