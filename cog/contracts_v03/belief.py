"""
COG v0.3 — Belief Contract

Beliefs are mutable working memory. They may strengthen, weaken, or disappear.

Constitutional invariant: CI-204 — Beliefs derive only from claims.
Constitutional invariant: CI-213 — No object may skip an epistemic stage.

Status: FROZEN (M1)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import hashlib


class BeliefState(Enum):
    """State of a belief in the epistemic pipeline."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CONTRADICTED = "contradicted"
    RETRACTED = "retracted"
    PROMOTED = "promoted"


@dataclass(frozen=True, slots=True)
class Belief:
    """
    Beliefs are mutable working memory.

    Pipeline: Claim → Belief → Knowledge

    They can:
    - increase confidence
    - decrease confidence
    - merge
    - split
    - contradict
    - are never deleted—only superseded/retracted
    """
    belief_id: str
    claim_id: str
    confidence: float
    state: BeliefState = BeliefState.WEAK
    support_count: int = 0
    contradiction_count: int = 0
    last_update: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    history: List[Dict[str, Any]] = field(default_factory=list)
    revision: int = 1
    superseded_by: Optional[str] = None
    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not self.belief_id:
            raise ValueError("belief_id is required")
        if not self.claim_id:
            raise ValueError("claim_id is required")

    def canonical_hash(self) -> str:
        """Deterministic hash for replay verification."""
        data = {
            "belief_id": self.belief_id,
            "claim_id": self.claim_id,
            "confidence": self.confidence,
            "state": self.state.value,
            "support_count": self.support_count,
            "contradiction_count": self.contradiction_count,
            "last_update": self.last_update,
            "history": self.history,
            "revision": self.revision,
            "superseded_by": self.superseded_by,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "belief_id": self.belief_id,
            "claim_id": self.claim_id,
            "confidence": self.confidence,
            "state": self.state.value,
            "support_count": self.support_count,
            "contradiction_count": self.contradiction_count,
            "last_update": self.last_update,
            "history": self.history,
            "revision": self.revision,
            "superseded_by": self.superseded_by,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Belief:
        data = data.copy()
        data["state"] = BeliefState(data["state"])
        return cls(**data)