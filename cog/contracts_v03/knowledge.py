"""
COG v0.3 — Knowledge Contract

Knowledge is promoted belief. Immutable after promotion.

Constitutional invariant: CI-205 — Knowledge derives only from beliefs.
Constitutional invariant: CI-209 — Knowledge is immutable.

Status: FROZEN (M1)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import hashlib


class KnowledgeState(Enum):
    """State of knowledge in the epistemic pipeline."""
    CREATED = "created"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


@dataclass(frozen=True, slots=True)
class Knowledge:
    """
    Knowledge is promoted belief. Immutable after creation.

    Pipeline: Belief → Knowledge

    Promotion requirements:
    - sufficient confidence
    - multiple evidence sources
    - no unresolved contradiction
    - guardian approval
    - provenance complete

    Knowledge never changes. If new evidence appears, create Knowledge v2.
    """
    knowledge_id: str
    derived_belief: str  # belief_id
    supporting_evidence: List[str]
    confidence: float
    promotion_timestamp: str
    state: KnowledgeState = KnowledgeState.CREATED
    version: int = 1
    superseded_by: Optional[str] = None
    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not self.knowledge_id:
            raise ValueError("knowledge_id is required")
        if not self.derived_belief:
            raise ValueError("derived_belief is required")
        if not self.supporting_evidence:
            raise ValueError("supporting_evidence cannot be empty")
        if not self.promotion_timestamp:
            raise ValueError("promotion_timestamp is required")

    def canonical_hash(self) -> str:
        """Deterministic hash for replay verification."""
        data = {
            "knowledge_id": self.knowledge_id,
            "derived_belief": self.derived_belief,
            "supporting_evidence": sorted(self.supporting_evidence),
            "confidence": self.confidence,
            "promotion_timestamp": self.promotion_timestamp,
            "state": self.state.value,
            "version": self.version,
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
            "knowledge_id": self.knowledge_id,
            "derived_belief": self.derived_belief,
            "supporting_evidence": self.supporting_evidence,
            "confidence": self.confidence,
            "promotion_timestamp": self.promotion_timestamp,
            "state": self.state.value,
            "version": self.version,
            "superseded_by": self.superseded_by,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Knowledge:
        data = data.copy()
        data["state"] = KnowledgeState(data["state"])
        return cls(**data)