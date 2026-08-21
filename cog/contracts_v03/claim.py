"""
COG v0.3 — Claim Contract

Constitutional invariant: CI-203 — Claims reference evidence only.

Status: FROZEN (M1)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import hashlib


class ClaimStatus(Enum):
    """Status of a claim in the epistemic pipeline."""
    NEW = "new"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    RETRACTED = "retracted"
    PROMOTED = "promoted"


@dataclass(frozen=True, slots=True)
class Claim:
    """
    A claim is an interpretable proposition inferred from evidence.

    Pipeline: Evidence → Claim → Belief → Knowledge

    Immutable identity. Status may change but claim_id never does.
    """
    claim_id: str
    claim_text: str
    derived_from: List[str]  # evidence_ids
    confidence: float
    status: ClaimStatus = ClaimStatus.NEW
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    schema_version: str = "1.0.0"

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not self.claim_id:
            raise ValueError("claim_id is required")
        if not self.claim_text:
            raise ValueError("claim_text is required")
        if not self.derived_from:
            raise ValueError("derived_from (evidence references) cannot be empty")

    def canonical_hash(self) -> str:
        """Deterministic hash for replay verification."""
        data = {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "derived_from": sorted(self.derived_from),
            "confidence": self.confidence,
            "status": self.status.value,
            "supporting_evidence": sorted(self.supporting_evidence),
            "contradicting_evidence": sorted(self.contradicting_evidence),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "schema_version": self.schema_version,
        }
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "derived_from": self.derived_from,
            "confidence": self.confidence,
            "status": self.status.value,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Claim:
        data = data.copy()
        data["status"] = ClaimStatus(data["status"])
        return cls(**data)