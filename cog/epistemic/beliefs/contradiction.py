"""
COG v0.3 M2 — Contradiction Model

Implements first-class contradiction objects per M2_EPISTEMIC_STATE_SPEC.md.

Constitutional invariant: CI-COG-206 - Contradictions cannot silently delete either belief.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import hashlib
import uuid


class SeverityLevel(Enum):
    """Severity of a contradiction."""
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class ContradictionStatus(Enum):
    """Status of a contradiction."""
    OPEN = "open"
    RESOLVED = "resolved"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class Contradiction:
    """
    Contradiction is a first-class object representing a conflict between beliefs.
    
    Key principle: Neither belief automatically wins. M2 records the contradiction;
    M3 will define the process for resolving or updating beliefs.
    """
    contradiction_id: str
    belief_a: str
    belief_b: str
    detected_at: str
    evidence_refs: List[str] = field(default_factory=list)
    severity: SeverityLevel = SeverityLevel.MODERATE
    status: ContradictionStatus = ContradictionStatus.OPEN
    resolution_ref: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def __post_init__(self):
        if not self.contradiction_id:
            raise ValueError("contradiction_id is required")
        if not self.belief_a:
            raise ValueError("belief_a is required")
        if not self.belief_b:
            raise ValueError("belief_b is required")
        if self.belief_a == self.belief_b:
            raise ValueError("belief_a and belief_b must be different")

    def canonical_hash(self) -> str:
        """Deterministic hash for replay verification."""
        import json, hashlib
        data = {
            "contradiction_id": self.contradiction_id,
            "belief_a": self.belief_a,
            "belief_b": self.belief_b,
            "detected_at": self.detected_at,
            "evidence_refs": sorted(self.evidence_refs),
            "severity": self.severity.value,
            "status": self.status.value,
            "resolution_ref": self.resolution_ref,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    def to_json(self) -> str:
        """Canonical JSON serialization."""
        import json
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contradiction_id": self.contradiction_id,
            "belief_a": self.belief_a,
            "belief_b": self.belief_b,
            "detected_at": self.detected_at,
            "evidence_refs": self.evidence_refs,
            "severity": self.severity.value,
            "status": self.status.value,
            "resolution_ref": self.resolution_ref,
            "metadata": self.metadata,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Contradiction':
        """Create from dictionary."""
        import json
        data = data.copy()
        data["severity"] = data.get("severity", "moderate")
        if isinstance(data.get("severity"), str):
            data["severity"] = data["severity"]
        data["status"] = data.get("status", "open")
        if isinstance(data.get("status"), str):
            data["status"] = data["status"]
        return cls(**data)


def detect_contradiction(belief_a, belief_b, evidence_refs: Optional[List[str]] = None, 
                         severity: str = "moderate") -> 'Contradiction':
    """
    Create a contradiction between two beliefs.
    
    Neither belief automatically wins. The contradiction is recorded for later resolution.
    """
    if belief_a.belief_id == belief_b.belief_id:
        raise ValueError("Cannot create contradiction with same belief")
    
    contradiction_id = f"CONT-{uuid.uuid4().hex[:12]}"
    
    return Contradiction(
        contradiction_id=contradiction_id,
        belief_a=belief_a.belief_id if hasattr(belief_a, 'belief_id') else belief_a,
        belief_b=belief_b.belief_id if hasattr(belief_b, 'belief_id') else belief_b,
        detected_at=datetime.utcnow().isoformat() + "Z",
        evidence_refs=evidence_refs or [],
    )


__all__ = [
    "SeverityLevel",
    "ContradictionStatus",
    "Contradiction",
    "detect_contradiction",
]