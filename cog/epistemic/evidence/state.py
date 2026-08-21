"""
COG v0.3 M2 — Evidence State

Implements the Evidence dataclass and state machine per M2_EPISTEMIC_STATE_SPEC.md.

Constitutional invariants:
- CI-COG-201: Evidence cannot become Knowledge directly
- CI-COG-202: Evidence cannot become Belief without Claim intermediary
- CI-COG-210: Epistemic state is deterministically replayable
- CI-COG-211: Serialization is canonical
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import hashlib
import uuid


class ValidityState(Enum):
    """Validity state of evidence."""
    RECEIVED = "received"
    VALIDATED = "validated"
    AVAILABLE = "available"
    REJECTED = "rejected"


class EvidenceType(Enum):
    """Type of evidence."""
    OBSERVATION = "observation"
    MEASUREMENT = "measurement"
    VERIFICATION = "verification"
    EXPERIENCE = "experience"


@dataclass(frozen=True, slots=True)
class IntegrityInfo:
    """Integrity verification information."""
    content_hash: str
    algorithm: str = "sha256"
    signature: Optional[str] = None
    verified_at: Optional[str] = None
    verified_by: Optional[str] = None


@dataclass(frozen=True, slots=True)
class Evidence:
    """
    Evidence is a provenance-tracked, validated observation from CCOS.
    
    Pipeline: Observation → Evidence → Claim → Belief → Knowledge
    
    Evidence is not automatically truth — it is a governed piece of evidence.
    """
    evidence_id: str
    schema_version: str = "1.0.0"
    source: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)
    observation_ref: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    evidence_type: EvidenceType = EvidenceType.OBSERVATION
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    validity: ValidityState = ValidityState.RECEIVED
    confidence: float = 0.5
    integrity: IntegrityInfo = field(default_factory=lambda: IntegrityInfo(content_hash=""))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate invariants on creation."""
        if not self.evidence_id:
            raise ValueError("evidence_id is required")
        if not self.source:
            raise ValueError("source is required")
        if not self.observation_ref:
            raise ValueError("observation_ref is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.validity not in ValidityState:
            raise ValueError(f"invalid validity state: {self.validity}")

        # Compute content hash if not provided
        if not self.integrity.content_hash:
            object.__setattr__(self, "integrity", IntegrityInfo(
                content_hash=self._compute_content_hash()
            ))

    def _compute_content_hash(self) -> str:
        """Compute deterministic hash of evidence content."""
        data = {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "provenance": self.provenance,
            "observation_ref": self.observation_ref,
            "content": self.content,
            "evidence_type": self.evidence_type.value,
            "timestamp": self.timestamp,
            "validity": self.validity.value,
            "confidence": self.confidence,
        }
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    def canonical_hash(self) -> str:
        """Deterministic hash for replay verification."""
        return self._compute_content_hash()

    def to_json(self) -> str:
        """Canonical JSON serialization."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "evidence_id": self.evidence_id,
            "schema_version": self.schema_version,
            "source": self.source,
            "provenance": self.provenance,
            "observation_ref": self.observation_ref,
            "content": self.content,
            "evidence_type": self.evidence_type.value,
            "timestamp": self.timestamp,
            "validity": self.validity.value,
            "confidence": self.confidence,
            "integrity": {
                "content_hash": self.integrity.content_hash,
                "algorithm": self.integrity.algorithm,
                "signature": self.integrity.signature,
                "verified_at": self.integrity.verified_at,
                "verified_by": self.integrity.verified_by,
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Evidence:
        """Create from dictionary."""
        data = data.copy()
        data["evidence_type"] = EvidenceType(data.get("evidence_type", "observation"))
        data["validity"] = ValidityState(data.get("validity", "received"))
        if "integrity" in data:
            integrity_data = data["integrity"]
            data["integrity"] = IntegrityInfo(
                content_hash=integrity_data.get("content_hash", ""),
                algorithm=integrity_data.get("algorithm", "sha256"),
                signature=integrity_data.get("signature"),
                verified_at=integrity_data.get("verified_at"),
                verified_by=integrity_data.get("verified_by"),
            )
        return cls(**data)


# Valid transitions per state machine
VALID_TRANSITIONS = {
    ValidityState.RECEIVED: {ValidityState.VALIDATED, ValidityState.REJECTED},
    ValidityState.VALIDATED: {ValidityState.AVAILABLE, ValidityState.REJECTED},
    ValidityState.AVAILABLE: set(),  # Terminal state
    ValidityState.REJECTED: set(),   # Terminal state
}


def is_valid_transition(from_state: ValidityState, to_state: ValidityState) -> bool:
    """Check if a validity state transition is allowed."""
    return to_state in VALID_TRANSITIONS.get(from_state, set())


def transition_evidence(evidence: Evidence, new_validity: ValidityState) -> Evidence:
    """
    Create new Evidence with updated validity.
    
    Evidence is immutable - this creates a new instance.
    """
    if not is_valid_transition(evidence.validity, new_validity):
        raise ValueError(
            f"Invalid transition: {evidence.validity.value} → {new_validity.value}"
        )
    
    return Evidence(
        evidence_id=evidence.evidence_id,
        schema_version=evidence.schema_version,
        source=evidence.source,
        provenance=evidence.provenance,
        observation_ref=evidence.observation_ref,
        content=evidence.content,
        evidence_type=evidence.evidence_type,
        timestamp=evidence.timestamp,
        validity=new_validity,
        confidence=evidence.confidence,
        integrity=evidence.integrity,
        metadata=evidence.metadata,
    )


__all__ = [
    "ValidityState",
    "EvidenceType",
    "IntegrityInfo",
    "Evidence",
    "is_valid_transition",
    "transition_evidence",
]