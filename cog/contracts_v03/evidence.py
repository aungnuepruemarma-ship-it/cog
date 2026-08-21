"""
COG v0.3 — Evidence Contract

Constitutional invariant: CI-201 — Evidence originates outside COG.
Constitutional invariant: CI-202 — Evidence never mutates.

Status: FROZEN (M1)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import hashlib


class VerificationState(Enum):
    """Verification state of evidence."""
    OBSERVED = "observed"
    VALIDATED = "validated"
    VERIFIED = "verified"
    ARCHIVED = "archived"


@dataclass(frozen=True, slots=True)
class Evidence:
    """
    Evidence is validated observation. Immutable after creation.

    Pipeline: Observation → Evidence → Claim → Belief → Knowledge
    """
    evidence_id: str
    source_observations: List[str]
    extracted_facts: Dict[str, Any]
    provenance_chain: List[Dict[str, Any]]
    confidence: float
    supporting_artifacts: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    verification_state: VerificationState = VerificationState.OBSERVED
    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def __post_init__(self):
        """Validate invariants on creation."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if not self.evidence_id:
            raise ValueError("evidence_id is required")
        if not self.source_observations:
            raise ValueError("source_observations cannot be empty")
        if not self.provenance_chain:
            raise ValueError("provenance_chain cannot be empty")

    def canonical_hash(self) -> str:
        """Deterministic hash for replay verification."""
        # Sort keys for canonical ordering
        data = {
            "evidence_id": self.evidence_id,
            "source_observations": sorted(self.source_observations),
            "extracted_facts": self.extracted_facts,
            "provenance_chain": self.provenance_chain,
            "confidence": self.confidence,
            "supporting_artifacts": sorted(self.supporting_artifacts),
            "contradictions": sorted(self.contradictions),
            "verification_state": self.verification_state.value,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    def to_json(self) -> str:
        """Canonical JSON serialization."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "evidence_id": self.evidence_id,
            "source_observations": self.source_observations,
            "extracted_facts": self.extracted_facts,
            "provenance_chain": self.provenance_chain,
            "confidence": self.confidence,
            "supporting_artifacts": self.supporting_artifacts,
            "contradictions": self.contradictions,
            "verification_state": self.verification_state.value,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Evidence:
        """Create from dictionary."""
        data = data.copy()
        data["verification_state"] = VerificationState(data["verification_state"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class EvidencePackage:
    """
    The sole external interface for COG.

    Contains:
    - package_id
    - schema_version
    - timestamp
    - provenance
    - Evidence[]
    - verification summary
    - guardian signature
    """
    package_id: str
    evidence_list: List[Evidence]
    provenance: Dict[str, Any]
    verification_summary: Dict[str, Any]
    guardian_signature: Optional[str] = None
    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def __post_init__(self):
        if not self.package_id:
            raise ValueError("package_id is required")
        if not self.evidence_list:
            raise ValueError("evidence_list cannot be empty")

    def canonical_hash(self) -> str:
        """Deterministic hash for replay verification."""
        data = {
            "package_id": self.package_id,
            "evidence_hashes": sorted(e.canonical_hash() for e in self.evidence_list),
            "provenance": self.provenance,
            "verification_summary": self.verification_summary,
            "guardian_signature": self.guardian_signature,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "package_id": self.package_id,
            "evidence_list": [e.to_dict() for e in self.evidence_list],
            "provenance": self.provenance,
            "verification_summary": self.verification_summary,
            "guardian_signature": self.guardian_signature,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvidencePackage:
        data = data.copy()
        data["evidence_list"] = [Evidence.from_dict(e) for e in data["evidence_list"]]
        return cls(**data)