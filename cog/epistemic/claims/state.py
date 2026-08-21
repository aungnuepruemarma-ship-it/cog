"""
COG v0.3 M2 — Claim State

Implements the Claim dataclass and state machine per M2_EPISTEMIC_STATE_SPEC.md.

Uses ClaimStatus from contracts_v03 directly.

Constitutional invariants:
- CI-COG-202: Evidence cannot become Belief without Claim intermediary
- CI-COG-203: Claim cannot become Knowledge directly
- CI-COG-210: Epistemic state is deterministically replayable
"""

from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import hashlib

from cog.contracts_v03.claim import ClaimStatus, Claim as ContractClaim


# Valid transitions per state machine
VALID_CLAIM_TRANSITIONS = {
    ClaimStatus.PROPOSED: {ClaimStatus.SUPPORTED, ClaimStatus.CONTESTED, ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED},
    ClaimStatus.SUPPORTED: {ClaimStatus.CONTESTED, ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED},
    ClaimStatus.CONTESTED: {ClaimStatus.SUPPORTED, ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED},
    ClaimStatus.REJECTED: {ClaimStatus.PROPOSED, ClaimStatus.SUPERSEDED},  # Can be re-proposed
    ClaimStatus.SUPERSEDED: set(),  # Terminal
}


def is_valid_claim_transition(from_status: ClaimStatus, to_status: ClaimStatus) -> bool:
    """Check if a claim status transition is allowed."""
    return to_status in VALID_CLAIM_TRANSITIONS.get(from_status, set())


def transition_claim(claim, new_status: ClaimStatus, reason: str = ""):
    """
    Create new Claim with updated status.
    
    Claim is immutable - this creates a new instance.
    """
    if not is_valid_claim_transition(claim.status, new_status):
        raise ValueError(
            f"Invalid claim transition: {claim.status.value} → {new_status.value}"
        )
    
    # Create new claim with updated status
    return ContractClaim(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        derived_from=claim.derived_from,
        confidence=claim.confidence,
        status=new_status,
        supporting_evidence=claim.supporting_evidence,
        contradicting_evidence=claim.contradicting_evidence,
        created_at=claim.created_at,
        updated_at=datetime.utcnow().isoformat() + "Z",
        schema_version=claim.schema_version,
    )


__all__ = [
    "ClaimStatus",
    "is_valid_claim_transition",
    "transition_claim",
    "VALID_CLAIM_TRANSITIONS",
]