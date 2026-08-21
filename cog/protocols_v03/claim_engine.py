"""
COG v0.3 — Claim Engine Protocol

Transforms evidence into structured claims.

Status: FROZEN (M1)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from cog.contracts.evidence import Evidence
from cog.contracts.claim import Claim, ClaimStatus


class ClaimEngine(ABC):
    """
    Transforms evidence into structured claims.

    Pipeline: Evidence → Claim
    """

    @abstractmethod
    def derive_claims(self, evidence_list: List[Evidence]) -> List[Claim]:
        """Derive claims from evidence."""
        ...

    @abstractmethod
    def merge_claims(self, claim_a: Claim, claim_b: Claim) -> Claim:
        """Merge two claims if they are equivalent."""
        ...

    @abstractmethod
    def contradict(self, claim: Claim, contradicting_evidence: List[str]) -> Claim:
        """Mark a claim as contradicted by new evidence."""
        ...

    @abstractmethod
    def retract(self, claim: Claim, reason: str) -> Claim:
        """Retract a claim with reason."""
        ...

    @abstractmethod
    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Retrieve a claim by ID."""
        ...