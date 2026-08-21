"""
COG v0.3 — Evidence Provider Protocol

The sole external interface between CCOS and COG.

Status: FROZEN (M1)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from cog.contracts.evidence import Evidence, EvidencePackage


class EvidenceProvider(ABC):
    """
    Constitutional boundary: CCOS → COG

    COG must never:
    - inspect the ledger
    - query memory directly
    - modify CCOS state
    - replay history
    - edit projections

    All such work belongs to CCOS.
    COG is a pure epistemic consumer.
    """

    @abstractmethod
    def get_evidence(self, evidence_id: str) -> Optional[Evidence]:
        """Retrieve a single evidence by ID."""
        ...

    @abstractmethod
    def get_evidence_package(self, package_id: str) -> Optional[EvidencePackage]:
        """Retrieve an evidence package by ID."""
        ...

    @abstractmethod
    def list_evidence_packages(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[EvidencePackage]:
        """List available evidence packages."""
        ...

    @abstractmethod
    def verify_package(self, package: EvidencePackage) -> bool:
        """Verify an evidence package against CCOS governance."""
        ...

    @abstractmethod
    def get_package_version(self, package_id: str) -> str:
        """Get the schema version of an evidence package."""
        ...