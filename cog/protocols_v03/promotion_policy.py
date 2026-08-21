"""
COG v0.3 — Knowledge Promotion Policy Protocol

Promotes beliefs to knowledge. Immutable output.

Status: FROZEN (M1)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from cog.contracts.belief import Belief
from cog.contracts.knowledge import Knowledge, KnowledgeState


class PromotionPolicy(ABC):
    """
    Promotes beliefs to knowledge. Immutable output.

    Pipeline: Belief → Knowledge

    Promotion requirements:
    - sufficient confidence
    - multiple evidence sources
    - no unresolved contradiction
    - guardian approval
    - provenance complete

    Knowledge never changes. If new evidence appears, create Knowledge v2.
    """

    @abstractmethod
    def eligible(self, belief: Belief) -> bool:
        """Check if belief is eligible for promotion."""
        ...

    @abstractmethod
    def promote(self, belief: Belief, guardian_approval: bool) -> Knowledge:
        """Promote a belief to knowledge."""
        ...

    @abstractmethod
    def reject(self, belief: Belief, reason: str) -> Belief:
        """Reject promotion with reason."""
        ...

    @abstractmethod
    def audit(self, knowledge: Knowledge) -> bool:
        """Audit a knowledge object for validity."""
        ...

    @abstractmethod
    def supersede(self, knowledge: Knowledge, new_belief: Belief) -> Knowledge:
        """Create new knowledge version superseding old."""
        ...