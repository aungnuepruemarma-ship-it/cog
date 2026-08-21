"""
COG v0.3 — Belief Engine Protocol

Working memory of cognition. Mutable beliefs with confidence tracking.

Status: FROZEN (M1)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from cog.contracts.claim import Claim
from cog.contracts.belief import Belief, BeliefState


class BeliefEngine(ABC):
    """
    Working memory of cognition. Mutable beliefs.

    Pipeline: Claim → Belief → Knowledge

    Beliefs can:
    - increase confidence
    - decrease confidence
    - merge
    - split
    - contradict
    - are never deleted—only superseded/retracted
    """

    @abstractmethod
    def create_belief(self, claim: Claim) -> Belief:
        """Create a new belief from a claim."""
        ...

    @abstractmethod
    def update_confidence(self, belief_id: str, delta: float, reason: str) -> Belief:
        """Update belief confidence with reason."""
        ...

    @abstractmethod
    def merge(self, belief_a: Belief, belief_b: Belief) -> Belief:
        """Merge two beliefs into one."""
        ...

    @abstractmethod
    def split(self, belief: Belief, new_claim_id: str) -> List[Belief]:
        """Split a belief into multiple."""
        ...

    @abstractmethod
    def contradict(self, belief_id: str, contradiction: str) -> Belief:
        """Record a contradiction against a belief."""
        ...

    @abstractmethod
    def retire(self, belief_id: str, reason: str) -> Belief:
        """Retire a belief (mark as retracted)."""
        ...

    @abstractmethod
    def get_belief(self, belief_id: str) -> Optional[Belief]:
        """Retrieve a belief by ID."""
        ...