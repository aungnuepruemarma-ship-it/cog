"""
COG v0.3 M2 — Belief State

Implements the Belief dataclass, belief graph, and revision chains per M2_EPISTEMIC_STATE_SPEC.md.

Uses BeliefState from contracts_v03 directly.

Constitutional invariants:
- CI-COG-204: Belief mutation creates revision history
- CI-COG-206: Contradictions cannot silently delete either belief
- CI-COG-207: Every belief has provenance
- CI-COG-210: Epistemic state is deterministically replayable
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import json
import hashlib
import uuid

from cog.contracts_v03.belief import BeliefState as ContractBeliefState, Belief as ContractBelief


# Use contract's BeliefState directly
BeliefState = ContractBeliefState


@dataclass(frozen=True, slots=True)
class RevisionRecord:
    """Record of a belief revision."""
    revision: int
    confidence: float
    reason: str
    timestamp: str
    actor: str


@dataclass(frozen=True, slots=True)
class BeliefRelationships:
    """Graph relationships for a belief."""
    supports: Set[str] = field(default_factory=set)      # belief_ids this belief supports
    contradicts: Set[str] = field(default_factory=set)   # belief_ids this belief contradicts
    depends_on: Set[str] = field(default_factory=set)    # belief_ids this belief depends on
    derives_from: Set[str] = field(default_factory=set)  # belief_ids this belief derives from
    superseded_by: Optional[str] = None                  # belief_id that supersedes this


@dataclass(frozen=True, slots=True)
class Belief:
    """
    Belief is a versioned epistemic state with confidence tracking.
    
    Pipeline: Claim → Belief → Knowledge
    
    Beliefs are mutable working memory - they can:
    - increase confidence
    - decrease confidence
    - merge
    - split
    - contradict
    - are never deleted—only superseded/retracted
    """
    belief_id: str
    claim_id: str
    confidence: float
    state: ContractBeliefState = ContractBeliefState.WEAK
    support_count: int = 0
    contradiction_count: int = 0
    revision: int = 1
    history: List[Dict[str, Any]] = field(default_factory=list)
    relationships: Dict[str, Any] = field(default_factory=dict)
    last_update: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    superseded_by: Optional[str] = None
    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def __post_init__(self):
        if not self.belief_id:
            raise ValueError("belief_id is required")
        if not self.claim_id:
            raise ValueError("claim_id is required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def canonical_hash(self) -> str:
        """Deterministic hash for replay verification."""
        import json, hashlib
        data = {
            "belief_id": self.belief_id,
            "claim_id": self.claim_id,
            "confidence": self.confidence,
            "state": self.state.value,
            "support_count": self.support_count,
            "contradiction_count": self.contradiction_count,
            "revision": self.revision,
            "history": self.history,
            "relationships": self.relationships,
            "last_update": self.last_update,
            "superseded_by": self.superseded_by,
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
            "belief_id": self.belief_id,
            "claim_id": self.claim_id,
            "confidence": self.confidence,
            "state": self.state.value,
            "support_count": self.support_count,
            "contradiction_count": self.contradiction_count,
            "revision": self.revision,
            "history": self.history,
            "relationships": self.relationships,
            "last_update": self.last_update,
            "superseded_by": self.superseded_by,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Belief:
        """Create from dictionary."""
        import json
        data = data.copy()
        data["state"] = ContractBeliefState(data["state"])
        return cls(**data)


# Valid state transitions for beliefs
VALID_BELIEF_STATE_TRANSITIONS = {
    ContractBeliefState.WEAK: {ContractBeliefState.MODERATE, ContractBeliefState.STRONG, ContractBeliefState.CONTRADICTED, ContractBeliefState.RETRACTED, ContractBeliefState.PROMOTED},
    ContractBeliefState.MODERATE: {ContractBeliefState.STRONG, ContractBeliefState.CONTRADICTED, ContractBeliefState.RETRACTED, ContractBeliefState.PROMOTED},
    ContractBeliefState.STRONG: {ContractBeliefState.CONTRADICTED, ContractBeliefState.RETRACTED, ContractBeliefState.PROMOTED},
    ContractBeliefState.CONTRADICTED: {ContractBeliefState.WEAK, ContractBeliefState.MODERATE, ContractBeliefState.STRONG, ContractBeliefState.RETRACTED, ContractBeliefState.PROMOTED},
    ContractBeliefState.RETRACTED: {ContractBeliefState.PROMOTED},  # Can be re-promoted
    ContractBeliefState.PROMOTED: set(),  # Terminal
}


def is_valid_belief_state_transition(from_state: ContractBeliefState, to_state: ContractBeliefState) -> bool:
    """Check if a belief state transition is allowed."""
    return to_state in VALID_BELIEF_STATE_TRANSITIONS.get(from_state, set())


def create_belief(claim_id: str, confidence: float, actor: str = "system") -> 'Belief':
    """Create a new belief from a claim."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    
    belief_id = f"BLF-{uuid.uuid4().hex[:12]}"
    return Belief(
        belief_id=belief_id,
        claim_id=claim_id,
        confidence=confidence,
        state=ContractBeliefState.WEAK,
        revision=1,
        history=[{
            "revision": 1,
            "confidence": confidence,
            "reason": "initial_creation",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "actor": actor,
        }],
    )


def update_confidence(belief: 'Belief', new_confidence: float, reason: str, actor: str = "system") -> 'Belief':
    """
    Create new belief revision with updated confidence.
    
    Belief is immutable - this creates a new instance with incremented revision.
    """
    if not 0.0 <= new_confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {new_confidence}")
    
    new_revision = belief.revision + 1
    new_history = belief.history + [{
        "revision": new_revision,
        "confidence": new_confidence,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "actor": actor,
    }]
    
    # Determine new state based on confidence
    if new_confidence >= 0.8:
        new_state = ContractBeliefState.STRONG
    elif new_confidence >= 0.6:
        new_state = ContractBeliefState.MODERATE
    elif new_confidence >= 0.4:
        new_state = ContractBeliefState.WEAK
    else:
        new_state = ContractBeliefState.CONTRADICTED
    
    return Belief(
        belief_id=belief.belief_id,
        claim_id=belief.claim_id,
        confidence=new_confidence,
        state=new_state,
        support_count=belief.support_count,
        contradiction_count=belief.contradiction_count,
        revision=belief.revision + 1,
        history=new_history,
        relationships=belief.relationships,
        last_update=datetime.utcnow().isoformat() + "Z",
        superseded_by=belief.superseded_by,
        schema_version=belief.schema_version,
        created_at=belief.created_at,
    )


def supersede_belief(belief: 'Belief', new_belief_id: str, reason: str, actor: str = "system") -> 'Belief':
    """Create new belief that supersedes this one."""
    superseded = Belief(
        belief_id=belief.belief_id,
        claim_id=belief.claim_id,
        confidence=belief.confidence,
        state=belief.state,
        support_count=belief.support_count,
        contradiction_count=belief.contradiction_count,
        revision=belief.revision,
        history=belief.history,
        relationships=belief.relationships,
        last_update=datetime.utcnow().isoformat() + "Z",
        superseded_by=new_belief_id,
        schema_version=belief.schema_version,
        created_at=belief.created_at,
    )
    return superseded


__all__ = [
    "Belief",
    "create_belief",
    "update_confidence",
    "supersede_belief",
]