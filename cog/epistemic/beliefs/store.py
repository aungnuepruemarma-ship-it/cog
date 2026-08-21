"""
COG v0.3 M2 — Belief Store and Graph

Implements the belief store with graph relationships per M2_EPISTEMIC_STATE_SPEC.md.

Constitutional invariants:
- CI-COG-204: Belief mutation creates revision history
- CI-COG-206: Contradictions cannot silently delete either belief
- CI-COG-210: Epistemic state is deterministically replayable
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterator, Set
from dataclasses import dataclass, field
import json
import threading
from cog.epistemic.beliefs.state import Belief
from cog.contracts_v03.belief import BeliefState as ContractBeliefState
from cog.epistemic.beliefs.contradiction import Contradiction


@dataclass
class BeliefStore:
    """
    Belief store with graph relationships.
    
    Constitutional: Deterministic ordering, revision chains preserved.
    """
    _beliefs: Dict[str, 'Belief'] = field(default_factory=dict)
    _order: List[str] = field(default_factory=list)  # Insertion order for determinism
    _contradictions: Dict[str, 'Contradiction'] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, belief: 'Belief') -> None:
        """Add belief to store. Rejects duplicate IDs."""
        with self._lock:
            if belief.belief_id in self._beliefs:
                raise ValueError(f"Belief {belief.belief_id} already exists")
            self._beliefs[belief.belief_id] = belief
            self._order.append(belief.belief_id)

    def get(self, belief_id: str) -> Optional['Belief']:
        """Retrieve belief by ID."""
        with self._lock:
            return self._beliefs.get(belief_id)

    def get_by_state(self, state: 'ContractBeliefState') -> List['Belief']:
        """Get all beliefs with given state."""
        with self._lock:
            return [self._beliefs[bid] for bid in self._order 
                   if self._beliefs[bid].state == state]

    def get_by_claim(self, claim_id: str) -> List['Belief']:
        """Get all beliefs derived from a claim."""
        with self._lock:
            return [self._beliefs[bid] for bid in self._order 
                   if self._beliefs[bid].claim_id == claim_id]

    def update_belief(self, belief_id: str, new_belief: 'Belief') -> None:
        """Update belief (creates new instance for revision)."""
        with self._lock:
            if belief_id not in self._beliefs:
                raise KeyError(f"Belief {belief_id} not found")
            self._beliefs[belief_id] = new_belief

    def add_contradiction(self, contradiction: 'Contradiction') -> None:
        """Record a contradiction."""
        with self._lock:
            self._contradictions[contradiction.contradiction_id] = contradiction

    def get_contradictions(self, belief_id: Optional[str] = None) -> List['Contradiction']:
        """Get all contradictions, optionally filtered by belief."""
        with self._lock:
            if belief_id is None:
                return list(self._contradictions.values())
            return [c for c in self._contradictions.values() 
                   if c.belief_a == belief_id or c.belief_b == belief_id]

    def iterator(self) -> Iterator['Belief']:
        """Iterate in deterministic insertion order."""
        with self._lock:
            for bid in self._order:
                yield self._beliefs[bid]

    def __len__(self) -> int:
        with self._lock:
            return len(self._beliefs)

    def __contains__(self, belief_id: str) -> bool:
        with self._lock:
            return belief_id in self._beliefs

    def all_beliefs(self) -> List['Belief']:
        """Get all beliefs in deterministic order."""
        with self._lock:
            return [self._beliefs[bid] for bid in self._order]

    def serialize(self) -> str:
        """Serialize entire store to canonical JSON."""
        import json
        with self._lock:
            data = {
                "beliefs": [self._beliefs[bid].to_dict() for bid in self._order],
                "order": self._order,
                "contradictions": [c.to_dict() for c in self._contradictions.values()],
            }
            return json.dumps(data, sort_keys=True, separators=(",", ":"))

    @classmethod
    def deserialize(cls, json_str: str) -> 'BeliefStore':
        """Deserialize store from canonical JSON."""
        import json
        from cog.epistemic.beliefs.state import Belief
        from cog.epistemic.beliefs.contradiction import Contradiction
        
        data = json.loads(json_str)
        store = cls()
        with store._lock:
            for b_dict in data["beliefs"]:
                belief = Belief.from_dict(b_dict)
                store._beliefs[belief.belief_id] = belief
            store._order = data["order"]
            for c_dict in data.get("contradictions", []):
                contradiction = Contradiction.from_dict(c_dict)
                store._contradictions[contradiction.contradiction_id] = contradiction
        return store


import threading