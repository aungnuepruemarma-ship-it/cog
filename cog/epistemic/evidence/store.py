"""
COG v0.3 M2 — Evidence Store

Append-only evidence store with deterministic ordering.

Constitutional invariants:
- CI-COG-210: Epistemic state is deterministically replayable
- CI-COG-211: Serialization is canonical
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterator
from dataclasses import dataclass, field
import json
import threading
from cog.epistemic.evidence.state import Evidence, ValidityState, is_valid_transition, transition_evidence


@dataclass
class EvidenceStore:
    """
    Append-only evidence store.
    
    Constitutional: Deterministic ordering, no mutation of stored evidence.
    """
    _evidence: Dict[str, Evidence] = field(default_factory=dict)
    _order: List[str] = field(default_factory=list)  # Insertion order for determinism
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, evidence: Evidence) -> None:
        """Add evidence to store. Rejects duplicate IDs."""
        with self._lock:
            if evidence.evidence_id in self._evidence:
                raise ValueError(f"Evidence {evidence.evidence_id} already exists")
            self._evidence[evidence.evidence_id] = evidence
            self._order.append(evidence.evidence_id)

    def get(self, evidence_id: str) -> Optional[Evidence]:
        """Retrieve evidence by ID."""
        with self._lock:
            return self._evidence.get(evidence_id)

    def get_by_validity(self, validity: ValidityState) -> List[Evidence]:
        """Get all evidence with given validity state."""
        with self._lock:
            return [self._evidence[eid] for eid in self._order 
                   if self._evidence[eid].validity == validity]

    def update_validity(self, evidence_id: str, new_validity) -> Evidence:
        """Update evidence validity (creates new Evidence instance)."""
        with self._lock:
            evidence = self._evidence.get(evidence_id)
            if not evidence:
                raise KeyError(f"Evidence {evidence_id} not found")
            
            if not is_valid_transition(evidence.validity, new_validity):
                raise ValueError(
                    f"Invalid transition: {evidence.validity.value} → {new_validity.value}"
                )
            
            new_evidence = transition_evidence(evidence, new_validity)
            self._evidence[evidence_id] = new_evidence
            return new_evidence

    def iterator(self) -> Iterator[Evidence]:
        """Iterate in deterministic insertion order."""
        with self._lock:
            for eid in self._order:
                yield self._evidence[eid]

    def __len__(self) -> int:
        with self._lock:
            return len(self._evidence)

    def __contains__(self, evidence_id: str) -> bool:
        with self._lock:
            return evidence_id in self._evidence

    def all_evidence(self) -> List[Evidence]:
        """Get all evidence in deterministic order."""
        with self._lock:
            return [self._evidence[eid] for eid in self._order]

    def serialize(self) -> str:
        """Serialize entire store to canonical JSON."""
        with self._lock:
            data = {
                "evidence": [self._evidence[eid].to_dict() for eid in self._order],
                "order": self._order,
            }
            return json.dumps(data, sort_keys=True, separators=(",", ":"))

    @classmethod
    def deserialize(cls, json_str: str) -> EvidenceStore:
        """Deserialize store from canonical JSON."""
        import json
        data = json.loads(json_str)
        store = cls()
        with store._lock:
            for e_dict in data["evidence"]:
                evidence = Evidence.from_dict(e_dict)
                store._evidence[evidence.evidence_id] = evidence
            store._order = data["order"]
        return store


__all__ = ["EvidenceStore"]