"""
COG v0.3 M2 — Evidence State Tests

Layer 2: Engine Conformance — Evidence state
"""

import pytest
import json
from cog.epistemic.evidence.state import (
    Evidence, ValidityState, EvidenceType, IntegrityInfo,
    is_valid_transition, transition_evidence
)
from cog.epistemic.evidence.store import EvidenceStore


class TestEvidenceState:
    """Evidence dataclass and state machine tests."""

    def test_evidence_creation_minimal(self):
        """Create evidence with minimal required fields."""
        evidence = Evidence(
            evidence_id="EVD-001",
            source="ccos.observation",
            observation_ref="OBS-001",
            content={"temperature": 101},
            evidence_type=EvidenceType.OBSERVATION,
            confidence=0.9,
        )
        assert evidence.evidence_id == "EVD-001"
        assert evidence.source == "ccos.observation"
        assert evidence.observation_ref == "OBS-001"
        assert evidence.content == {"temperature": 101}
        assert evidence.evidence_type == EvidenceType.OBSERVATION
        assert evidence.confidence == 0.9
        assert evidence.validity == ValidityState.RECEIVED
        assert evidence.schema_version == "1.0.0"

    def test_evidence_auto_hash(self):
        """Evidence computes content hash automatically."""
        evidence = Evidence(
            evidence_id="EVD-002",
            source="ccos.observation",
            observation_ref="OBS-001",
            content={"key": "value"},
            confidence=0.5,
        )
        assert evidence.integrity.content_hash
        assert len(evidence.integrity.content_hash) == 64

    def test_evidence_immutable(self):
        """Evidence is immutable (frozen dataclass)."""
        evidence = Evidence(
            evidence_id="EVD-003",
            source="ccos.observation",
            observation_ref="OBS-001",
            content={},
            confidence=0.5,
        )
        with pytest.raises(AttributeError):
            evidence.confidence = 0.99
        with pytest.raises(AttributeError):
            evidence.evidence_id = "OTHER"

    def test_evidence_confidence_bounds(self):
        """Confidence must be in [0, 1]."""
        with pytest.raises(ValueError):
            Evidence(
                evidence_id="EVD-BAD",
                source="ccos.observation",
                observation_ref="OBS-001",
                content={},
                confidence=1.5,
            )
        with pytest.raises(ValueError):
            Evidence(
                evidence_id="EVD-BAD2",
                source="ccos.observation",
                observation_ref="OBS-001",
                content={},
                confidence=-0.1,
            )

    def test_evidence_required_fields(self):
        """Required fields cannot be empty."""
        with pytest.raises(ValueError):
            Evidence(evidence_id="", source="ccos", observation_ref="OBS", content={}, confidence=0.5)
        with pytest.raises(ValueError):
            Evidence(evidence_id="EVD", source="", observation_ref="OBS", content={}, confidence=0.5)
        with pytest.raises(ValueError):
            Evidence(evidence_id="EVD", source="ccos", observation_ref="", content={}, confidence=0.5)


class TestEvidenceValidityTransitions:
    """Test evidence state machine transitions."""

    def test_received_to_validated(self):
        """RECEIVED → VALIDATED is valid."""
        assert is_valid_transition(ValidityState.RECEIVED, ValidityState.VALIDATED)

    def test_received_to_rejected(self):
        """RECEIVED → REJECTED is valid."""
        assert is_valid_transition(ValidityState.RECEIVED, ValidityState.REJECTED)

    def test_validated_to_available(self):
        """VALIDATED → AVAILABLE is valid."""
        assert is_valid_transition(ValidityState.VALIDATED, ValidityState.AVAILABLE)

    def test_validated_to_rejected(self):
        """VALIDATED → REJECTED is valid."""
        assert is_valid_transition(ValidityState.VALIDATED, ValidityState.REJECTED)

    def test_available_terminal(self):
        """AVAILABLE has no outgoing transitions."""
        assert not is_valid_transition(ValidityState.AVAILABLE, ValidityState.RECEIVED)
        assert not is_valid_transition(ValidityState.AVAILABLE, ValidityState.VALIDATED)

    def test_rejected_terminal(self):
        """REJECTED has no outgoing transitions."""
        assert not is_valid_transition(ValidityState.REJECTED, ValidityState.RECEIVED)

    def test_transition_creates_new_evidence(self):
        """Transition creates new Evidence instance (immutability)."""
        evidence = Evidence(
            evidence_id="EVD-001",
            source="ccos.observation",
            observation_ref="OBS-001",
            content={},
            confidence=0.5,
        )
        new_evidence = transition_evidence(evidence, ValidityState.VALIDATED)
        
        assert new_evidence is not evidence
        assert new_evidence.validity == ValidityState.VALIDATED
        assert new_evidence.evidence_id == "EVD-001"
        assert new_evidence.source == evidence.source
        assert new_evidence.content == evidence.content

    def test_invalid_transition_raises(self):
        """Invalid transitions raise ValueError."""
        evidence = Evidence(
            evidence_id="EVD-001",
            source="ccos.observation",
            observation_ref="OBS-001",
            content={},
            confidence=0.5,
        )
        # Can't go from RECEIVED directly to AVAILABLE
        with pytest.raises(ValueError):
            transition_evidence(evidence, ValidityState.AVAILABLE)
        # Can't go backwards
        evidence = Evidence(
            evidence_id="EVD-002",
            source="ccos",
            observation_ref="OBS-001",
            content={},
            confidence=0.5,
            validity=ValidityState.VALIDATED,
        )
        with pytest.raises(ValueError):
            transition_evidence(evidence, ValidityState.RECEIVED)


class TestEvidenceStore:
    """EvidenceStore append-only store tests."""

    def test_store_add_and_get(self):
        """Add and retrieve evidence."""
        store = EvidenceStore()
        evidence = Evidence(
            evidence_id="EVD-001",
            source="ccos.observation",
            observation_ref="OBS-001",
            content={"temp": 101},
            confidence=0.9,
        )
        store.add(evidence)
        retrieved = store.get("EVD-001")
        assert retrieved is not None
        assert retrieved.evidence_id == "EVD-001"
        assert retrieved.content == {"temp": 101}

    def test_store_rejects_duplicate(self):
        """Store rejects duplicate evidence IDs."""
        store = EvidenceStore()
        evidence = Evidence(
            evidence_id="EVD-001",
            source="ccos.observation",
            observation_ref="OBS-001",
            content={},
            confidence=0.5,
        )
        store.add(evidence)
        with pytest.raises(ValueError):
            store.add(evidence)

    def test_store_get_by_validity(self):
        """Get evidence by validity state."""
        store = EvidenceStore()
        e1 = Evidence(evidence_id="E1", source="s", observation_ref="o1", content={}, confidence=0.5)
        e2 = Evidence(evidence_id="E2", source="s", observation_ref="o2", content={}, confidence=0.5,
                      validity=ValidityState.VALIDATED)
        store.add(e1)
        store.add(e2)
        received = store.get_by_validity(ValidityState.RECEIVED)
        validated = store.get_by_validity(ValidityState.VALIDATED)
        assert len(received) == 1
        assert len(validated) == 1

    def test_store_update_validity(self):
        """Update validity creates new evidence."""
        store = EvidenceStore()
        evidence = Evidence(
            evidence_id="EVD-001",
            source="ccos",
            observation_ref="OBS",
            content={},
            confidence=0.5,
        )
        store.add(evidence)
        new_evidence = store.update_validity("EVD-001", ValidityState.VALIDATED)
        assert new_evidence.validity == ValidityState.VALIDATED
        assert store.get("EVD-001").validity == ValidityState.VALIDATED

    def test_store_invalid_transition_rejected(self):
        """Store rejects invalid transitions."""
        store = EvidenceStore()
        evidence = Evidence(
            evidence_id="EVD-001",
            source="ccos",
            observation_ref="OBS",
            content={},
            confidence=0.5,
            validity=ValidityState.RECEIVED,
        )
        store.add(evidence)
        with pytest.raises(ValueError):
            store.update_validity("EVD-001", ValidityState.AVAILABLE)  # Can't skip VALIDATED

    def test_store_deterministic_order(self):
        """Store maintains insertion order."""
        store = EvidenceStore()
        for i in range(5):
            evidence = Evidence(
                evidence_id=f"EVD-{i}",
                source="ccos",
                observation_ref=f"OBS-{i}",
                content={},
                confidence=0.5,
            )
            store.add(evidence)
        order = [e.evidence_id for e in store.iterator()]
        assert order == ["EVD-0", "EVD-1", "EVD-2", "EVD-3", "EVD-4"]

    def test_store_serialization_roundtrip(self):
        """Store serializes and deserializes correctly."""
        import json
        store = EvidenceStore()
        for i in range(3):
            evidence = Evidence(
                evidence_id=f"EVD-{i}",
                source="ccos",
                observation_ref=f"OBS-{i}",
                content={"index": i},
                confidence=0.5,
            )
            store.add(evidence)
        json_str = store.serialize()
        restored = EvidenceStore.deserialize(json_str)
        assert len(restored) == 3
        for i in range(3):
            assert f"EVD-{i}" in restored
            assert restored.get(f"EVD-{i}").content == {"index": i}

    def test_evidence_integrity_hash(self):
        """Evidence computes integrity hash."""
        evidence = Evidence(
            evidence_id="EVD-001",
            source="ccos",
            observation_ref="OBS",
            content={"key": "value"},
            confidence=0.5,
        )
        assert evidence.integrity.content_hash
        assert len(evidence.integrity.content_hash) == 64

    def test_evidence_canonical_hash_deterministic(self):
        """Evidence canonical hash is deterministic."""
        evidence = Evidence(
            evidence_id="EVD-001",
            source="ccos",
            observation_ref="OBS",
            content={"key": "value"},
            confidence=0.5,
        )
        hash1 = evidence.canonical_hash()
        hash2 = evidence.canonical_hash()
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_evidence_serialization_roundtrip(self):
        """Evidence serializes and deserializes correctly."""
        import json
        original = Evidence(
            evidence_id="EVD-001",
            source="ccos",
            observation_ref="OBS",
            content={"temp": 101},
            confidence=0.9,
        )
        json_str = original.to_json()
        restored = Evidence.from_dict(json.loads(json_str))
        assert restored.evidence_id == original.evidence_id
        assert restored.content == original.content
        assert restored.canonical_hash() == original.canonical_hash()