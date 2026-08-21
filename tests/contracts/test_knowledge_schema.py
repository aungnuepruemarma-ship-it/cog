"""
COG v0.3 — Knowledge Schema Conformance Tests

Layer 0: Schema Conformance

Test: KNO-001..005
"""

import pytest
import json
from cog.contracts_v03.knowledge import Knowledge, KnowledgeState


class TestKnowledgeSchema:
    """KNO-001..005: Knowledge schema conformance."""

    def test_kno_001_only_promoted_beliefs_allowed(self):
        """KNO-001: Only promoted beliefs allowed."""
        knowledge = Knowledge(
            knowledge_id="KNO-001",
            derived_belief="BLF-001",
            supporting_evidence=["EVD-001", "EVD-002"],
            confidence=0.95,
            promotion_timestamp="2026-01-01T00:00:00Z",
        )
        assert knowledge.derived_belief == "BLF-001"
        assert knowledge.state == KnowledgeState.CREATED

    def test_kno_002_promotion_timestamp_required(self):
        """KNO-002: Promotion timestamp required."""
        with pytest.raises(ValueError):
            Knowledge(
                knowledge_id="KNO-BAD",
                derived_belief="BLF-001",
                supporting_evidence=["EVD-001"],
                confidence=0.9,
                promotion_timestamp="",
            )

    def test_kno_003_immutable(self):
        """KNO-003: Knowledge is immutable."""
        knowledge = Knowledge(
            knowledge_id="KNO-002",
            derived_belief="BLF-001",
            supporting_evidence=["EVD-001"],
            confidence=0.9,
            promotion_timestamp="2026-01-01T00:00:00Z",
        )
        with pytest.raises(AttributeError):
            knowledge.confidence = 0.99
        with pytest.raises(AttributeError):
            knowledge.knowledge_id = "OTHER"

    def test_kno_004_provenance_preserved(self):
        """KNO-004: Provenance preserved."""
        knowledge = Knowledge(
            knowledge_id="KNO-003",
            derived_belief="BLF-001",
            supporting_evidence=["EVD-001", "EVD-002", "EVD-003"],
            confidence=0.95,
            promotion_timestamp="2026-01-01T00:00:00Z",
        )
        assert knowledge.supporting_evidence == ["EVD-001", "EVD-002", "EVD-003"]

    def test_kno_005_stable_serialization(self):
        """KNO-005: Stable serialization."""
        knowledge = Knowledge(
            knowledge_id="KNO-004",
            derived_belief="BLF-001",
            supporting_evidence=["EVD-001"],
            confidence=0.9,
            promotion_timestamp="2026-01-01T00:00:00Z",
        )
        json1 = knowledge.to_json()
        json2 = knowledge.to_json()
        assert json1 == json2

    def test_knowledge_hash_deterministic(self):
        """Knowledge hash is deterministic."""
        knowledge = Knowledge(
            knowledge_id="KNO-005",
            derived_belief="BLF-001",
            supporting_evidence=["EVD-001"],
            confidence=0.9,
            promotion_timestamp="2026-01-01T00:00:00Z",
        )
        hash1 = knowledge.canonical_hash()
        hash2 = knowledge.canonical_hash()
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_knowledge_validation_required_fields(self):
        """Required fields cannot be empty."""
        with pytest.raises(ValueError):
            Knowledge(
                knowledge_id="",
                derived_belief="BLF-001",
                supporting_evidence=["EVD-001"],
                confidence=0.9,
                promotion_timestamp="2026-01-01T00:00:00Z",
            )
        with pytest.raises(ValueError):
            Knowledge(
                knowledge_id="KNO-006",
                derived_belief="",
                supporting_evidence=["EVD-001"],
                confidence=0.9,
                promotion_timestamp="2026-01-01T00:00:00Z",
            )
        with pytest.raises(ValueError):
            Knowledge(
                knowledge_id="KNO-007",
                derived_belief="BLF-001",
                supporting_evidence=[],
                confidence=0.9,
                promotion_timestamp="2026-01-01T00:00:00Z",
            )

    def test_knowledge_serialization_roundtrip(self):
        """Knowledge serializes and deserializes correctly."""
        import json
        original = Knowledge(
            knowledge_id="KNO-008",
            derived_belief="BLF-001",
            supporting_evidence=["EVD-001", "EVD-002"],
            confidence=0.95,
            promotion_timestamp="2026-01-01T00:00:00Z",
            state=KnowledgeState.ACTIVE,
            version=2,
        )
        json_str = original.to_json()
        restored = Knowledge.from_dict(json.loads(json_str))
        assert restored.knowledge_id == original.knowledge_id
        assert restored.derived_belief == original.derived_belief
        assert restored.confidence == original.confidence
        assert restored.state == original.state
        assert restored.canonical_hash() == original.canonical_hash()