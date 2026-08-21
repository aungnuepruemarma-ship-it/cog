"""
COG v0.3 — Belief Schema Conformance Tests

Layer 0: Schema Conformance

Test: BLF-001..005
"""

import pytest
import json
from cog.contracts_v03.belief import Belief, BeliefState


class TestBeliefSchema:
    """BLF-001..005: Belief schema conformance."""

    def test_blf_001_valid_confidence(self):
        """BLF-001: Valid confidence."""
        belief = Belief(
            belief_id="BLF-001",
            claim_id="CLM-001",
            confidence=0.75,
        )
        assert 0.0 <= belief.confidence <= 1.0

        with pytest.raises(ValueError):
            Belief(belief_id="BLF-BAD", claim_id="CLM-001", confidence=1.5)
        with pytest.raises(ValueError):
            Belief(belief_id="BLF-BAD2", claim_id="CLM-001", confidence=-0.1)

    def test_blf_002_valid_state(self):
        """BLF-002: Valid state."""
        belief = Belief(
            belief_id="BLF-002",
            claim_id="CLM-001",
            confidence=0.5,
            state=BeliefState.WEAK,
        )
        assert belief.state == BeliefState.WEAK

        belief2 = Belief(
            belief_id="BLF-003",
            claim_id="CLM-001",
            confidence=0.8,
            state=BeliefState.STRONG,
        )
        assert belief2.state == BeliefState.STRONG

    def test_blf_003_contradiction_list_valid(self):
        """BLF-003: Contradiction list valid."""
        belief = Belief(
            belief_id="BLF-004",
            claim_id="CLM-001",
            confidence=0.5,
            contradiction_count=2,
            history=[{"action": "contradicted", "detail": "EVD-003"}],
        )
        assert belief.contradiction_count == 2
        assert len(belief.history) == 1

    def test_blf_004_revision_chain_preserved(self):
        """BLF-004: Revision chain preserved."""
        belief = Belief(
            belief_id="BLF-005",
            claim_id="CLM-001",
            confidence=0.5,
            revision=3,
            history=[
                {"revision": 1, "confidence": 0.3},
                {"revision": 2, "confidence": 0.4},
                {"revision": 3, "confidence": 0.5},
            ],
        )
        assert belief.revision == 3
        assert len(belief.history) == 3

    def test_blf_005_version_immutable(self):
        """BLF-005: Version immutable (frozen dataclass)."""
        belief = Belief(
            belief_id="BLF-006",
            claim_id="CLM-001",
            confidence=0.5,
        )
        with pytest.raises(AttributeError):
            belief.revision = 5
        with pytest.raises(AttributeError):
            belief.belief_id = "OTHER"

    def test_belief_validation_required_fields(self):
        """Required fields cannot be empty."""
        with pytest.raises(ValueError):
            Belief(belief_id="", claim_id="CLM-001", confidence=0.5)
        with pytest.raises(ValueError):
            Belief(belief_id="BLF-007", claim_id="", confidence=0.5)

    def test_belief_serialization_roundtrip(self):
        """Belief serializes and deserializes correctly."""
        import json
        original = Belief(
            belief_id="BLF-008",
            claim_id="CLM-001",
            confidence=0.9,
            state=BeliefState.STRONG,
            support_count=5,
            contradiction_count=0,
            revision=2,
        )
        json_str = original.to_json()
        restored = Belief.from_dict(json.loads(json_str))
        assert restored.belief_id == original.belief_id
        assert restored.claim_id == original.claim_id
        assert restored.confidence == original.confidence
        assert restored.state == original.state
        assert restored.canonical_hash() == original.canonical_hash()

    def test_belief_hash_deterministic(self):
        """Belief hash is deterministic."""
        belief = Belief(
            belief_id="BLF-009",
            claim_id="CLM-001",
            confidence=0.5,
        )
        hash1 = belief.canonical_hash()
        hash2 = belief.canonical_hash()
        assert hash1 == hash2
        assert len(hash1) == 64