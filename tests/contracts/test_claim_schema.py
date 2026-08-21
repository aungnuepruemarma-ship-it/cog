"""
COG v0.3 — Claim Schema Conformance Tests

Layer 0: Schema Conformance

Test: CLM-001..005
"""

import pytest
import json
from cog.contracts_v03.claim import Claim, ClaimStatus


class TestClaimSchema:
    """CLM-001..005: Claim schema conformance."""

    def test_clm_001_required_fields(self):
        """CLM-001: Required fields exist."""
        claim = Claim(
            claim_id="CLM-001",
            claim_text="System is overheating",
            derived_from=["EVD-001", "EVD-002"],
            confidence=0.9,
        )
        assert claim.claim_id == "CLM-001"
        assert claim.claim_text == "System is overheating"
        assert claim.derived_from == ["EVD-001", "EVD-002"]
        assert claim.confidence == 0.9
        assert claim.status == ClaimStatus.NEW
        assert claim.created_at is not None
        assert claim.updated_at is not None
        assert claim.schema_version == "1.0.0"

    def test_clm_002_references_existing_evidence(self):
        """CLM-002: References existing evidence."""
        claim = Claim(
            claim_id="CLM-002",
            claim_text="Temperature anomaly detected",
            derived_from=["EVD-001"],
            confidence=0.8,
        )
        assert claim.derived_from == ["EVD-001"]
        assert len(claim.derived_from) >= 1

    def test_clm_003_confidence_bounded(self):
        """CLM-003: Confidence bounded [0, 1]."""
        with pytest.raises(ValueError):
            Claim(
                claim_id="CLM-BAD",
                claim_text="Test",
                derived_from=["EVD-001"],
                confidence=1.5,
            )
        with pytest.raises(ValueError):
            Claim(
                claim_id="CLM-BAD2",
                claim_text="Test",
                derived_from=["EVD-001"],
                confidence=-0.1,
            )

    def test_clm_004_immutable_identity(self):
        """CLM-004: Immutable identity."""
        claim = Claim(
            claim_id="CLM-003",
            claim_text="Test claim",
            derived_from=["EVD-001"],
            confidence=0.5,
        )
        with pytest.raises(AttributeError):
            claim.claim_id = "OTHER"
        with pytest.raises(AttributeError):
            claim.claim_text = "Other text"

    def test_clm_005_serialization_deterministic(self):
        """CLM-005: Serialization is deterministic."""
        claim = Claim(
            claim_id="CLM-004",
            claim_text="Deterministic test",
            derived_from=["EVD-001", "EVD-002"],
            confidence=0.75,
        )
        json1 = claim.to_json()
        json2 = claim.to_json()
        assert json1 == json2

    def test_claim_hash_deterministic(self):
        """Claim hash is deterministic."""
        claim = Claim(
            claim_id="CLM-005",
            claim_text="Hash test",
            derived_from=["EVD-001"],
            confidence=0.5,
        )
        hash1 = claim.canonical_hash()
        hash2 = claim.canonical_hash()
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_claim_validation_required_fields(self):
        """Required fields cannot be empty."""
        with pytest.raises(ValueError):
            Claim(
                claim_id="",
                claim_text="Test",
                derived_from=["EVD-001"],
                confidence=0.5,
            )
        with pytest.raises(ValueError):
            Claim(
                claim_id="CLM-006",
                claim_text="",
                derived_from=["EVD-001"],
                confidence=0.5,
            )
        with pytest.raises(ValueError):
            Claim(
                claim_id="CLM-007",
                claim_text="Test",
                derived_from=[],
                confidence=0.5,
            )

    def test_claim_serialization_roundtrip(self):
        """Claim serializes and deserializes correctly."""
        original = Claim(
            claim_id="CLM-008",
            claim_text="Roundtrip test",
            derived_from=["EVD-001", "EVD-002"],
            confidence=0.9,
            status=ClaimStatus.SUPPORTED,
            supporting_evidence=["EVD-001"],
            contradicting_evidence=["EVD-003"],
        )
        json_str = original.to_json()
        restored = Claim.from_dict(json.loads(json_str))
        assert restored.claim_id == original.claim_id
        assert restored.claim_text == original.claim_text
        assert restored.confidence == original.confidence
        assert restored.status == original.status
        assert restored.canonical_hash() == original.canonical_hash()