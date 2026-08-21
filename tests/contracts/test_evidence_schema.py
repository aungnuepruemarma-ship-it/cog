"""
COG v0.3 — Evidence Schema Conformance Tests

Layer 0: Schema Conformance

Test: EVD-001..005
"""

import pytest
import json
from cog.contracts_v03.evidence import Evidence, EvidencePackage, VerificationState


class TestEvidenceSchema:
    """EVD-001..005: Evidence schema conformance."""

    def test_evd_001_required_fields_exist(self):
        """EVD-001: Required fields exist."""
        evidence = Evidence(
            evidence_id="EVD-001",
            source_observations=["OBS-001", "OBS-002"],
            extracted_facts={"temperature": 101, "unit": "celsius"},
            provenance_chain=[{"source": "sensor", "timestamp": "2026-01-01T00:00:00Z"}],
            confidence=0.95,
        )
        assert evidence.evidence_id == "EVD-001"
        assert evidence.source_observations == ["OBS-001", "OBS-002"]
        assert evidence.extracted_facts == {"temperature": 101, "unit": "celsius"}
        assert evidence.provenance_chain == [{"source": "sensor", "timestamp": "2026-01-01T00:00:00Z"}]
        assert evidence.confidence == 0.95
        assert evidence.verification_state == VerificationState.OBSERVED
        assert evidence.schema_version == "1.0.0"
        assert evidence.created_at is not None

    def test_evd_002_schema_version_valid(self):
        """EVD-002: Schema version valid."""
        evidence = Evidence(
            evidence_id="EVD-002",
            source_observations=["OBS-001"],
            extracted_facts={},
            provenance_chain=[{"source": "test"}],
            confidence=0.5,
            schema_version="1.0.0",
        )
        assert evidence.schema_version == "1.0.0"

    def test_evd_003_immutable_after_creation(self):
        """EVD-003: Evidence is immutable after creation."""
        evidence = Evidence(
            evidence_id="EVD-003",
            source_observations=["OBS-001"],
            extracted_facts={},
            provenance_chain=[{"source": "test"}],
            confidence=0.5,
        )
        # frozen dataclass prevents mutation
        with pytest.raises(AttributeError):
            evidence.confidence = 0.99
        with pytest.raises(AttributeError):
            evidence.evidence_id = "OTHER"

    def test_evd_004_serialization_stable(self):
        """EVD-004: Serialization is stable."""
        evidence = Evidence(
            evidence_id="EVD-004",
            source_observations=["OBS-001"],
            extracted_facts={"key": "value"},
            provenance_chain=[{"source": "test"}],
            confidence=0.5,
        )
        json1 = evidence.to_json()
        json2 = evidence.to_json()
        assert json1 == json2

    def test_evd_005_hash_deterministic(self):
        """EVD-005: Hash is deterministic."""
        evidence = Evidence(
            evidence_id="EVD-005",
            source_observations=["OBS-001"],
            extracted_facts={"key": "value"},
            provenance_chain=[{"source": "test"}],
            confidence=0.5,
        )
        hash1 = evidence.canonical_hash()
        hash2 = evidence.canonical_hash()
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex

    def test_evidence_validation_confidence_bounds(self):
        """Confidence must be in [0, 1]."""
        with pytest.raises(ValueError):
            Evidence(
                evidence_id="EVD-BAD",
                source_observations=["OBS-001"],
                extracted_facts={},
                provenance_chain=[{"source": "test"}],
                confidence=1.5,
            )
        with pytest.raises(ValueError):
            Evidence(
                evidence_id="EVD-BAD2",
                source_observations=["OBS-001"],
                extracted_facts={},
                provenance_chain=[{"source": "test"}],
                confidence=-0.1,
            )

    def test_evidence_validation_required_fields(self):
        """Required fields cannot be empty."""
        with pytest.raises(ValueError):
            Evidence(
                evidence_id="",
                source_observations=["OBS-001"],
                extracted_facts={},
                provenance_chain=[{"source": "test"}],
                confidence=0.5,
            )
        with pytest.raises(ValueError):
            Evidence(
                evidence_id="EVD-006",
                source_observations=[],
                extracted_facts={},
                provenance_chain=[{"source": "test"}],
                confidence=0.5,
            )
        with pytest.raises(ValueError):
            Evidence(
                evidence_id="EVD-007",
                source_observations=["OBS-001"],
                extracted_facts={},
                provenance_chain=[],
                confidence=0.5,
            )

    def test_evidence_package_schema(self):
        """EvidencePackage schema conformance."""
        evidence = Evidence(
            evidence_id="EVD-008",
            source_observations=["OBS-001"],
            extracted_facts={},
            provenance_chain=[{"source": "test"}],
            confidence=0.5,
        )
        package = EvidencePackage(
            package_id="PKG-001",
            evidence_list=[evidence],
            provenance={"source": "ccos"},
            verification_summary={"total": 1, "verified": 1},
        )
        assert package.package_id == "PKG-001"
        assert len(package.evidence_list) == 1
        assert package.provenance == {"source": "ccos"}
        assert package.verification_summary == {"total": 1, "verified": 1}
        assert package.schema_version == "1.0.0"

    def test_evidence_package_hash_deterministic(self):
        """EvidencePackage hash is deterministic."""
        evidence = Evidence(
            evidence_id="EVD-009",
            source_observations=["OBS-001"],
            extracted_facts={},
            provenance_chain=[{"source": "test"}],
            confidence=0.5,
        )
        package = EvidencePackage(
            package_id="PKG-002",
            evidence_list=[evidence],
            provenance={"source": "ccos"},
            verification_summary={"total": 1},
        )
        hash1 = package.canonical_hash()
        hash2 = package.canonical_hash()
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_evidence_serialization_roundtrip(self):
        """Evidence serializes and deserializes correctly."""
        original = Evidence(
            evidence_id="EVD-010",
            source_observations=["OBS-001"],
            extracted_facts={"key": "value"},
            provenance_chain=[{"source": "test"}],
            confidence=0.75,
            verification_state=VerificationState.VERIFIED,
        )
        json_str = original.to_json()
        restored = Evidence.from_dict(json.loads(json_str))
        assert restored.evidence_id == original.evidence_id
        assert restored.confidence == original.confidence
        assert restored.verification_state == original.verification_state
        assert restored.canonical_hash() == original.canonical_hash()


import json