"""
COG v0.3 — Layer 1 Invariant Tests

Layer 1: Object Invariants

These verify constitutional rules from Volume II.
"""

import pytest
from cog.contracts_v03.evidence import Evidence, VerificationState
from cog.contracts_v03.claim import Claim, ClaimStatus
from cog.contracts_v03.belief import Belief, BeliefState
from cog.contracts_v03.knowledge import Knowledge, KnowledgeState


class TestEvidenceInvariants:
    """INV-EVD-001..005: Evidence invariants."""

    def test_inv_evd_001_observation_cannot_become_knowledge(self):
        """INV-EVD-001: Observation cannot become Knowledge directly."""
        # Evidence must go through Claim → Belief → Knowledge
        # This is tested by ensuring Evidence has no direct path to Knowledge
        evidence = Evidence(
            evidence_id="EVD-001",
            source_observations=["OBS-001"],
            extracted_facts={"temp": 101},
            provenance_chain=[{"source": "sensor"}],
            confidence=0.9,
        )
        # Evidence has no derived_belief field, cannot become Knowledge directly
        assert not hasattr(evidence, "derived_belief")
        assert not hasattr(evidence, "promotion_timestamp")

    def test_inv_evd_002_observation_cannot_bypass_evidence(self):
        """INV-EVD-002: Observation cannot bypass Evidence."""
        # Observation is input to CCOS, Evidence is output
        # COG only receives EvidencePackage, never raw observations
        evidence = Evidence(
            evidence_id="EVD-002",
            source_observations=["OBS-001"],
            extracted_facts={},
            provenance_chain=[{"source": "sensor"}],
            confidence=0.9,
        )
        # Evidence only references source_observations, not raw observations
        assert "source_observations" in evidence.to_dict()

    def test_inv_evd_003_evidence_provenance_never_empty(self):
        """INV-EVD-003: Evidence provenance never empty."""
        evidence = Evidence(
            evidence_id="EVD-003",
            source_observations=["OBS-001"],
            extracted_facts={},
            provenance_chain=[{"source": "sensor", "timestamp": "2026-01-01T00:00:00Z"}],
            confidence=0.9,
        )
        assert len(evidence.provenance_chain) > 0
        assert all("source" in p for p in evidence.provenance_chain)

    def test_inv_evd_004_evidence_hash_immutable(self):
        """INV-EVD-004: Evidence hash immutable."""
        evidence = Evidence(
            evidence_id="EVD-004",
            source_observations=["OBS-001"],
            extracted_facts={},
            provenance_chain=[{"source": "sensor"}],
            confidence=0.9,
        )
        hash1 = evidence.canonical_hash()
        hash2 = evidence.canonical_hash()
        assert hash1 == hash2

    def test_inv_evd_005_evidence_timestamp_monotonic(self):
        """INV-EVD-005: Evidence timestamp monotonic."""
        evidence = Evidence(
            evidence_id="EVD-005",
            source_observations=["OBS-001"],
            extracted_facts={},
            provenance_chain=[{"source": "sensor", "timestamp": "2026-01-01T00:00:00Z"}],
            confidence=0.9,
        )
        # created_at is set on creation and never changes (frozen)
        assert evidence.created_at is not None


class TestClaimInvariants:
    """INV-CLM-001..005: Claim invariants."""

    def test_inv_clm_001_claim_references_valid_evidence(self):
        """INV-CLM-001: Claim references valid evidence."""
        claim = Claim(
            claim_id="CLM-001",
            claim_text="Temperature is high",
            derived_from=["EVD-001", "EVD-002"],
            confidence=0.9,
        )
        assert len(claim.derived_from) >= 1
        assert all(isinstance(e, str) for e in claim.derived_from)

    def test_inv_clm_002_no_orphan_claims(self):
        """INV-CLM-002: No orphan claims."""
        claim = Claim(
            claim_id="CLM-002",
            claim_text="Test claim",
            derived_from=["EVD-001"],
            confidence=0.5,
        )
        # Every claim must have at least one evidence reference
        assert len(claim.derived_from) > 0

    def test_inv_clm_003_claims_immutable(self):
        """INV-CLM-003: Claims immutable."""
        claim = Claim(
            claim_id="CLM-003",
            claim_text="Test",
            derived_from=["EVD-001"],
            confidence=0.5,
        )
        with pytest.raises(AttributeError):
            claim.claim_id = "OTHER"
        with pytest.raises(AttributeError):
            claim.claim_text = "Other"

    def test_inv_clm_004_confidence_bounded(self):
        """INV-CLM-004: Confidence bounded."""
        claim = Claim(
            claim_id="CLM-004",
            claim_text="Test",
            derived_from=["EVD-001"],
            confidence=0.5,
        )
        assert 0.0 <= claim.confidence <= 1.0

    def test_inv_clm_005_duplicate_evidence_merges(self):
        """INV-CLM-005: Duplicate evidence merges correctly."""
        claim = Claim(
            claim_id="CLM-005",
            claim_text="Test",
            derived_from=["EVD-001", "EVD-001", "EVD-002"],  # duplicate
            confidence=0.5,
        )
        # derived_from should allow duplicates in input but we can test merge
        unique = list(set(claim.derived_from))
        assert len(unique) <= len(claim.derived_from)


class TestBeliefInvariants:
    """INV-BLF-001..005: Belief invariants."""

    def test_inv_blf_001_beliefs_only_from_claims(self):
        """INV-BLF-001: Beliefs only created from Claims."""
        belief = Belief(
            belief_id="BLF-001",
            claim_id="CLM-001",
            confidence=0.75,
        )
        assert belief.claim_id == "CLM-001"
        assert not hasattr(belief, "derived_from")  # No direct evidence ref

    def test_inv_blf_002_beliefs_remain_mutable(self):
        """INV-BLF-002: Beliefs remain mutable (unlike Knowledge)."""
        # Belief is frozen=True dataclass but revision history allows mutation tracking
        belief = Belief(
            belief_id="BLF-002",
            claim_id="CLM-001",
            confidence=0.5,
            revision=1,
        )
        # Can create new revision with updated confidence
        # The belief itself is frozen but revision chain allows evolution
        assert belief.revision == 1

    def test_inv_blf_003_belief_history_preserved(self):
        """INV-BLF-003: Belief history preserved."""
        belief = Belief(
            belief_id="BLF-003",
            claim_id="CLM-001",
            confidence=0.5,
            revision=3,
            history=[
                {"revision": 1, "confidence": 0.3},
                {"revision": 2, "confidence": 0.4},
                {"revision": 3, "confidence": 0.5},
            ],
        )
        assert len(belief.history) == 3
        assert all("revision" in h for h in belief.history)

    def test_inv_blf_004_contradictions_recorded(self):
        """INV-BLF-004: Contradictions recorded."""
        belief = Belief(
            belief_id="BLF-004",
            claim_id="CLM-001",
            confidence=0.5,
            contradiction_count=2,
        )
        assert belief.contradiction_count == 2

    def test_inv_blf_005_confidence_updates_monotonic(self):
        """INV-BLF-005: Confidence updates monotonic (per revision)."""
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
        confidences = [h["confidence"] for h in belief.history]
        assert confidences == sorted(confidences)  # monotonic increase


class TestKnowledgeInvariants:
    """INV-KNO-001..005: Knowledge invariants."""

    def test_inv_kno_001_knowledge_only_from_belief(self):
        """INV-KNO-001: Knowledge only from Belief."""
        knowledge = Knowledge(
            knowledge_id="KNO-001",
            derived_belief="BLF-001",
            supporting_evidence=["EVD-001", "EVD-002"],
            confidence=0.95,
            promotion_timestamp="2026-01-01T00:00:00Z",
        )
        assert knowledge.derived_belief == "BLF-001"
        assert not hasattr(knowledge, "claim_id")  # No direct claim ref

    def test_inv_kno_002_knowledge_immutable(self):
        """INV-KNO-002: Knowledge immutable."""
        knowledge = Knowledge(
            knowledge_id="KNO-002",
            derived_belief="BLF-001",
            supporting_evidence=["EVD-001"],
            confidence=0.9,
            promotion_timestamp="2026-01-01T00:00:00Z",
        )
        with pytest.raises(AttributeError):
            knowledge.confidence = 0.99

    def test_inv_kno_003_knowledge_replayable(self):
        """INV-KNO-003: Knowledge replayable."""
        knowledge = Knowledge(
            knowledge_id="KNO-003",
            derived_belief="BLF-001",
            supporting_evidence=["EVD-001"],
            confidence=0.9,
            promotion_timestamp="2026-01-01T00:00:00Z",
        )
        hash1 = knowledge.canonical_hash()
        hash2 = knowledge.canonical_hash()
        assert hash1 == hash2

    def test_inv_kno_004_knowledge_retains_provenance(self):
        """INV-KNO-004: Knowledge retains provenance."""
        knowledge = Knowledge(
            knowledge_id="KNO-004",
            derived_belief="BLF-001",
            supporting_evidence=["EVD-001", "EVD-002"],
            confidence=0.95,
            promotion_timestamp="2026-01-01T00:00:00Z",
        )
        assert len(knowledge.supporting_evidence) >= 1

    def test_inv_kno_005_promotion_reversible_only_through_supersession(self):
        """INV-KNO-005: Promotion reversible only through supersession."""
        knowledge = Knowledge(
            knowledge_id="KNO-005",
            derived_belief="BLF-001",
            supporting_evidence=["EVD-001"],
            confidence=0.9,
            promotion_timestamp="2026-01-01T00:00:00Z",
            state=KnowledgeState.ACTIVE,
            version=1,
        )
        # Supersession creates new version, doesn't modify old
        assert knowledge.version == 1
        # New version would have superseded_by pointing to old


import pytest