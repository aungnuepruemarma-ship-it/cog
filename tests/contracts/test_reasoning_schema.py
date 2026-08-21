"""
COG v0.3 — Reasoning Schema Conformance Tests

Layer 0: Schema Conformance
"""

import pytest
import json
from cog.contracts_v03.reasoning import ReasoningContext, ReasoningResult, ReasoningMode


class TestReasoningSchema:
    """Reasoning schema conformance."""

    def test_reasoning_context_required_fields(self):
        """Required fields exist."""
        ctx = ReasoningContext(
            reasoning_id="RSN-001",
            goal="Determine root cause",
            knowledge_used=["KNO-001", "KNO-002"],
            assumptions=["System is stable", "No external interference"],
            mode=ReasoningMode.DEDUCTIVE,
        )
        assert ctx.reasoning_id == "RSN-001"
        assert ctx.goal == "Determine root cause"
        assert ctx.knowledge_used == ["KNO-001", "KNO-002"]
        assert ctx.assumptions == ["System is stable", "No external interference"]
        assert ctx.mode == ReasoningMode.DEDUCTIVE
        assert ctx.schema_version == "1.0.0"
        assert ctx.created_at is not None

    def test_reasoning_context_validation(self):
        """Required fields cannot be empty."""
        with pytest.raises(ValueError):
            ReasoningContext(
                reasoning_id="",
                goal="Test",
                knowledge_used=["KNO-001"],
                assumptions=["test"],
            )
        with pytest.raises(ValueError):
            ReasoningContext(
                reasoning_id="RSN-002",
                goal="",
                knowledge_used=["KNO-001"],
                assumptions=["test"],
            )
        with pytest.raises(ValueError):
            ReasoningContext(
                reasoning_id="RSN-003",
                goal="Test",
                knowledge_used=[],
                assumptions=["test"],
            )

    def test_reasoning_context_hash_deterministic(self):
        """Hash is deterministic."""
        ctx = ReasoningContext(
            reasoning_id="RSN-004",
            goal="Hash test",
            knowledge_used=["KNO-001"],
            assumptions=["System is stable"],
            mode=ReasoningMode.INDUCTIVE,
        )
        hash1 = ctx.canonical_hash()
        hash2 = ctx.canonical_hash()
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_reasoning_context_serialization_roundtrip(self):
        """Serialization roundtrip."""
        original = ReasoningContext(
            reasoning_id="RSN-005",
            goal="Roundtrip test",
            knowledge_used=["KNO-001", "KNO-002"],
            assumptions=["A1", "A2"],
            mode=ReasoningMode.ABDUCTIVE,
        )
        json_str = original.to_json()
        restored = ReasoningContext.from_dict(json.loads(json_str))
        assert restored.reasoning_id == original.reasoning_id
        assert restored.goal == original.goal
        assert restored.mode == original.mode
        assert restored.canonical_hash() == original.canonical_hash()

    def test_reasoning_result_required_fields(self):
        """Required fields exist."""
        result = ReasoningResult(
            reasoning_id="RSN-006",
            context_id="RSN-001",
            inference_steps=[{"step": 1, "rule": "modus ponens"}],
            conclusion="System overheating caused by fan failure",
            confidence=0.92,
            evidence_trace=["EVD-001", "EVD-002"],
        )
        assert result.reasoning_id == "RSN-006"
        assert result.context_id == "RSN-001"
        assert len(result.inference_steps) == 1
        assert result.conclusion == "System overheating caused by fan failure"
        assert result.confidence == 0.92
        assert result.evidence_trace == ["EVD-001", "EVD-002"]
        assert result.schema_version == "1.0.0"

    def test_reasoning_result_confidence_bounds(self):
        """Confidence bounded."""
        with pytest.raises(ValueError):
            ReasoningResult(
                reasoning_id="RSN-BAD",
                context_id="RSN-001",
                inference_steps=[],
                conclusion="Test",
                confidence=1.5,
                evidence_trace=[],
            )

    def test_reasoning_result_hash_deterministic(self):
        """Hash is deterministic."""
        result = ReasoningResult(
            reasoning_id="RSN-007",
            context_id="RSN-001",
            inference_steps=[],
            conclusion="Test",
            confidence=0.5,
            evidence_trace=[],
        )
        hash1 = result.canonical_hash()
        hash2 = result.canonical_hash()
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_reasoning_result_serialization_roundtrip(self):
        """Serialization roundtrip."""
        original = ReasoningResult(
            reasoning_id="RSN-008",
            context_id="RSN-001",
            inference_steps=[{"step": 1}],
            conclusion="Roundtrip test",
            confidence=0.85,
            evidence_trace=["EVD-001"],
        )
        json_str = original.to_json()
        restored = ReasoningResult.from_dict(json.loads(json_str))
        assert restored.reasoning_id == original.reasoning_id
        assert restored.conclusion == original.conclusion
        assert restored.confidence == original.confidence
        assert restored.canonical_hash() == original.canonical_hash()


import json