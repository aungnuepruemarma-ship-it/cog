"""
COG v0.3 — Intent Schema Conformance Tests

Layer 0: Schema Conformance
"""

import pytest
import json
from cog.contracts_v03.intent import Intent, IntentPriority, IntentStatus


class TestIntentSchema:
    """Intent schema conformance."""

    def test_intent_required_fields(self):
        """Required fields exist."""
        intent = Intent(
            intent_id="INT-001",
            goal="Fix fan failure",
            constraints={"max_temp": 80, "timeout": 300},
            priority=IntentPriority.HIGH,
        )
        assert intent.intent_id == "INT-001"
        assert intent.goal == "Fix fan failure"
        assert intent.constraints == {"max_temp": 80, "timeout": 300}
        assert intent.priority == IntentPriority.HIGH
        assert intent.status == IntentStatus.CREATED
        assert intent.success_criteria == []
        assert intent.generated_from_reasoning == []
        assert intent.schema_version == "1.0.0"
        assert intent.created_at is not None

    def test_intent_validation(self):
        """Required fields cannot be empty."""
        with pytest.raises(ValueError):
            Intent(
                intent_id="",
                goal="Test",
                constraints={},
            )
        with pytest.raises(ValueError):
            Intent(
                intent_id="INT-002",
                goal="",
                constraints={},
            )

    def test_intent_hash_deterministic(self):
        """Hash is deterministic."""
        intent = Intent(
            intent_id="INT-003",
            goal="Hash test",
            constraints={"test": True},
        )
        hash1 = intent.canonical_hash()
        hash2 = intent.canonical_hash()
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_intent_serialization_roundtrip(self):
        """Serialization roundtrip."""
        original = Intent(
            intent_id="INT-004",
            goal="Roundtrip test",
            constraints={"key": "value"},
            priority=IntentPriority.CRITICAL,
            success_criteria=["criteria1", "criteria2"],
            generated_from_reasoning=["RSN-001", "RSN-002"],
        )
        json_str = original.to_json()
        restored = Intent.from_dict(json.loads(json_str))
        assert restored.intent_id == original.intent_id
        assert restored.goal == original.goal
        assert restored.priority == original.priority
        assert restored.canonical_hash() == original.canonical_hash()


import json