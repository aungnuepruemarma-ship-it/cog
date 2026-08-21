"""
COG v0.3 — Intent Contract

Intent is the export boundary of COG.

Constitutional invariant: CI-207 — Planner consumes reasoning only.
Constitutional invariant: CI-208 — Planner never edits knowledge.

Status: FROZEN (M1)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import hashlib


class IntentPriority(Enum):
    """Priority of an intent."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class IntentStatus(Enum):
    """Status of an intent."""
    CREATED = "created"
    QUEUED = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class Intent:
    """
    Intent is the export boundary of COG.

    Pipeline: Knowledge → Reasoning → Planning → Intent

    Planner never edits knowledge.
    Intent becomes the input to execution (outside COG).
    """
    intent_id: str
    goal: str
    constraints: Dict[str, Any]
    priority: IntentPriority = IntentPriority.NORMAL
    status: IntentStatus = IntentStatus.CREATED
    success_criteria: List[str] = field(default_factory=list)
    budget: Optional[Dict[str, Any]] = None
    deadline: Optional[str] = None
    generated_from_reasoning: List[str] = field(default_factory=list)  # reasoning_ids
    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def __post_init__(self):
        if not self.intent_id:
            raise ValueError("intent_id is required")
        if not self.goal:
            raise ValueError("goal is required")

    def canonical_hash(self) -> str:
        data = {
            "intent_id": self.intent_id,
            "goal": self.goal,
            "constraints": self.constraints,
            "priority": self.priority.value,
            "status": self.status.value,
            "success_criteria": sorted(self.success_criteria),
            "budget": self.budget,
            "deadline": self.deadline,
            "generated_from_reasoning": sorted(self.generated_from_reasoning),
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "goal": self.goal,
            "constraints": self.constraints,
            "priority": self.priority.value,
            "status": self.status.value,
            "success_criteria": self.success_criteria,
            "budget": self.budget,
            "deadline": self.deadline,
            "generated_from_reasoning": self.generated_from_reasoning,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Intent:
        data = data.copy()
        data["priority"] = IntentPriority(data["priority"])
        data["status"] = IntentStatus(data["status"])
        return cls(**data)