"""
COG v0.3 — Reasoning Contract

Reasoning consumes only knowledge. Never beliefs.

Constitutional invariant: CI-206 — Reasoning consumes knowledge only.
Constitutional invariant: CI-207 — Planning cannot mutate epistemic state.

Status: FROZEN (M1)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json
import hashlib


class ReasoningMode(Enum):
    """Mode of reasoning."""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"


@dataclass(frozen=True, slots=True)
class ReasoningContext:
    """
    Context for a reasoning operation.

    Reasoning consumes only knowledge. Never beliefs.
    Output: structured conclusions with complete evidence traces.
    """
    reasoning_id: str
    goal: str
    knowledge_used: List[str]  # knowledge_ids
    assumptions: List[str]
    mode: ReasoningMode = ReasoningMode.DEDUCTIVE
    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def __post_init__(self):
        if not self.reasoning_id:
            raise ValueError("reasoning_id is required")
        if not self.goal:
            raise ValueError("goal is required")
        if not self.knowledge_used:
            raise ValueError("knowledge_used cannot be empty")

    def canonical_hash(self) -> str:
        data = {
            "reasoning_id": self.reasoning_id,
            "goal": self.goal,
            "knowledge_used": sorted(self.knowledge_used),
            "assumptions": sorted(self.assumptions),
            "mode": self.mode.value,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasoning_id": self.reasoning_id,
            "goal": self.goal,
            "knowledge_used": self.knowledge_used,
            "assumptions": self.assumptions,
            "mode": self.mode.value,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReasoningContext:
        data = data.copy()
        data["mode"] = ReasoningMode(data["mode"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ReasoningResult:
    """
    Result of a reasoning operation.

    Deterministic given identical knowledge.
    """
    reasoning_id: str
    context_id: str
    inference_steps: List[Dict[str, Any]]
    conclusion: str
    confidence: float
    evidence_trace: List[str]
    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    def canonical_hash(self) -> str:
        data = {
            "reasoning_id": self.reasoning_id,
            "context_id": self.context_id,
            "inference_steps": self.inference_steps,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "evidence_trace": sorted(self.evidence_trace),
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reasoning_id": self.reasoning_id,
            "context_id": self.context_id,
            "inference_steps": self.inference_steps,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "evidence_trace": self.evidence_trace,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReasoningResult:
        return cls(**data)