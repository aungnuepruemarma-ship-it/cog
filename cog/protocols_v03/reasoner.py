"""
COG v0.3 — Reasoner Protocol

Reasoning consumes only knowledge. Deterministic output.

Status: FROZEN (M1)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from cog.contracts.reasoning import ReasoningContext, ReasoningResult, ReasoningMode


class Reasoner(ABC):
    """
    Reasoning consumes only knowledge. Never beliefs.

    Output: structured conclusions with complete evidence traces.

    Deterministic given identical knowledge.
    """

    @abstractmethod
    def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Perform reasoning over knowledge."""
        ...

    @abstractmethod
    def explain(self, result: ReasoningResult) -> Dict[str, Any]:
        """Explain a reasoning result."""
        ...

    @abstractmethod
    def trace(self, result: ReasoningResult) -> List[str]:
        """Trace the evidence chain for a result."""
        ...

    @abstractmethod
    def verify(self, result: ReasoningResult) -> bool:
        """Verify a reasoning result."""
        ...

    @abstractmethod
    def get_reasoning(self, reasoning_id: str) -> Optional[ReasoningResult]:
        """Retrieve a reasoning result by ID."""
        ...