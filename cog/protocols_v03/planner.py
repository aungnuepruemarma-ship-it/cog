"""
COG v0.3 — Planner Protocol

Planner consumes reasoning. Produces Intent.

Status: FROZEN (M1)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from cog.contracts.reasoning import ReasoningResult
from cog.contracts.intent import Intent, IntentPriority, IntentStatus


class Planner(ABC):
    """
    Planning consumes reasoning. Produces Intent.

    Pipeline: Knowledge → Reasoning → Planning → Intent

    Planner never edits knowledge.
    Intent becomes the export boundary.
    """

    @abstractmethod
    def create_intent(
        self,
        reasoning_results: List[ReasoningResult],
        constraints: Dict[str, Any],
        priority: IntentPriority = IntentPriority.NORMAL
    ) -> Intent:
        """Create an intent from reasoning results."""
        ...

    @abstractmethod
    def prioritize(self, intents: List[Intent]) -> List[Intent]:
        """Prioritize intents based on constraints and criteria."""
        ...

    @abstractmethod
    def evaluate(self, intent: Intent) -> Dict[str, Any]:
        """Evaluate an intent against success criteria."""
        ...

    @abstractmethod
    def cancel(self, intent: Intent, reason: str) -> Intent:
        """Cancel an intent."""
        ...

    @abstractmethod
    def get_intent(self, intent_id: str) -> Optional[Intent]:
        """Retrieve an intent by ID."""
        ...