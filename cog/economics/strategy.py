"""Phase 20: Reasoning Economics.

Every strategy gets measured (accuracy, latency, tokens, cost, reliability,
energy). Cog chooses the cheapest reasoning strategy that still meets the
required quality. The selector is implemented; what M0 lacks is multiple
strategies to choose between -- Experiences already record the metrics that
will feed it (see cog/bench.py for the aggregation).

Layer 4 (FrugalGPT-style cascade) plugs in here: a ModelLadder is an ordered
list of model tiers (cheapest first). The strategist builds one candidate per
tier plus skill_replay, and the selector picks the cheapest tier whose accuracy
still meets the verification threshold. Never call the biggest model first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cog.runtime.adapter import ModelAdapter  # noqa: F401  (kept for typing tiers)


@dataclass
class StrategyMetrics:
    name: str
    accuracy: float = 0.0  # in [0, 1]
    latency_s: float = 0.0
    tokens: int = 0
    cost: float = 0.0
    reliability: float = 0.0  # in [0, 1]
    energy: float = 0.0

    def total_cost(self) -> float:
        """Single comparable expense figure; weights evolve with evidence."""
        return self.cost + self.latency_s * 0.01 + self.tokens * 1e-6 + self.energy


@dataclass
class ModelTier:
    """One rung of the FrugalGPT ladder.

    ``prior_accuracy`` is the optimistic prior used until this tier has its own
    measured verified-rate; ``cost`` is its relative expense (0 = free replay,
    1 = cheap local, 5 = frontier API). ``adapter`` is the concrete model.
    """

    name: str
    adapter: Any  # ModelAdapter
    cost: float = 1.0
    prior_accuracy: float = 0.75
    prior_latency_s: float = 0.05


class StrategySelector:
    """Pick the cheapest strategy that still meets the required quality."""

    def choose(
        self,
        strategies: list[StrategyMetrics],
        required_accuracy: float = 0.0,
        required_reliability: float = 0.0,
    ) -> StrategyMetrics | None:
        qualified = [
            s
            for s in strategies
            if s.accuracy >= required_accuracy and s.reliability >= required_reliability
        ]
        if not qualified:
            return None
        return min(qualified, key=lambda s: s.total_cost())
