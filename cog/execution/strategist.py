"""Phase 20 wired into the loop: the PlanningStrategist.

Before every planning step, the strategist asks: is there a compiled skill
whose goal template matches this task? If so, two strategies compete —

- ``skill_replay``: instantiate the induced program, zero model calls;
- ``model_plan``:   ask the model, cost 1 call.

The StrategySelector picks the cheapest strategy that still meets the
required quality (the verification threshold). Skill accuracy is the
skill's *evolved* confidence: verified replays raise it, failed replays
halve it (and the Evolution Engine retires it once it drops out of
contention). Model accuracy is its measured verified-rate, persisted in
the ConceptStore.
"""

from __future__ import annotations

from cog.economics.strategy import ModelTier, StrategyMetrics, StrategySelector
from cog.execution.planner import Plan, Planner
from cog.learning.skill_compiler import instantiate_plan, match_skill
from cog.memory.base import MemoryRecord
from cog.memory.router import MemoryRouter
from cog.workspace.workspace import TaskWorkspace

STATS_RECORD_ID = "concept_strategy_stats"
_TIER_ACCURACY_PREFIX = "model_tier:"
_DEFAULT_MODEL_ACCURACY = 0.75  # optimistic prior until evidence accumulates


class PlanningStrategist:
    def __init__(
        self,
        planner: Planner,
        memory: MemoryRouter,
        selector: StrategySelector | None = None,
        required_accuracy: float = 0.7,
        ladder: list[ModelTier] | None = None,
    ) -> None:
        self.planner = planner
        self.memory = memory
        self.selector = selector or StrategySelector()
        self.required_accuracy = required_accuracy
        # Ordered cheapest-first; the selector picks the cheapest tier that
        # still meets the verification threshold (FrugalGPT cascade).
        self.ladder = list(ladder) if ladder else []

    def plan(
        self, workspace: TaskWorkspace, force_model: bool = False
    ) -> tuple[Plan, str, MemoryRecord | None, ModelTier | None]:
        """Return (plan, strategy_name, skill_record_if_replayed, chosen_tier).

        ``chosen_tier`` is the ModelTier the runtime should plan with (None for
        skill_replay or when no ladder is configured).
        """
        if not force_model:
            options: list[StrategyMetrics] = []
            match = match_skill(self.memory.skills, workspace.goal)
            if match is not None:
                record, bound = match
                options.append(
                    StrategyMetrics(
                        name="skill_replay",
                        accuracy=record.confidence,
                        reliability=record.confidence,
                        cost=0.0,
                        latency_s=0.001,
                    )
                )
            # Layer 4 (FrugalGPT): offer every configured model tier as a
            # candidate, regardless of whether a skill matched. The selector
            # picks the cheapest tier meeting the verification threshold.
            for tier in self.ladder:
                options.append(
                    StrategyMetrics(
                        name=f"model:{tier.name}",
                        accuracy=self._tier_accuracy(tier),
                        reliability=self._tier_accuracy(tier),
                        cost=tier.cost,
                        latency_s=tier.prior_latency_s,
                    )
                )
            if options:
                choice = self.selector.choose(
                    options, required_accuracy=self.required_accuracy
                )
                if choice is not None and choice.name == "skill_replay" and match is not None:
                    plan = instantiate_plan(record, bound)
                    workspace.plan = [s.to_dict() for s in plan.steps]
                    return plan, "skill_replay", record, None
                if choice is not None and choice.name.startswith("model:"):
                    tier = next(
                        t for t in self.ladder if f"model:{t.name}" == choice.name
                    )
                    return self.planner.plan(workspace), f"model:{tier.name}", None, tier
        # Forced model, or no ladder configured: use the planner's own adapter.
        return self.planner.plan(workspace), "model_plan", None, None

    def update_skill(self, record: MemoryRecord, verified: bool) -> float:
        """Skill evolution: verified replays reinforce, failures halve."""
        confidence = (
            min(1.0, record.confidence * 0.9 + 0.1) if verified else record.confidence * 0.5
        )
        content = dict(record.content)
        content["uses"] = content.get("uses", 0) + 1
        self.memory.skills.add(
            content, tags=list(record.tags), confidence=confidence, record_id=record.id
        )
        return confidence

    def record_outcome(self, strategy: str, verified: bool) -> None:
        stats = self._stats()
        entry = stats.setdefault(strategy, {"runs": 0, "verified": 0})
        entry["runs"] += 1
        entry["verified"] += int(verified)
        self.memory.concepts.add(stats, tags=["strategy_stats"], record_id=STATS_RECORD_ID)

    def _stats(self) -> dict:
        record = self.memory.concepts.get(STATS_RECORD_ID)
        return dict(record.content) if record else {}

    def _tier_accuracy(self, tier: ModelTier) -> float:
        """Measured verified-rate for this tier, or its prior until evidence."""
        entry = self._stats().get(f"{_TIER_ACCURACY_PREFIX}{tier.name}")
        if not entry or not entry.get("runs"):
            return tier.prior_accuracy
        return entry["verified"] / entry["runs"]

    def record_tier_outcome(self, tier_name: str, verified: bool) -> None:
        """Persist a tier's verified-rate so the ladder learns from evidence."""
        stats = self._stats()
        key = f"{_TIER_ACCURACY_PREFIX}{tier_name}"
        entry = stats.setdefault(key, {"runs": 0, "verified": 0})
        entry["runs"] += 1
        entry["verified"] += int(verified)
        self.memory.concepts.add(stats, tags=["strategy_stats"], record_id=STATS_RECORD_ID)

    def _model_accuracy(self) -> float:
        entry = self._stats().get("model_plan")
        if not entry or not entry.get("runs"):
            return _DEFAULT_MODEL_ACCURACY
        return entry["verified"] / entry["runs"]
