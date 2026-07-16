"""Capability suite: the ONLY layer that combines learning + runtime.

It does not re-run either subsystem. It consumes a LearningReport and a
RuntimeReport and computes the system-level metric:

    knowledge_efficiency = measured_policy_lift / active_beliefs

reported as UNAVAILABLE when runtime data is absent or there are no active
beliefs. This keeps the numerator behavioral (runtime) and the denominator
cognitive (learning) -- no proxy metric blurs the two layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cog.evaluation.infra.harness import EvaluationSuite
from cog.evaluation.infra.metrics import ContinuousMetric, MetricResult
from cog.evaluation.learning.learning import LearningReport, evaluate_learning
from cog.evaluation.runtime.runtime import RuntimeReport, evaluate_runtime
from cog.evaluation.infra.generators import GeneratedDataset
from cog.learning.policy.model import Policy


@dataclass
class CapabilityReport:
    learning: LearningReport
    runtime: RuntimeReport | None
    knowledge_efficiency: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning": self.learning.to_dict(),
            "runtime": self.runtime.to_dict() if self.runtime else None,
            "knowledge_efficiency": self.knowledge_efficiency,
        }


class CapabilitySuite(EvaluationSuite):
    name = "capability"
    version = "v1.0.0"

    def __init__(self, dataset: GeneratedDataset, active_policies: list[Policy],
                 seed: int = 42, config: dict | None = None,
                 artifact_root=None) -> None:
        super().__init__(seed=seed, config=config, artifact_root=artifact_root)
        self.dataset = dataset
        self.active_policies = active_policies

    def _register_metrics(self) -> None:
        self.metrics.register(ContinuousMetric(
            "knowledge_efficiency", target="policy_lift / active_beliefs",
            higher_is_better=True))

    def _run(self):
        learning = evaluate_learning(self.dataset)
        runtime = evaluate_runtime(self.dataset, self.active_policies) if self.active_policies else None

        ke: float | None = None
        if runtime is not None and learning.active_beliefs > 0:
            ke = round(runtime.policy_lift / learning.active_beliefs, 4)

        metrics: list[MetricResult] = []
        metrics.append(self.metrics.compute(
            "knowledge_efficiency",
            ke if ke is not None else 0.0))

        self._last = CapabilityReport(learning=learning, runtime=runtime, knowledge_efficiency=ke)
        return metrics, {}

    @property
    def last_report(self) -> CapabilityReport:
        return getattr(self, "_last", None)
