"""Runtime benchmark: behavioral improvement metrics.

This module evaluates POLICIES (prescriptive interventions) against the hidden
intervention labels (effective_interventions). It deliberately does NOT consume
belief observation labels -- that is the learning benchmark's concern.

RuntimeReport (behavior-only metrics):
    baseline_success   -- no-learning control success rate
    treatment_success  -- learned-policy treatment success rate
    policy_lift        -- treatment - control (behavioral improvement)
    policy_precision   -- frac of active policy actions that are effective
    policy_recall      -- frac of effective interventions covered by active policies
    runtime_cost       -- mean per-task wall-clock seconds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cog.learning.policy.model import Policy
from cog.learning.stats import proportion_ci


@dataclass
class RuntimeReport:
    baseline_success: float = 0.0
    treatment_success: float = 0.0
    policy_lift: float = 0.0
    policy_lift_ci: tuple[float, float] | None = None  # Wald CI of the lift (treatment - control)
    policy_precision: float | None = None
    policy_recall: float | None = None
    runtime_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_success": self.baseline_success,
            "treatment_success": self.treatment_success,
            "policy_lift": self.policy_lift,
            "policy_lift_ci": self.policy_lift_ci,
            "policy_precision": self.policy_precision,
            "policy_recall": self.policy_recall,
            "runtime_cost": self.runtime_cost,
        }


def policy_precision_recall(active_policies: list[Policy],
                            effective_interventions: set[str]) -> tuple[float | None, float | None]:
    """Precision/recall of the active policies' chosen interventions vs the
    ground-truth effective intervention set.

    Precision = |active actions that are effective| / |active actions|
    Recall    = |effective interventions covered by active actions| / |effective|
    """
    if not active_policies:
        return None, None
    actions = [p.action for p in active_policies]
    effective_actions = [a for a in actions if a in effective_interventions]
    precision = len(effective_actions) / len(actions) if actions else None
    covered = {a for a in actions if a in effective_interventions}
    recall = len(covered) / len(effective_interventions) if effective_interventions else None
    return precision, recall
