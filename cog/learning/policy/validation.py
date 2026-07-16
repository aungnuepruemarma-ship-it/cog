"""Phase 3, Policy track: policy validation via the shared A/B laboratory.

Validation is a controlled experiment, not an assertion:
    Group A (baseline): tasks solved WITHOUT the policy
    Group B (treatment): tasks solved WITH the policy injected

We reuse cog.experiment.ab.run_experiment — the single canonical measurement
primitive. This module only builds the two variants and interprets the
ExperimentReport into a ValidationVerdict. It never promotes; the lifecycle
does that.

The two Variants here are thin callables so validation is testable without a
full runtime. In production, the caller supplies variants built from
ab.make_runtime_variant(... policy_injection=policy.action).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from cog.experiment.ab import ExperimentReport, Variant, run_experiment
from cog.runtime.task import Task

# Minimum absolute improvement (B over A) and its CI lower bound to PASS.
MIN_IMPROVEMENT = 0.10
CI_LOWER = 0.0

SolveFn = Callable[[Task], bool]


@dataclass
class ValidationVerdict:
    policy_id: str
    passed: bool
    baseline_failure_rate: float
    policy_failure_rate: float
    improvement: float                       # abs reduction in failure rate
    ci95: tuple[float, float]
    p_value: float | None
    effect_size: float | None
    recommendation: str                      # PROMOTE_TO_EXPERIMENTAL | REJECT
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "passed": self.passed,
            "baseline_failure_rate": round(self.baseline_failure_rate, 4),
            "policy_failure_rate": round(self.policy_failure_rate, 4),
            "improvement": round(self.improvement, 4),
            "ci95": [round(x, 4) for x in self.ci95],
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "recommendation": self.recommendation,
            "detail": self.detail,
        }


def validate_policy(
    policy_id: str,
    baseline_solve: SolveFn,
    treatment_solve: SolveFn,
    tasks: list[Task],
    seed: int = 0,
) -> ValidationVerdict:
    baseline = Variant(id="baseline", solve=baseline_solve)
    treatment = Variant(id=f"policy:{policy_id}", solve=treatment_solve)
    report: ExperimentReport = run_experiment(baseline, treatment, tasks, seed=seed)

    base_fail = 1.0 - report.a_stats.success_rate
    pol_fail = 1.0 - report.b_stats.success_rate
    improvement = base_fail - pol_fail  # absolute failure-rate reduction

    # Statistically sound difference-of-proportions 95% CI (B vs A):
    #   (p_b - p_a) +/- z * sqrt(p_a(1-p_a)/n_a + p_b(1-p_b)/n_b)
    import math
    pa = report.a_stats.success_rate
    pb = report.b_stats.success_rate
    se = math.sqrt(pa * (1 - pa) / report.n + pb * (1 - pb) / report.n)
    ci_low = (pb - pa) - 1.96 * se     # treat as failure reduction: (base_fail - pol_fail) CI
    ci_high = (pb - pa) + 1.96 * se
    # Express as failure-rate delta CI (base_fail - pol_fail = (1-pa) - (1-pb) = pb - pa)
    delta_ci = (ci_low, ci_high)

    passed = improvement >= MIN_IMPROVEMENT and delta_ci[0] > CI_LOWER
    return ValidationVerdict(
        policy_id=policy_id,
        passed=passed,
        baseline_failure_rate=base_fail,
        policy_failure_rate=pol_fail,
        improvement=improvement,
        ci95=delta_ci,
        p_value=report.comparison_pvalue,
        effect_size=report.comparison_effect,
        recommendation="PROMOTE_TO_EXPERIMENTAL" if passed else "REJECT",
        detail={"n": report.n, "experiment_id": report.experiment_id},
    )
