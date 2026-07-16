"""Step 4: Experiment Manager — orchestrates A/B comparisons with the Ledger.

Builds ON the generic A/B runner (cog.experiment.ab) — it does NOT duplicate
statistics. Its responsibilities are:

1. Register hypotheses (what we're testing)
2. Schedule experiments (which variants, which task battery)
3. Invoke the A/B runner
4. Record results in the Scientific Ledger as claims
5. Enforce promotion / no-regression policies

The policies are explicit, evidence-gated rules:

- PROMOTE: variant B is adopted over A only if
    n >= min_n AND p_value < alpha AND effect_size >= min_effect
- NO-REGRESSION: if B was previously adopted and a new experiment shows
    significant regression, flag for human/auto review

Nothing is promoted without verified evidence meeting these gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from cog.experiment.ab import ExperimentReport, Variant, run_experiment
from cog.runtime.task import Task
from cog.science.ledger import Ledger


@dataclass
class PromotionPolicy:
    """Evidence gates a variant must clear to be promoted."""

    min_n: int = 30           # minimum sample size
    alpha: float = 0.05       # significance threshold
    min_effect: float = 0.1   # minimum Cohen's h (small effect)
    min_success_rate: float = 0.5  # must beat random chance


@dataclass
class ExperimentSpec:
    """What to run: a hypothesis, two variants, a task battery, a seed."""

    hypothesis_id: str
    hypothesis: str
    variant_a: Variant
    variant_b: Variant
    tasks: list[Task]
    seed: int = 0
    task_battery_version: str | None = None


class ExperimentManager:
    """Runs experiments, records claims, enforces promotion policy."""

    def __init__(
        self,
        memory=None,  # MemoryRouter; optional so the promotion policy is unit-testable without a live DB
        policy: PromotionPolicy | None = None,
        ledger: "Ledger | None" = None,
    ) -> None:
        self.memory = memory
        # The ledger is only needed when recording claims (run/_record_experiment).
        # Constructing it lazily keeps the promotion *decision* logic independently
        # testable: _decision / check_no_regression never touch the ledger.
        self._ledger = ledger
        self.policy = policy or PromotionPolicy()

    @property
    def ledger(self) -> "Ledger":
        if self._ledger is None:
            if self.memory is None:
                raise RuntimeError(
                    "ExperimentManager has no memory and no injected ledger; "
                    "recording claims requires one of them."
                )
            self._ledger = Ledger(self.memory)
        return self._ledger

    def run(self, spec: ExperimentSpec) -> ExperimentReport:
        """Execute an experiment and record the outcome as a ledger claim.

        Returns the ExperimentReport (same object the A/B runner produced).
        Side effect: a claim is filed in the Scientific Ledger.
        """
        report = run_experiment(
            spec.variant_a,
            spec.variant_b,
            spec.tasks,
            seed=spec.seed,
            task_battery_version=spec.task_battery_version,
        )
        self._record_experiment(spec, report)
        return report

    def _decision(self, report: ExperimentReport) -> tuple[str, float]:
        """Apply the promotion policy to an experiment report.

        Returns (decision, confidence) where decision is one of
        "adopted" | "rejected" | "inconclusive".
        """
        pol = self.policy
        # Insufficient sample size
        if report.n < pol.min_n:
            return ("inconclusive", 0.0)
        # B must beat A significantly
        if report.p_value is None or report.p_value >= pol.alpha:
            return ("inconclusive", 0.0)
        if (report.effect_size or 0) < pol.min_effect:
            return ("inconclusive", 0.0)
        if report.b_stats.success_rate < pol.min_success_rate:
            return ("rejected", 1.0 - report.b_stats.success_rate)
        # Promotion justified
        confidence = 1.0 - (report.p_value or 1.0)
        return ("adopted", confidence)

    def _record_experiment(self, spec: ExperimentSpec, report: ExperimentReport) -> None:
        """File a claim in the Scientific Ledger for this experiment outcome."""
        decision, confidence = self._decision(report)
        subject_id = f"experiment:{spec.hypothesis_id}"
        metrics = {
            "n": report.n,
            "a_success_rate": report.a_stats.success_rate,
            "b_success_rate": report.b_stats.success_rate,
            "a_ci": [report.a_stats.ci_low, report.a_stats.ci_high],
            "b_ci": [report.b_stats.ci_low, report.b_stats.ci_high],
            "effect_size": report.effect_size,
            "p_value": report.p_value,
            "a_mean_latency": report.a_stats.mean_latency,
            "b_mean_latency": report.b_stats.mean_latency,
            "experiment_id": report.experiment_id,
            "task_battery_version": report.task_battery_version,
            "seed": report.seed,
        }
        self.ledger.record_claim(
            subject_id=subject_id,
            hypothesis=spec.hypothesis,
            experiment=(
                f"A/B comparison {report.variant_a_id} vs {report.variant_b_id} "
                f"on {report.n} tasks"
            ),
            dataset=[],  # task battery fingerprint is in metrics
            metrics=metrics,
            decision=decision,
            confidence=confidence,
            reproducible=True,  # deterministic given seed + battery
            claim_id=f"claim_experiment_{spec.hypothesis_id}_{report.experiment_id}",
            claim_type="experiment",  # protected evidence record (gated by PromotionGate)
            meta={
                "status": "completed",
                "passed_policy": decision == "adopted",
                "treatment_id": report.variant_b_id,
                "baseline_id": report.variant_a_id,
                "experiment_id": report.experiment_id,
            },
            _evidence_authority="experiment_runner",  # this module IS the authorized experiment producer
        )

    def check_no_regression(
        self, hypothesis_id: str, new_report: ExperimentReport
    ) -> tuple[bool, str]:
        """Enforce that an adopted variant hasn't regressed.

        Returns (ok, message). If a previous experiment adopted B and the new
        one shows significant regression below the BEST previously-adopted
        rate, ok=False with a warning.
        """
        claims = self.ledger.claims_about(f"experiment:{hypothesis_id}")
        if not claims:
            return (True, "no prior claims to compare")
        # Find all adopted claims, track the best B success rate seen
        adopted = [c for c in claims if c.content.get("decision") == "adopted"]
        if not adopted:
            return (True, "no prior adoption")
        best_prev_rate = max(
            c.content.get("metrics", {}).get("b_success_rate", 0.0) for c in adopted
        )
        new_rate = new_report.b_stats.success_rate
        # Regression if the new B rate drops materially below the best
        # previously-adopted B rate. We do NOT gate on new_report.p_value here
        # because the new report compares B against A (the current baseline),
        # whereas regression is about B dropping relative to its OWN past
        # performance. A significant absolute drop is sufficient evidence.
        if new_rate < best_prev_rate - 0.1:  # 10% absolute drop
            return (
                False,
                f"REGRESSION: B dropped from {best_prev_rate:.1%} to {new_rate:.1%}",
            )
        return (True, "no regression detected")
