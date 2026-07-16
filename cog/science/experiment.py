"""Thin experiment adapter: run a battery, emit one canonical experiment claim.

This is intentionally SMALL. It is not a second experiment framework. Its
single job:

    run(baseline, treatment, tasks)
        -> compute stats via cog.learning.stats (the shared primitives)
        -> apply PromotionPolicy (evidence-class aware)
        -> record ONE claim_type="experiment" claim

The claim schema here mirrors cog/experiment/manager.py's PromotionPolicy
field names (min_n, alpha, min_effect) and experiment-claim fields, so a
later consolidation into the canonical experiment subsystem is a drop-in.

It does NOT promote. Promotion is a separate, provenance-checked step
(Ledger.promote_claim) that references the experiment claim produced here.

Evidence classes (each its own acceptance criteria):
    formal                 -- proofs / invariants / exact equivalence
    deterministic_empirical-- reproducible benchmarks, no randomness
    statistical            -- A/B / stochastic evaluation (needs n, p, h)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from cog.learning.stats import compare_proportions, report_from_counts
from cog.memory.router import MemoryRouter
from cog.science.ledger import Ledger


class EvidenceClass(str, Enum):
    FORMAL = "formal"
    DETERMINISTIC_EMPIRICAL = "deterministic_empirical"
    STATISTICAL = "statistical"


@dataclass
class PromotionPolicy:
    """Canonical policy object. Field names match cog/experiment/manager.py
    so the two can be consolidated without migration."""

    min_n: int = 30
    alpha: float = 0.05
    min_effect: float = 0.1

    def evaluate(self, evidence_class: EvidenceClass, metrics: dict[str, Any]) -> tuple[bool, str]:
        """Decide whether an experiment's evidence is sufficient to promote.

        Returns (passed, reason). The experiment layer calls this once, when
        the experiment completes, and bakes the result into the claim as
        ``passed_policy``. The promotion gate later only READS that flag.
        """
        if evidence_class == EvidenceClass.FORMAL:
            if not metrics.get("reproducible", False):
                return (False, "formal evidence requires reproducible=True")
            return (True, "formal proof accepted")
        if evidence_class == EvidenceClass.DETERMINISTIC_EMPIRICAL:
            if not metrics.get("reproducible", False):
                return (False, "deterministic empirical requires reproducible=True")
            return (True, "deterministic empirical accepted")
        # STATISTICAL
        n = metrics.get("n")
        if not isinstance(n, int) or n < self.min_n:
            return (False, f"statistical needs n>={self.min_n}, got {n}")
        p = metrics.get("p_value")
        if p is None or p >= self.alpha:
            return (False, f"not significant (p={p} >= {self.alpha})")
        h = metrics.get("effect_size") or 0.0
        if h < self.min_effect:
            return (False, f"effect too small (h={h} < {self.min_effect})")
        return (True, "statistical promotion accepted")


@dataclass
class ExperimentSpec:
    subject_id: str
    hypothesis: str
    baseline: Callable[[Any], Any]      # callable returning a result per task
    treatment: Callable[[Any], Any]    # callable returning a result per task
    tasks: list[Any]
    evidence_class: EvidenceClass = EvidenceClass.STATISTICAL
    baseline_id: str = "baseline"
    treatment_id: str = "treatment"
    policy: PromotionPolicy = field(default_factory=PromotionPolicy)


def _successes(fn: Callable[[Any], Any], tasks: list[Any]) -> tuple[int, int, list[float]]:
    succ = 0
    lat: list[float] = []
    for t in tasks:
        res = fn(t)
        ok = bool(res.get("verified")) if isinstance(res, dict) else bool(res)
        succ += 1 if ok else 0
        if isinstance(res, dict) and isinstance(res.get("latency_s"), (int, float)):
            lat.append(res["latency_s"])
    return succ, len(tasks), lat


def run_experiment(ledger: Ledger, spec: ExperimentSpec) -> dict[str, Any]:
    """Run baseline vs treatment on the battery, record one experiment claim.

    Returns the recorded experiment claim's metrics dict (including
    passed_policy). Does NOT promote -- promotion is a separate step that
    references the returned experiment_id.
    """
    b_succ, b_n, b_lat = _successes(spec.baseline, spec.tasks)
    t_succ, t_n, t_lat = _successes(spec.treatment, spec.tasks)

    b_stats = report_from_counts(b_succ, b_n, latencies=b_lat)
    t_stats = report_from_counts(t_succ, t_n, latencies=t_lat)
    eff, pv = compare_proportions(t_succ, t_n, b_succ, b_n)

    metrics = {
        "n": t_n,
        "baseline_success_rate": b_stats.success_rate,
        "treatment_success_rate": t_stats.success_rate,
        "effect_size": eff,
        "p_value": pv,
        "baseline_ci": [b_stats.ci_low, b_stats.ci_high],
        "treatment_ci": [t_stats.ci_low, t_stats.ci_high],
        "evidence_class": spec.evidence_class.value,
        "baseline_id": spec.baseline_id,
        "treatment_id": spec.treatment_id,
        "reproducible": (spec.evidence_class != EvidenceClass.STATISTICAL),
    }
    passed, reason = spec.policy.evaluate(spec.evidence_class, metrics)
    metrics["passed_policy"] = passed
    metrics["policy_reason"] = reason
    metrics["status"] = "completed"  # retained for backward-compatible records
    metrics["treatment_id"] = spec.treatment_id
    metrics["baseline_id"] = spec.baseline_id

    claim = ledger.record_claim(
        subject_id=spec.subject_id,
        hypothesis=spec.hypothesis,
        experiment=f"{spec.evidence_class.value} comparison "
                   f"{spec.baseline_id} vs {spec.treatment_id} on {t_n} tasks",
        dataset=[],
        metrics=metrics,
        decision="adopted" if passed else "rejected",
        confidence=1.0 - (pv if pv is not None else 1.0),
        reproducible=(spec.evidence_class != EvidenceClass.STATISTICAL),
        claim_type="experiment",
        # Authorized evidence producer: only this runner may emit "experiment" claims.
        _evidence_authority="experiment_runner",
        meta={
            "status": "completed",
            "passed_policy": passed,
            "treatment_id": spec.treatment_id,
            "baseline_id": spec.baseline_id,
            "policy_reason": reason,
        },
        claim_id=f"exp_{spec.subject_id}",
    )
    return {"experiment_id": claim.id, **metrics}
