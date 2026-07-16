"""Runtime benchmark: behavioral improvement through policy injection.

Runs the REAL runtime A/B (control = no-learning, treatment = active policies
injected) and reports behavior-only metrics. Policy correctness is measured
against the hidden effective_interventions labels; belief observation labels are
NOT consumed here (that is the learning benchmark).

Owns cog.experiment.runtime_ab only. Does not import generator labels or the
belief layer.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from cog.evaluation.infra.generators import GeneratedDataset
from cog.evaluation.infra.baselines import NoLearningRuntime
from cog.evaluation.runtime.ab_metrics import RuntimeReport, policy_precision_recall
from cog.experiment.runtime_ab import run_runtime_ab
from cog.learning.policy.model import Policy
from cog.learning.policy.runtime import PolicyContext
from cog.learning.stats import proportion_ci
from cog.runtime.task import Task


def _tasks_from_dataset(dataset: GeneratedDataset, n: int = 60) -> list[Task]:
    tasks: list[Task] = []
    for i, exp in enumerate(dataset.experiences[:n]):
        tasks.append(Task(
            goal="deploy flask app", purpose="build container",
            domain=exp.domain, difficulty="medium",
        ))
    return tasks


def evaluate_runtime(dataset: GeneratedDataset,
                     active_policies: list[Policy]) -> RuntimeReport:
    storage = Path(mkdtemp(prefix="rt_bench_"))
    tasks = _tasks_from_dataset(dataset)

    # Build the planner context from the active policies (same injection path
    # production uses).
    ctx = PolicyContext(policies=active_policies,
                        justifications=[list(p.justification) for p in active_policies])

    report = run_runtime_ab(ctx, tasks, storage, seed=42)

    baseline = report.a_stats.success_rate
    treatment = report.b_stats.success_rate
    lift = treatment - baseline
    cost = (sum(report.b_latency_samples) / len(report.b_latency_samples)
            if report.b_latency_samples else 0.0)

    # Wald CI for the lift (difference of two proportions). Without this the
    # lift is just a raw number and not interpretable as "improvement".
    n_a = report.a_stats.n if hasattr(report.a_stats, "n") else len(tasks)
    n_b = report.b_stats.n if hasattr(report.b_stats, "n") else len(tasks)
    a_lo, a_hi = proportion_ci(report.a_successes, n_a)
    b_lo, b_hi = proportion_ci(report.b_successes, n_b)
    w_a, w_b = (a_hi - a_lo) / 2.0, (b_hi - b_lo) / 2.0  # Wilson half-widths
    se = ((w_b / 1.96) ** 2 + (w_a / 1.96) ** 2) ** 0.5  # SE of the difference
    lift_ci = (round(lift - 1.96 * se, 4), round(lift + 1.96 * se, 4))

    # Policy correctness vs hidden intervention labels (only if labels exist).
    eff: set[str] = set()
    if dataset.labels:
        eff = set().union(*[v.effective_interventions for v in dataset.labels.values()])
    precision, recall = policy_precision_recall(active_policies, eff)

    return RuntimeReport(
        baseline_success=round(baseline, 4),
        treatment_success=round(treatment, 4),
        policy_lift=round(lift, 4),
        policy_lift_ci=lift_ci,
        policy_precision=precision,
        policy_recall=recall,
        runtime_cost=round(cost, 6),
    )
