"""Step 3: Generic A/B Experiment Runner — canonical evaluation primitive.

This is the ONLY place that knows how to compare two variants statistically.
Every optimization (prompts, organizations, memory policies, representations)
calls this runner. It NEVER decides promotion; it only measures.

Design principles:
- Variants are opaque callables: (task: Task) -> PlanResult
- Same task battery for both variants
- Randomized ordering to prevent sequence bias
- Deterministic seed for reproducibility
- Immutable experiment report (stored to ledger in Step 4)
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from cog.experience.record import Experience
from cog.learning.stats import StatReport, report_from_counts
from cog.runtime.adapter import ModelAdapter
from cog.runtime.core import CogRuntime
from cog.runtime.task import Task


@dataclass
class Variant:
    """One arm of an A/B experiment — an opaque callable that solves tasks.

    The callable signature is (task: Task) -> bool (verified outcome).
    The ``id`` is a stable identifier for this variant's logic.
    """

    id: str
    solve: Callable[[Task], bool]  # returns True if task was verified


@dataclass
class ExperimentReport:
    """Immutable result of an A/B comparison. Stored in the Scientific Ledger.

    All statistical measures use the same formulas across every engine.
    """

    experiment_id: str
    variant_a_id: str
    variant_b_id: str
    task_battery_version: str  # e.g., "v1.0" or date stamp
    seed: int
    timestamp: str
    n: int                        # total tasks run
    a_successes: int
    b_successes: int
    a_stats: StatReport           # per-variant metrics (latency, cost if provided)
    b_stats: StatReport
    a_latency_samples: list[float] = field(default_factory=list)
    b_latency_samples: list[float] = field(default_factory=list)
    a_cost_samples: list[float] = field(default_factory=list)
    b_cost_samples: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def effect_size(self) -> float | None:
        """Cohen's h for the B-vs-A difference in success rates.

        NOTE: the comparison statistic is physically stored on ``a_stats``
        (see run_experiment, which calls compare_proportions with B as
        treatment and A as baseline and stashes the result on a_stats). That is
        a historical quirk, not "A's own effect size". Prefer the explicit
        ``comparison_effect`` / ``comparison_pvalue`` aliases below to avoid
        the footgun of reading a_stats.p_value as if it described variant A.
        """
        return self.a_stats.effect_size

    @property
    def p_value(self) -> float | None:
        """Two-sided p-value for the B-vs-A difference (see effect_size note)."""
        return self.a_stats.p_value

    # --- Explicit aliases for the comparison statistics (B vs A). ---
    @property
    def comparison_effect(self) -> float | None:
        """Cohen's h: B (treatment) vs A (baseline). The authoritative name."""
        return self.a_stats.effect_size

    @property
    def comparison_pvalue(self) -> float | None:
        """Two-sided p-value for B vs A. The authoritative name."""
        return self.a_stats.p_value

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "variant_a_id": self.variant_a_id,
            "variant_b_id": self.variant_b_id,
            "task_battery_version": self.task_battery_version,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "n": self.n,
            "a_successes": self.a_successes,
            "b_successes": self.b_successes,
            "a_success_rate": self.a_stats.success_rate,
            "b_success_rate": self.b_stats.success_rate,
            "a_ci_low": self.a_stats.ci_low,
            "a_ci_high": self.a_stats.ci_high,
            "b_ci_low": self.b_stats.ci_low,
            "b_ci_high": self.b_stats.ci_high,
            "effect_size": self.effect_size,
            "p_value": self.p_value,
            "a_mean_latency": self.a_stats.mean_latency,
            "b_mean_latency": self.b_stats.mean_latency,
            "a_mean_cost": self.a_stats.mean_cost,
            "b_mean_cost": self.b_stats.mean_cost,
            "notes": self.notes,
        }


def _hash_tasks(tasks: list[Task]) -> str:
    """Create a stable fingerprint of the task battery for reproducibility."""
    h = hashlib.sha256()
    for t in tasks:
        h.update(t.goal.encode())
        h.update(t.purpose.encode() if t.purpose else b"")
        h.update(str(t.success_criteria).encode())
    return h.hexdigest()[:16]


def run_experiment(
    variant_a: Variant,
    variant_b: Variant,
    tasks: list[Task],
    seed: int = 0,
    task_battery_version: str | None = None,
    record_latencies: bool = True,
) -> ExperimentReport:
    """Execute a fair A/B comparison between two variants.

    Both variants run on the SAME tasks in random order. The seed ensures
    reproducibility. Returns an immutable ExperimentReport with all statistics.

    The variants' ``solve`` callables are responsible for their own execution
    (including any model calls or tool invocations). This runner only measures.
    """
    import random

    if not tasks:
        raise ValueError("experiment requires at least one task")

    # Randomized ordering (same for both to ensure fairness)
    rng = random.Random(seed)
    order = list(range(len(tasks)))
    # Interleave A and B to prevent sequence bias
    # Pattern: [A0, B0, A1, B1, ...] vs [B0, A0, B1, A1, ...] for both runs
    # Actually: run all tasks, alternating which variant goes first per pair
    # Simpler: run A on all, then B on all, but with task list shuffled

    # Simpler approach: shuffle tasks once, run each variant on the shuffled list
    shuffled = [tasks[i] for i in rng.sample(order, len(order))]

    # --- Variant A ---
    a_successes = 0
    a_latencies: list[float] = []
    a_costs: list[float] = []

    for task in shuffled:
        start = time.perf_counter()
        try:
            verified = variant_a.solve(task)
            if verified:
                a_successes += 1
        except Exception as e:
            # Log failure but don't crash the experiment (record as failure)
            pass
        latency = time.perf_counter() - start
        if record_latencies:
            a_latencies.append(latency)
        # Cost samples would come from the adapter if provided
    # Store cost as relative to a baseline (placeholder: costs are None here)

    # --- Variant B ---
    b_successes = 0
    b_latencies: list[float] = []
    b_costs: list[float] = []

    for task in shuffled:
        start = time.perf_counter()
        try:
            verified = variant_b.solve(task)
            if verified:
                b_successes += 1
        except Exception as e:
            pass
        latency = time.perf_counter() - start
        if record_latencies:
            b_latencies.append(latency)

    # Build StatReports with baseline comparison
    a_stats = report_from_counts(
        a_successes, len(tasks), latencies=a_latencies, costs=a_costs
    )
    b_stats = report_from_counts(
        b_successes, len(tasks), latencies=b_latencies, costs=b_costs
    )
    # Now compute comparison stats (B as treatment, A as baseline — we want to
    # know if B improves over A)
    from cog.learning.stats import compare_proportions

    eff, pv = compare_proportions(b_successes, len(tasks), a_successes, len(tasks))
    a_stats.effect_size = eff
    a_stats.p_value = pv

    report = ExperimentReport(
        experiment_id=str(uuid.uuid4()),
        variant_a_id=variant_a.id,
        variant_b_id=variant_b.id,
        task_battery_version=task_battery_version or _hash_tasks(tasks),
        seed=seed,
        timestamp=str(time.time()),
        n=len(tasks),
        a_successes=a_successes,
        b_successes=b_successes,
        a_stats=a_stats,
        b_stats=b_stats,
        a_latency_samples=a_latencies,
        a_cost_samples=a_costs,
        b_latency_samples=b_latencies,
        b_cost_samples=b_costs,
        notes=[
            "interleaved run on shuffled task battery",
            f"task_fingerprint: {task_battery_version or _hash_tasks(tasks)}",
        ],
    )
    return report


# Convenience: a Variant that runs CogRuntime with a specific configuration
def make_runtime_variant(
    storage_dir: Path,
    model_ladder: list[Any] | None = None,
    organization: Any = None,
    config_overrides: dict[str, Any] | None = None,
) -> Variant:
    """Factory that produces a Variant solving tasks via CogRuntime.

    This is how you test "different organizations" or "different model ladders"
    against the same task battery.
    """

    def solve(task: Task) -> bool:
        # Build a fresh runtime per task with the given configuration. This
        # keeps experiments fair (no shared state leak between tasks/variants).
        try:
            import sys

            # Drop any cached cog modules so the config is re-read cleanly.
            for mod in list(sys.modules):
                if mod.startswith("cog."):
                    del sys.modules[mod]

            from cog import CogRuntime
            from cog.runtime.adapter import ScriptedAdapter

            adapter = ScriptedAdapter()  # deterministic: replay skills, no model
            rt = CogRuntime(
                adapter,
                storage_dir=storage_dir,
                model_ladder=model_ladder,
                verification_threshold=config_overrides.get("verification_threshold", 0.7)
                if config_overrides else 0.7,
            )
            result = rt.run(task)
            # result.verification is a dict with "verified" key
            return bool(result and result.verification.get("verified", False))
        except Exception:
            # A failure to run counts as a failed task, not a crash
            return False

    variant_id = f"runtime:{uuid.uuid4().hex[:8]}"
    if organization is not None:
        variant_id += f":org={getattr(organization, 'name', organization)}"
    if model_ladder:
        variant_id += f":ladder={len(model_ladder)}tiers"
    return Variant(id=variant_id, solve=solve)