"""Shared statistical reporting for Cog's learning loop and experiments.

Every inductive engine (skill compile, representation competition, primitive
discovery, organization selection, routing) should report the *same* evidence
format so improvements are comparable across layers. This module provides:

- ``StatReport`` — n, success_rate, 95% CI, effect_size, p_value, latency/cost
- ``proportion_ci`` — Wilson score interval for a success count
- ``compare_proportions`` — two-sample z-test for A/B (returns effect size +
  p-value)
- ``bootstrap_ci`` — non-parametric CI for arbitrary means (latency, cost)

Design: pure functions, no side effects, fully deterministic given inputs.
Reused by the learning loop (step 2) and the Experiment Manager (step 4).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field


@dataclass
class StatReport:
    """One comparable evidence packet.

    Fields:
        n            — sample count
        success_rate — observed verified-rate in [0,1]
        ci_low       — lower bound of 95% CI on success_rate
        ci_high      — upper bound of 95% CI on success_rate
        effect_size  — standardized difference vs baseline (Cohen's h for
                       proportions, or None if no baseline)
        p_value      — two-sided test p-value vs baseline (or None)
        mean_latency — mean seconds per task (or None)
        mean_cost    — mean relative cost per task (or None)
    """

    n: int = 0
    success_rate: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    effect_size: float | None = None
    p_value: float | None = None
    mean_latency: float | None = None
    mean_cost: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StatReport":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def __str__(self) -> str:
        if self.n == 0:
            return "StatReport(n=0, no data)"
        parts = [f"n={self.n}", f"success={self.success_rate:.1%}",
                 f"95% CI=[{self.ci_low:.1%}, {self.ci_high:.1%}]"]
        if self.effect_size is not None:
            parts.append(f"effect={self.effect_size:.3f}")
        if self.p_value is not None:
            parts.append(f"p={self.p_value:.4f}")
        if self.mean_latency is not None:
            parts.append(f"latency={self.mean_latency:.2f}s")
        if self.mean_cost is not None:
            parts.append(f"cost={self.mean_cost:.2f}")
        return "StatReport(" + ", ".join(parts) + ")"


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's algorithm, good to ~1e-9).

    Used for z-scores in confidence intervals and hypothesis tests.
    """
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
         1.383577518672690e2, -3.066479806614716e1, 2.506628277459239]
    b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
         6.680131188771972e1, -1.328068155288572e1]
    c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
         3.754408661907416]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def proportion_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    More accurate than the normal approximation when n is small or p is near 0
    or 1 — exactly the regime Cog's early evidence lives in.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def compare_proportions(
    succ_a: int, n_a: int, succ_b: int, n_b: int
) -> tuple[float | None, float | None]:
    """Two-sample pooled z-test for proportions.

    Returns (effect_size, p_value) where effect_size is Cohen's h:
        h = 2 * arcsin(sqrt(p_a)) - 2 * arcsin(sqrt(p_b))
    Positive h means A outperforms B. p_value is two-sided.
    """
    if n_a == 0 or n_b == 0:
        return (None, None)
    p_a = succ_a / n_a
    p_b = succ_b / n_b
    # Cohen's h
    h = 2 * math.asin(math.sqrt(p_a)) - 2 * math.asin(math.sqrt(p_b))
    # Pooled z-test
    p_pool = (succ_a + succ_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return (h, 0.0 if p_a != p_b else 1.0)
    z = (p_a - p_b) / se
    # two-sided p via normal CDF
    p_value = 2 * (1 - _norm_cdf(abs(z)))
    return (round(h, 4), round(p_value, 6))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bootstrap_ci(
    samples: list[float], n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Non-parametric bootstrap CI for the mean of arbitrary samples.

    Returns (mean, low, high) at the 2.5/97.5 percentiles. Deterministic given
    ``seed`` so reports are reproducible. Used for latency/cost where the
    distribution is unknown and may be skewed.
    """
    import random
    if not samples:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        boot = [rng.choice(samples) for _ in samples]
        means.append(sum(boot) / len(boot))
    means.sort()
    mean = sum(samples) / len(samples)
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]
    return (mean, lo, hi)


def report_from_counts(
    successes: int,
    n: int,
    baseline_succ: int | None = None,
    baseline_n: int | None = None,
    latencies: list[float] | None = None,
    costs: list[float] | None = None,
) -> StatReport:
    """Build a StatReport from raw counts + optional baselines.

    If a baseline is supplied, effect_size and p_value are computed vs it.
    Latency/cost means get bootstrap CIs (means stored, CI in notes via dict).
    """
    if n == 0:
        return StatReport()
    p = successes / n
    ci_low, ci_high = proportion_ci(successes, n)
    rep = StatReport(
        n=n,
        success_rate=p,
        ci_low=ci_low,
        ci_high=ci_high,
    )
    if baseline_succ is not None and baseline_n and baseline_n > 0:
        eff, pv = compare_proportions(successes, n, baseline_succ, baseline_n)
        rep.effect_size = eff
        rep.p_value = pv
    if latencies:
        mean, lo, hi = bootstrap_ci(latencies)
        rep.mean_latency = round(mean, 4)
    if costs:
        mean, lo, hi = bootstrap_ci(costs)
        rep.mean_cost = round(mean, 4)
    return rep
