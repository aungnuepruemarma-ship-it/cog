"""Evaluation infrastructure: statistical utilities.

Reuses the real cog.learning.stats (Wald CI, Cohen's h, bootstrap) and adds
calibration (Expected Calibration Error) used by the capability suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cog.learning.stats import compare_proportions, proportion_ci, bootstrap_ci


def wald_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wald confidence interval for a proportion."""
    return proportion_ci(successes, n, z)


def cohens_h(succ_b: int, n_b: int, succ_a: int, n_a: int) -> float:
    _, h, _ = compare_proportions(succ_b, n_b, succ_a, n_a, z=1.96)
    return float(h)


@dataclass
class CalibrationResult:
    expected_calibration_error: float
    bins: list[dict[str, Any]]
    well_calibrated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_calibration_error": self.expected_calibration_error,
            "bins": self.bins,
            "well_calibrated": self.well_calibrated,
        }


def compute_calibration(confidences: list[float], accuracies: list[float],
                        n_bins: int = 5) -> CalibrationResult:
    """Expected Calibration Error.

    confidences / accuracies are paired per belief. ECE = weighted mean of
    |mean confidence - accuracy| across bins. Lower is better; <=0.1 is the
    v0.1 target.
    """
    if not confidences:
        return CalibrationResult(0.0, [], True)
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        idx = [k for k, c in enumerate(confidences) if lo <= c < hi or (i == n_bins - 1 and c == hi)]
        if not idx:
            continue
        avg_conf = sum(confidences[k] for k in idx) / len(idx)
        acc = sum(accuracies[k] for k in idx) / len(idx)
        bins.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "avg_confidence": avg_conf,
            "accuracy": acc,
            "error": abs(avg_conf - acc),
            "count": len(idx),
        })
    total = sum(b["count"] for b in bins) or 1
    ece = sum(b["error"] * b["count"] for b in bins) / total
    return CalibrationResult(ece, bins, bool(ece <= 0.1))
