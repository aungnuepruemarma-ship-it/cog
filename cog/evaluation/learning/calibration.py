"""Learning benchmark: calibration (Expected Calibration Error).

Calibration uses a DETERMINISTIC train/eval split: beliefs are synthesized on the
train split, then their confidence is compared against observed accuracy on the
HELD-OUT eval split. No leakage -- eval experiences never reach the synthesizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cog.evaluation.infra.stats import compute_calibration
from cog.experience.record import Experience
from cog.learning.belief.model import Belief, BeliefState


def _preflight_absent(exp: Experience) -> bool:
    tools = [s.get("tool", "") for s in (exp.execution or [])]
    return not any("preflight" in t or "inspect" in t or "check" in t for t in tools)


def _obs_failure_rate(belief: Belief, eval_exps: list[Experience]) -> float | None:
    """Observed failure rate on eval split for the belief's condition (domain +
    preflight-absent). Returns None if no matching eval experiences."""
    dom = belief.scope.domain
    matched = [e for e in eval_exps
               if e.domain == dom and _preflight_absent(e)]
    if not matched:
        return None
    failures = sum(1 for e in matched if e.outcome == "failure")
    return failures / len(matched)


def calibration_ece(active_beliefs: list[Belief],
                    eval_exps: list[Experience],
                    n_bins: int = 5):
    """Compute ECE over ACTIVE beliefs using held-out eval outcomes.

    Returns a CalibrationResult (expected_calibration_error, bins, well_calibrated).
    Beliefs with no matching eval experiences are excluded (cannot be assessed).
    """
    confidences: list[float] = []
    accuracies: list[float] = []
    for b in active_beliefs:
        rate = _obs_failure_rate(b, eval_exps)
        if rate is None:
            continue
        # The belief predicts failure_probability; observed accuracy for that
        # prediction is the empirical failure rate under its condition.
        confidences.append(float(b.confidence))
        accuracies.append(float(rate))
    return compute_calibration(confidences, accuracies, n_bins=n_bins)
