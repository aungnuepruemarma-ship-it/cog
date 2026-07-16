"""Epistemic validation suite: result + report types.

A test returns a TestResult. The harness aggregates them and emits a verdict
plus the decision gate. This is the "constitution test" for Cog: future
changes must not break the guarantees these tests encode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestResult:
    name: str
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "metrics": self.metrics,
            "detail": self.detail,
        }


# Decision gate: the bar a v0.1 release must clear before adding more layers.
GATE = {
    "false_active_beliefs": 0,        # no spurious belief may reach ACTIVE
    "scope_leakage": 0,               # no over-generalized belief
    "unsupported_promotions": 0,      # every ACTIVE policy justified by a real belief
    "replay_success_min": 0.95,       # >=95% deterministic structured reproduction
    "contradiction_detection_min": 0.90,  # >=90% of contradictions detected
    "evidence_validation_rate": 1.0,  # 100% of stored evidence validates
}


def evaluate_gate(results: list[TestResult]) -> dict[str, Any]:
    """Check the decision gate; return whether Cog may proceed to Policy/HTN."""
    by_name = {r.name: r for r in results}
    checks: dict[str, bool] = {}

    fp = by_name.get("false_pattern")
    checks["false_active_beliefs"] = bool(fp and fp.metrics.get("false_active", 0) == 0)

    sp = by_name.get("scope_precision")
    checks["scope_leakage"] = bool(sp and sp.metrics.get("over_generalizations", 1) == 0)

    ab = by_name.get("adversarial_poisoning")
    checks["unsupported_promotions"] = bool(ab and ab.metrics.get("false_active", 0) == 0)

    rp = by_name.get("replay")
    checks["replay_success_min"] = bool(rp and rp.metrics.get("replay_rate", 0) >= GATE["replay_success_min"])

    cd = by_name.get("contradiction_detection")
    checks["contradiction_detection_min"] = bool(
        cd and cd.metrics.get("detection_rate", 0) >= GATE["contradiction_detection_min"]
    )

    ev = by_name.get("evidence_efficiency")
    checks["evidence_validation_rate"] = bool(
        ev and ev.metrics.get("validation_rate", 0) >= GATE["evidence_validation_rate"]
    )

    all_pass = all(r.passed for r in results) and all(checks.values())
    return {
        "gate": GATE,
        "checks": checks,
        "proceed": all_pass,
        "verdict": "PASS" if all_pass else "FAIL",
    }
