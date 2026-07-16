"""Epistemic validation suite: orchestration + reporting.

Runs the v0.1 guarantee tests against the real Belief/Policy engines and emits
a PASS/FAIL verdict with the decision gate. This is the constitution test:
Cog may not proceed to higher layers (HTN, skills, causal graphs, autonomous
modification) until this suite passes.
"""

from __future__ import annotations

import sys
from typing import Callable

from cog.evaluation.epistemic_suite.report import TestResult, evaluate_gate


# Test functions are imported lazily to avoid import cycles at module load.
def _registry() -> list[Callable[[], TestResult]]:
    from cog.evaluation.epistemic_suite.tests import (
        test_false_pattern,
        test_scope_precision,
        test_replay,
        test_contradiction_detection,
        test_evidence_efficiency,
        test_adversarial_poisoning,
        test_belief_policy_cascade,
    )

    return [
        test_false_pattern.test_false_pattern,
        test_scope_precision.test_scope_precision,
        test_replay.test_replay,
        test_contradiction_detection.test_contradiction_detection,
        test_evidence_efficiency.test_evidence_efficiency,
        test_adversarial_poisoning.test_adversarial_poisoning,
        test_belief_policy_cascade.test_belief_policy_cascade,
    ]


class EpistemicTestHarness:
    def __init__(self, tests: list[Callable[[], TestResult]] | None = None) -> None:
        self.tests = tests or _registry()
        self.results: list[TestResult] = []

    def run_all(self) -> dict:
        self.results = []
        for test in self.tests:
            try:
                self.results.append(test())
            except Exception as exc:  # a crashing test is a failure, surfaced honestly
                self.results.append(
                    TestResult(name=getattr(test, "__name__", "unknown"),
                               passed=False,
                               detail=f"EXCEPTION: {type(exc).__name__}: {exc}")
                )
        return self.generate_report()

    def generate_report(self) -> dict:
        passed = sum(1 for r in self.results if r.passed)
        gate = evaluate_gate(self.results)
        report = {
            "summary": {
                "passed": passed,
                "total": len(self.results),
                "pass_rate": passed / len(self.results) if self.results else 0.0,
                "verdict": "PASS" if passed == len(self.results) and gate["proceed"] else "FAIL",
            },
            "gate": gate,
            "tests": {r.name: r.to_dict() for r in self.results},
        }
        return report

    def print_report(self) -> None:
        rep = self.generate_report()
        s = rep["summary"]
        print("=" * 64)
        print("   COG EPISTEMIC VALIDATION REPORT")
        print("=" * 64)
        print(f"   Summary: {s['passed']}/{s['total']} tests passed")
        print(f"   Verdict: {s['verdict']}")
        print("-" * 64)
        print("   Test Results:")
        for name, t in rep["tests"].items():
            mark = "OK " if t["passed"] else "FAIL"
            m = ", ".join(f"{k}={v}" for k, v in t["metrics"].items())
            print(f"   [{mark}] {name:28s} {m}")
            if t["detail"]:
                print(f"         detail: {t['detail']}")
        print("-" * 64)
        print("   Gate checks:")
        for k, v in rep["gate"]["checks"].items():
            print(f"     {'PASS' if v else 'FAIL'}  {k}")
        if rep["gate"]["proceed"]:
            print("   Recommendation: Belief Engine validated. Safe to proceed.")
        else:
            print("   Recommendation: DO NOT proceed. Fix failing guarantees first.")
        print("=" * 64)


def main() -> int:
    harness = EpistemicTestHarness()
    harness.run_all()
    harness.print_report()
    gate = evaluate_gate(harness.results)
    return 0 if gate["proceed"] else 1


if __name__ == "__main__":
    sys.exit(main())
