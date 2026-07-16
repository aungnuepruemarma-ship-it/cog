"""Test 5: Evidence efficiency (v0.1 epistemic metrics).

Replaces the premature "learning velocity" with what v0.1 can honestly measure:
given a healthy evidence set, does the engine turn evidence into CORRECT beliefs
efficiently, with zero false promotions and 100% evidence validation?

Metrics: candidate_rate, active_rate, false_active_count (must be 0),
evidence_per_active_belief, evidence_validation_rate (must be 1.0).
"""

from __future__ import annotations

from cog.evaluation.epistemic_suite.report import TestResult
from cog.evaluation.epistemic_suite.tests._helpers import (
    fresh_dir, run_engine_with_tester, active_beliefs, _SupportTester,
)
from cog.evaluation.infra.generators import gen_block
from cog.experience.store import ExperienceStore
from cog.learning.belief.model import BeliefState


def test_evidence_efficiency() -> TestResult:
    tmp = fresh_dir()
    store = ExperienceStore(tmp / "exp")

    # A clean, well-separated signal: preflight helps for docker_build.
    for e in gen_block(40, tool="docker_build", domain="software",
                       with_preflight=False, failed=True,
                       category="dependency_failure", start=0):
        store.add(e)
    for e in gen_block(40, tool="docker_build", domain="software",
                       with_preflight=True, failed=False,
                       category="dependency_failure", start=100):
        store.add(e)

    n_evidence = store.count()
    cases = run_engine_with_tester(store, tmp / "beliefs.db", _SupportTester())

    active = [c for c in cases if c.final_state == BeliefState.ACTIVE]
    false_active = sum(1 for c in cases
                       if c.final_state in (BeliefState.ACTIVE, BeliefState.SUPPORTED)
                       and "docker_build" not in c.belief.claim.condition.get("task", ""))

    # Evidence validation: every stored experience must have passed validate().
    invalid = 0
    for row in store.filter(domain="software"):
        exp = store.get(row["id"])
        if not exp.is_valid():
            invalid += 1
    validation_rate = 1.0 - (invalid / n_evidence if n_evidence else 0.0)

    evidence_per_active = (n_evidence / len(active)) if active else float("inf")

    return TestResult(
        name="evidence_efficiency",
        passed=(false_active == 0 and validation_rate >= 1.0 and len(active) >= 1),
        metrics={
            "evidence": n_evidence,
            "candidates": len(cases),
            "active": len(active),
            "false_active": false_active,
            "evidence_per_active_belief": round(evidence_per_active, 2) if active else None,
            "validation_rate": round(validation_rate, 4),
        },
        detail=f"active={len(active)}, false_active={false_active}, validation_rate={validation_rate:.2%}",
    )
