"""Test 1: False-pattern resistance (v0.1 discriminator guarantee).

Correlation is not enough; the intervention must change outcomes. If the
failure occurs with AND without the candidate condition (hidden cause), the
discriminator finds Group B (intervention present) also fails -> lift ~= 0 ->
INSUFFICIENT -> held in TESTING. No ACTIVE belief about that condition.

This validates the core epistemic rule: "intervention must change outcomes."
"""

from __future__ import annotations

from cog.evaluation.epistemic_suite.report import TestResult
from cog.evaluation.epistemic_suite.tests._helpers import (
    fresh_dir, run_engine, any_active_with,
)
from cog.evaluation.infra.generators import gen_block
from cog.experience.store import ExperienceStore


def test_false_pattern() -> TestResult:
    tmp = fresh_dir()
    store = ExperienceStore(tmp / "exp")
    # Hidden cause X (dependency_failure) triggers failure regardless of preflight.
    # Group A: docker_build fails WITHOUT preflight.
    for e in gen_block(50, tool="docker_build", domain="software",
                       with_preflight=False, failed=True,
                       category="dependency_failure", start=0):
        store.add(e)
    # Group B: docker_build fails WITH preflight present (so preflight does NOT help).
    for e in gen_block(50, tool="docker_build", domain="software",
                       with_preflight=True, failed=True,
                       category="dependency_failure", start=100):
        store.add(e)

    cases = run_engine(store, tmp / "beliefs.db")
    # No ACTIVE belief should reference docker_build (the spurious condition).
    false_active = 1 if any_active_with(tmp / "beliefs.db", "task", "docker_build") else 0

    return TestResult(
        name="false_pattern",
        passed=false_active == 0,
        metrics={
            "candidates": len(cases),
            "false_active": false_active,
            "note": "Group B (preflight present) also failed -> lift~0 -> INSUFFICIENT",
        },
        detail=f"{len(cases)} candidates, false_active={false_active}",
    )
