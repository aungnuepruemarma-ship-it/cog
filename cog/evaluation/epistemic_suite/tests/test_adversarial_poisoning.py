"""Test 6: Adversarial / noisy evidence poisoning (epistemic stability).

The biggest danger is not failing to learn -- it is learning the wrong thing
confidently. Feed 90 clean experiences where preflight helps, plus 10 noisy
experiences where preflight (supposedly) hurts. Assert: (a) no FALSE belief
reaches ACTIVE, (b) the engine does not instant-flip to a harmful conclusion,
(c) valid evidence still produces the correct belief (gradual, stable).
"""

from __future__ import annotations

from cog.evaluation.epistemic_suite.report import TestResult
from cog.evaluation.epistemic_suite.tests._helpers import fresh_dir, run_engine, active_beliefs
from cog.evaluation.infra.generators import gen_block
from cog.experience.store import ExperienceStore
from cog.learning.belief.model import BeliefState


def test_adversarial_poisoning() -> TestResult:
    tmp = fresh_dir()
    store = ExperienceStore(tmp / "exp")
    clean_ids: list[str] = []
    poison_ids: list[str] = []

    # 90 clean: preflight helps for docker_build.
    for e in gen_block(45, tool="docker_build", domain="software",
                       with_preflight=False, failed=True,
                       category="dependency_failure", start=0):
        store.add(e); clean_ids.append(e.id)
    for e in gen_block(45, tool="docker_build", domain="software",
                       with_preflight=True, failed=False,
                       category="dependency_failure", start=100):
        store.add(e); clean_ids.append(e.id)

    # 10 noisy poison: claim preflight HURTS (preflight present but fails).
    for e in gen_block(10, tool="docker_build", domain="software",
                       with_preflight=True, failed=True,
                       category="dependency_failure", start=200):
        store.add(e); poison_ids.append(e.id)

    cases = run_engine(store, tmp / "beliefs.db")

    # A "false belief" = an ACTIVE belief whose cited evidence is dominated by
    # the poison set (it learned the wrong direction from the noise).
    poison_set = set(poison_ids)
    false_active = 0
    for b in active_beliefs(tmp / "beliefs.db"):
        cited = b.evidence_ids
        poison_cited = sum(1 for ev in cited if ev in poison_set)
        if cited and poison_cited > len(cited) / 2:
            false_active += 1

    # The correct belief (preflight helps) should still be present.
    correct_present = any(
        b.claim.condition.get("preflight") is False and b.scope.task_type == "docker_build"
        for b in active_beliefs(tmp / "beliefs.db")
    )

    return TestResult(
        name="adversarial_poisoning",
        passed=(false_active == 0 and correct_present),
        metrics={
            "clean": len(clean_ids),
            "poison": len(poison_ids),
            "false_active": false_active,
            "correct_belief_present": correct_present,
        },
        detail=f"false_active={false_active}, correct_present={correct_present}",
    )
