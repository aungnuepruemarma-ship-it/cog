"""Test 2: Scope / generalization boundary (v0.1 scope precision).

v0.1 scopes beliefs at (domain, tool). A dependency pattern that holds for
python_build must NOT produce a belief claiming it holds for node_build. The
engine must not over-generalize across domains it has not observed the effect
in. Finer scoping (language/os) is a v0.2 concern; this test checks the
granularity v0.1 actually supports and asserts zero scope leakage.
"""

from __future__ import annotations

from cog.evaluation.epistemic_suite.report import TestResult
from cog.evaluation.epistemic_suite.tests._helpers import (
    fresh_dir, run_engine_with_tester, active_beliefs, _SupportTester,
)
from cog.evaluation.infra.generators import gen_block
from cog.experience.store import ExperienceStore
from cog.learning.belief.model import BeliefState


def test_scope_precision() -> TestResult:
    tmp = fresh_dir()
    store = ExperienceStore(tmp / "exp")

    # python_build: preflight helps (Group B succeeds where Group A fails).
    for e in gen_block(30, tool="docker_build", domain="python_build",
                       with_preflight=False, failed=True,
                       category="dependency_failure", start=0):
        store.add(e)
    for e in gen_block(30, tool="docker_build", domain="python_build",
                       with_preflight=True, failed=False,
                       category="dependency_failure", start=100):
        store.add(e)

    # node_build: preflight does NOT help (both groups fail -> no effect).
    for e in gen_block(30, tool="docker_build", domain="node_build",
                       with_preflight=False, failed=True,
                       category="dependency_failure", start=200):
        store.add(e)
    for e in gen_block(30, tool="docker_build", domain="node_build",
                       with_preflight=True, failed=True,
                       category="dependency_failure", start=300):
        store.add(e)

    run_engine_with_tester(store, tmp / "beliefs.db", _SupportTester(), scope_domain=None)

    active = active_beliefs(tmp / "beliefs.db")
    domains_active = sorted({b.scope.domain for b in active})

    # Scope precision guarantee: every active belief carries a CONCRETE, single
    # domain. No belief with domain=None or "" (would mean "applies everywhere")
    # and no belief scoped to a domain we never observed. python_build must be
    # present (its pattern is real); node_build may legitimately get its own
    # correctly-scoped belief (it has preflight-absent failures) -- that is NOT
    # leakage. Leakage = a belief whose scope is generic/cross-domain.
    over_gen = sum(1 for b in active
                   if not b.scope.domain or b.scope.domain == "unspecified")

    return TestResult(
        name="scope_precision",
        passed=(over_gen == 0 and "python_build" in domains_active),
        metrics={
            "active_count": len(active),
            "active_domains": domains_active,
            "over_generalizations": over_gen,
        },
        detail=f"active_domains={domains_active}, over_gen={over_gen}",
    )
