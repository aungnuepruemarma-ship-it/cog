"""Shared helpers for the epistemic validation suite (v0.1 scope)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from cog.experience.store import ExperienceStore
from cog.learning.belief.store import BeliefStore
from cog.learning.belief.engine import BeliefEngine
from cog.learning.belief.model import BeliefState


def fresh_dir(prefix: str = "episuite_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))


def run_engine(experience_store: ExperienceStore, belief_db: Path) -> list:
    """Run the real BeliefEngine over an experience store; return BeliefCases."""
    beliefs = BeliefStore(belief_db)
    engine = BeliefEngine(beliefs, experience_store)
    return engine.run(scope_domain="software", min_evidence=10)


def run_engine_with_tester(experience_store: ExperienceStore, belief_db: Path,
                           tester, scope_domain: str | None = None) -> list:
    """Run the engine with a caller-supplied tester (e.g. a stub returning
    SUPPORT/CHALLENGE). Used by suite tests that isolate a specific guarantee
    (scope precision, evidence efficiency) without coupling to the engine's
    internal replay mechanics. scope_domain=None means no domain filter."""
    beliefs = BeliefStore(belief_db)
    engine = BeliefEngine(beliefs, experience_store, tester=tester)
    return engine.run(scope_domain=scope_domain, min_evidence=10)


class _SupportTester:
    """Stub tester: always returns SUPPORT so the engine's real promotion loop
    runs end-to-end. Used to test synthesis/scope, not statistics."""
    def __init__(self, *a, **k):
        pass
    def run(self, belief):
        from cog.learning.belief.testing import ExperimentResult
        return ExperimentResult(
            belief_id=belief.id,
            group_a_n=15, group_a_failure_rate=0.8,
            group_b_n=15, group_b_failure_rate=0.1,
            lift=0.6, ci95=(0.2, 1.0), verdict="SUPPORT",
        )


def active_beliefs(belief_db: Path) -> list:
    bs = BeliefStore(belief_db)
    return bs.by_state(BeliefState.ACTIVE) + bs.by_state(BeliefState.SUPPORTED)


def any_active_with(belief_db: Path, cond_key: str, cond_val: str) -> bool:
    for b in active_beliefs(belief_db):
        if b.claim.condition.get(cond_key) == cond_val:
            return True
    return False
