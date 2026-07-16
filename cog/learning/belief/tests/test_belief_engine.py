"""Deterministic end-to-end test of the Belief Engine v0.1 (closed loop).

Proves the review's exit condition:

    From validated experiences, Cog creates candidate beliefs, tests them
    through replay experiments, and promotes only evidence-backed beliefs.

No LLM. Builds real Experience records in an ExperienceStore, runs the
BeliefEngine, and asserts:
  * a candidate belief is discovered from repeated observations,
  * the discriminator experiment runs (Group A vs Group B),
  * the experiment DECIDES the promotion (ACTIVE when lift is real),
  * belief state is immutable history (transitions recorded, never overwritten).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cog.experience.record import (
    CausalGraph,
    Experience,
    ExperienceContext,
    FailureInfo,
    RealityDelta,
    ReplayInfo,
)
from cog.experience.store import ExperienceStore
from cog.learning.belief.engine import BeliefEngine
from cog.learning.belief.model import BeliefState as BState
from cog.learning.belief.store import BeliefStore


def _exp(exp_id: str, *, tool: str, with_preflight: bool, failed: bool,
         category: str = "dependency_failure", domain: str = "software") -> Experience:
    steps = []
    if with_preflight:
        steps.append({"index": 0, "tool": "dep_preflight", "args": {},
                      "result": "ok", "error": None, "duration_s": 1.0})
    steps.append({"index": len(steps), "tool": tool, "args": {},
                  "result": None if failed else "ok",
                  "error": "missing package" if failed else None,
                  "duration_s": 1.0})
    return Experience(
        id=exp_id, task_id="t1", goal="deploy", purpose="",
        domain=domain, difficulty="medium", constraints=[],
        success_criteria=[], context=ExperienceContext(),
        reality_delta=RealityDelta() if not failed else RealityDelta(
            unexpected_conditions=["pkg missing"]),
        workspace={}, reasoning={}, execution=steps,
        verification={"verified": not failed, "confidence": 0.0 if failed else 0.95},
        metrics=__import__("cog.experience.record", fromlist=["ExperienceMetrics"]).ExperienceMetrics(),
        failure=FailureInfo(category=category, error_signature="MISSING_X", failed_step=tool) if failed
        else FailureInfo(),
        causal=CausalGraph(failure_node=tool if failed else None,
                           caused_by=[category] if failed else []),
        replay=ReplayInfo(environment_snapshot="sha256:abc"),
        outcome="failure" if failed else "success",
    )


def test_belief_engine_closed_loop() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        exp_dir = Path(tmp) / "exp"
        bel_dir = Path(tmp) / "bel"
        store = ExperienceStore(exp_dir)

        # Group A: 30 docker_build FAILURES, no preflight.
        for i in range(30):
            store.add(_exp(f"a_{i:03d}", tool="docker_build", with_preflight=False, failed=True))
        # Group B: 30 docker_build SUCCESSES, WITH preflight present.
        for i in range(30):
            store.add(_exp(f"b_{i:03d}", tool="docker_build", with_preflight=True, failed=False))

        beliefs = BeliefStore(bel_dir)
        engine = BeliefEngine(beliefs, store)

        cases = engine.run(scope_domain="software", min_evidence=10)
        assert cases, "engine should discover at least one candidate belief"

        # Exactly the docker_build dependency_failure belief should be promoted.
        promoted = [c for c in cases if c.final_state == BState.ACTIVE]
        assert promoted, f"expected an ACTIVE belief, got states={[c.final_state.value for c in cases]}"
        bel = promoted[0].belief
        assert bel.claim.condition.get("task") == "docker_build"
        assert bel.claim.condition.get("preflight") is False

        # The experiment must show real lift (Group A failed, Group B succeeded).
        exp = promoted[0].experiment
        assert exp.group_a_n == 30 and exp.group_a_failure_rate == 1.0
        assert exp.group_b_n == 30 and exp.group_b_failure_rate == 0.0
        assert exp.lift == 1.0  # 100-point failure-rate reduction
        assert exp.verdict == "SUPPORT"

        # Immutable history: transitions recorded, belief_tests recorded.
        transitions = beliefs.transitions(belief_id=bel.id)
        assert len(transitions) >= 3  # proposed->testing->supported->active
        tests = beliefs.tests(belief_id=bel.id)
        assert len(tests) == 1 and tests[0]["kind"] == "discriminator"

        # Reload from disk: state persisted, history intact.
        reloaded = BeliefStore(bel_dir).get(bel.id)
        assert reloaded is not None and reloaded.state == BState.ACTIVE

    print("test_belief_engine_closed_loop: OK")


def test_insufficient_evidence_holds_belief() -> None:
    """When Group B is absent, the experiment must NOT promote (honest boundary)."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ExperienceStore(Path(tmp) / "exp")
        # Only Group A (failures, no preflight); no Group B exists.
        for i in range(20):
            store.add(_exp(f"a_{i:03d}", tool="docker_build", with_preflight=False, failed=True))
        beliefs = BeliefStore(Path(tmp) / "bel")
        engine = BeliefEngine(beliefs, store)
        cases = engine.run(scope_domain="software", min_evidence=10)
        assert cases, "should still propose a candidate from Group A"
        # No Group B -> INSUFFICIENT -> held in TESTING, never ACTIVE.
        assert all(c.final_state != BState.ACTIVE for c in cases)
        assert any(c.experiment.needs_live_ab for c in cases)
    print("test_insufficient_evidence_holds_belief: OK")


if __name__ == "__main__":
    test_belief_engine_closed_loop()
    test_insufficient_evidence_holds_belief()
    print("\nALL BELIEF-ENGINE TESTS PASSED")
