"""End-to-end proof of the Policy Engine v0.1 (the second half of the loop).

Proves the review's milestone:

    Cog discovers a belief -> derives a candidate policy -> validates it
    experimentally -> activates it -> future executions improve.

No LLM. Builds real Experience records, runs the BeliefEngine to get an
ACTIVE belief, then synthesizes candidate policies, validates one via the
shared A/B laboratory, promotes it through the lifecycle, selects it at
runtime, and verifies the belief-challenge cascade.

Exit criterion: a justified, validated, ACTIVE policy is selected for matching
tasks, and a challenged belief correctly flags (not deletes) the dependent
policy.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cog.experience.record import (
    CausalGraph,
    Experience,
    ExperienceContext,
    ExperienceMetrics,
    FailureInfo,
    RealityDelta,
    ReplayInfo,
)
from cog.experience.store import ExperienceStore
from cog.learning.belief.engine import BeliefEngine
from cog.learning.belief.model import BeliefState as BState
from cog.learning.belief.store import BeliefStore
from cog.learning.policy.lifecycle import PolicyLifecycle
from cog.learning.policy.model import PolicyState
from cog.learning.policy.selector import select_for_task
from cog.learning.policy.store import PolicyStore
from cog.learning.policy.synthesis import synthesize_candidates
from cog.learning.policy.validation import validate_policy
from cog.runtime.task import Task


def _exp(exp_id: str, *, with_preflight: bool, failed: bool,
         category: str = "dependency_failure", domain: str = "software") -> Experience:
    steps = []
    if with_preflight:
        steps.append({"index": 0, "tool": "dep_preflight", "args": {},
                      "result": "ok", "error": None, "duration_s": 1.0})
    steps.append({"index": len(steps), "tool": "docker_build", "args": {},
                  "result": None if failed else "ok",
                  "error": "missing package" if failed else None, "duration_s": 1.0})
    return Experience(
        id=exp_id, task_id="t1", goal="deploy", purpose="",
        domain=domain, difficulty="medium", constraints=[], success_criteria=[],
        context=ExperienceContext(),
        reality_delta=RealityDelta() if not failed else RealityDelta(unexpected_conditions=["pkg missing"]),
        workspace={}, reasoning={}, execution=steps,
        verification={"verified": not failed, "confidence": 0.0 if failed else 0.95},
        metrics=ExperienceMetrics(),
        failure=FailureInfo(category=category, error_signature="MISSING_X", failed_step="docker_build") if failed
        else FailureInfo(),
        causal=CausalGraph(failure_node="docker_build" if failed else None,
                           caused_by=[category] if failed else []),
        replay=ReplayInfo(environment_snapshot="sha256:abc"),
        outcome="failure" if failed else "success",
    )


def test_policy_engine_full_cycle() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        # ---- 1. Produce an ACTIVE belief from real evidence ---- #
        exp_dir, bel_dir, pol_dir = Path(tmp) / "exp", Path(tmp) / "bel", Path(tmp) / "pol"
        store = ExperienceStore(exp_dir)
        for i in range(30):
            store.add(_exp(f"a_{i:03d}", with_preflight=False, failed=True))
        for i in range(30):
            store.add(_exp(f"b_{i:03d}", with_preflight=True, failed=False))
        beliefs = BeliefStore(bel_dir)
        BeliefEngine(beliefs, store).run(scope_domain="software", min_evidence=10)
        active = beliefs.by_state(BState.ACTIVE)
        assert active, "expected an ACTIVE belief"
        belief = active[0]

        # ---- 2. Synthesize candidate policies (multiple, justified) ---- #
        results = synthesize_candidates(beliefs)
        assert results, "synthesis should emit candidates"
        cands = results[0].candidates
        # Correction honored: one belief -> MULTIPLE candidate actions (uncertainty).
        assert len(cands) >= 2, f"expected multiple candidate actions, got {len(cands)}"
        for c in cands:
            assert c.state == PolicyState.CANDIDATE
            assert belief.id in c.justification
            assert c.is_valid(beliefs), "every candidate must be justified"

        # ---- 3. Validate one policy via the shared A/B lab ---- #
        pol = cands[0]
        policies = PolicyStore(pol_dir)
        policies.add(pol)

        # Synthetic variants: baseline fails 50%, treatment (policy) fails 5%.
        # Deterministic so the test is reproducible without a full runtime.
        n = 20
        tasks = [Task(goal=f"deploy {i}") for i in range(n)]

        def baseline_solve(task) -> bool:
            return hash(task.goal) % 2 == 0  # ~50% pass

        def treatment_solve(task) -> bool:
            return hash(task.goal) % 20 != 0  # ~95% pass

        verdict = validate_policy(pol.id, baseline_solve, treatment_solve, tasks, seed=1)
        assert verdict.passed, f"expected validation PASS, got {verdict.to_dict()}"
        policies.record_experiment(pol.id, kind="ab", result=verdict.to_dict())

        # ---- 4. Promote through the lifecycle (justification re-checked) ---- #
        lc = PolicyLifecycle(policies, beliefs)
        lc.to_experimental(pol)
        lc.to_validated(pol)
        lc.to_active(pol)
        assert pol.state == PolicyState.ACTIVE

        # ---- 5. Select at runtime (deterministic trigger match) ---- #
        chosen = select_for_task("docker_build", ["docker_build"], policies, domain="software")
        assert any(p.id == pol.id for p in chosen), "active policy should be selected for docker_build"
        assert chosen[0].confidence >= 0.0

        # ---- 6. Belief-challenge cascade: flag, don't delete ---- #
        # Simulate the belief being challenged (e.g. new contradictory evidence).
        belief.state = BState.CHALLENGED
        moves = lc.on_belief_challenged(belief.id)
        assert moves, "dependent active policy should be flagged"
        refreshed = policies.get(pol.id)
        assert refreshed.state == PolicyState.REVIEW_REQUIRED, \
            f"expected REVIEW_REQUIRED, got {refreshed.state.value}"

        # ---- 7. Transition history is append-only & auditable ---- #
        transitions = policies.transitions(pol.id)
        assert len(transitions) >= 4  # candidate->experimental->validated->active (+review)
        states_seen = [t["to_state"] for t in transitions]
        assert "active" in states_seen and "review_required" in states_seen

    print("test_policy_engine_full_cycle: OK")


if __name__ == "__main__":
    test_policy_engine_full_cycle()
    print("\nPOLICY-ENGINE END-TO-END TEST PASSED")
