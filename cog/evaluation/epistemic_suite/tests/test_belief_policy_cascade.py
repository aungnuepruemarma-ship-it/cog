"""Test 7: Belief->Policy cascade + drift (folded regression guarantee).

When an ACTIVE belief is challenged, every dependent ACTIVE policy must move to
REVIEW_REQUIRED (not silently retired -- a challenged belief may only need a
narrower scope). After the cascade, the policy must no longer be selected as
ACTIVE, so stale policies cannot keep driving execution.

This is the drift/regression guarantee: beliefs and policies stay consistent
over time, and a challenged belief cannot leave a zombie policy in control.
"""

from __future__ import annotations

from cog.evaluation.epistemic_suite.report import TestResult
from cog.learning.belief.store import BeliefStore
from cog.learning.belief.model import Belief, BeliefClaim, BeliefScope, BeliefState, BeliefStatistics
from cog.learning.policy.store import PolicyStore
from cog.learning.policy.model import Policy, PolicyEffect, PolicyState
from cog.learning.policy.lifecycle import PolicyLifecycle
from cog.learning.policy.selector import select_policies
from cog.learning.belief.lifecycle import BeliefLifecycle
import tempfile
from pathlib import Path


def test_belief_policy_cascade() -> TestResult:
    tmp = Path(tempfile.mkdtemp(prefix="cascade_"))
    bs = BeliefStore(tmp / "beliefs.db")
    ps = PolicyStore(tmp / "policies.db")

    # ACTIVE belief.
    belief = Belief(
        id="bel_cascade",
        claim=BeliefClaim(
            condition={"task": "docker_build", "preflight": False, "domain": "software"},
            prediction={"failure_probability": 0.85, "category": "dependency_failure"},
        ),
        evidence_ids=[f"e{i}" for i in range(30)],
        statistics=BeliefStatistics(sample_size=30, success_rate=0.15,
                                    confidence_interval=(0.7, 1.0)),
        scope=BeliefScope(domain="software", task_type="docker_build", environment="default"),
        confidence=0.9,
        state=BeliefState.ACTIVE,
    )
    bs.add(belief)

    # ACTIVE policy justified by that belief.
    policy = Policy(
        id="pol_cascade",
        action="run dependency inspection before build",
        trigger={"task_type": "docker_build", "tool": "docker_build", "domain": "software"},
        justification=[belief.id],
        state=PolicyState.ACTIVE,
        confidence=0.9,
        evidence_ids=["e1"],
        expected_effect=PolicyEffect("docker_build_failure_rate", "decrease", -0.4),
        created_at="t", last_validated="t",
    )
    ps.add(policy)

    # Before challenge: policy is selectable as ACTIVE.
    before = select_policies({"task_type": "docker_build", "tools": {"docker_build"},
                              "domain": "software"}, ps)
    selectable_before = any(p.id == policy.id for p in before)

    # Challenge the belief (environment changed; preflight no longer helps).
    lifecycle = PolicyLifecycle(ps, bs)
    blifecycle = BeliefLifecycle()
    blifecycle.to_challenged(belief, by_belief_id="bel_new_env",
                             reason="new env: preflight no longer reduces failures")
    bs.save_state(belief, fro="active", reason="challenged")

    # Cascade: dependent ACTIVE policy -> REVIEW_REQUIRED.
    moves = lifecycle.on_belief_challenged(belief.id)
    policy = ps.get(policy.id)
    after_state = policy.state

    # After cascade: the policy must NOT be selected as ACTIVE.
    after = select_policies({"task_type": "docker_build", "tools": {"docker_build"},
                             "domain": "software"}, ps)
    selectable_after = any(p.id == policy.id for p in after)

    correct = (
        selectable_before
        and after_state == PolicyState.REVIEW_REQUIRED
        and not selectable_after
        and len(moves) >= 1
    )
    return TestResult(
        name="belief_policy_cascade",
        passed=correct,
        metrics={
            "selectable_before": selectable_before,
            "policy_state_after": after_state.value,
            "cascade_moves": len(moves),
            "selectable_after": selectable_after,
        },
        detail=f"before={selectable_before} after_state={after_state.value} after_selectable={selectable_after}",
    )
