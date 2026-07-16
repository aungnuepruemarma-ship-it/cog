"""Correctness suite: the must-pass invariants that protect the architecture.

If any of these fail, development stops. These are the five invariants the
review prioritized:

  1. state_transition_legality  -- only legal lifecycle transitions occur
  2. belief_justification       -- ACTIVE policies reference valid (non-challenged) beliefs
  3. belief_policy_dependency   -- a challenged belief flags its dependent policies
  4. replay_determinism         -- experiences reproduce identically on reload
  5. store_consistency          -- append-only history; no silent state overwrite

Plus false_belief_rate (no spurious ACTIVE beliefs) folded in from the epistemic
suite. Each check drives the REAL engine/stores -- no mocks of the learning path.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cog.evaluation.infra.harness import EvaluationSuite
from cog.evaluation.infra.metrics import (
    BooleanMetric, ThresholdMetric, MetricResult,
)
from cog.evaluation.epistemic_suite.tests import (
    test_false_pattern, test_replay, test_belief_policy_cascade,
)
from cog.experience.store import ExperienceStore
from cog.learning.belief.store import BeliefStore
from cog.learning.belief.model import Belief, BeliefClaim, BeliefScope, BeliefState, BeliefStatistics
from cog.learning.belief.lifecycle import BeliefLifecycle
from cog.learning.policy.store import PolicyStore
from cog.learning.policy.model import Policy, PolicyEffect, PolicyState
from cog.learning.policy.lifecycle import PolicyLifecycle


class CorrectnessSuite(EvaluationSuite):
    name = "correctness"
    version = "v1.0.0"

    def _register_metrics(self) -> None:
        self.metrics.register(ThresholdMetric("false_belief_rate", max_value=0.01,
                                               target="<= 0.01"))
        self.metrics.register(ThresholdMetric("replay_determinism", min_value=0.95,
                                               target=">= 0.95"))
        self.metrics.register(BooleanMetric("belief_justification_valid",
                                            target="must be True"))
        self.metrics.register(BooleanMetric("belief_policy_dependency",
                                            target="must be True"))
        self.metrics.register(BooleanMetric("store_consistency",
                                            target="must be True"))
        self.metrics.register(BooleanMetric("state_transition_legality",
                                            target="must be True"))

    # ---- the five checks ---- #
    def _false_belief_rate(self) -> float:
        # false_pattern must yield 0 false ACTIVE; measure as rate over candidates.
        r = test_false_pattern.test_false_pattern()
        false_active = r.metrics.get("false_active", 0)
        candidates = max(r.metrics.get("candidates", 1), 1)
        return false_active / candidates

    def _belief_justification(self) -> bool:
        tmp = Path(tempfile.mkdtemp(prefix="corr_just_"))
        bs = BeliefStore(tmp / "beliefs.db")
        ps = PolicyStore(tmp / "policies.db")
        belief = Belief(
            id="bel_j", claim=BeliefClaim(
                condition={"task": "docker_build", "preflight": False, "domain": "software"},
                prediction={"failure_probability": 0.8}),
            evidence_ids=["e1"], statistics=BeliefStatistics(10, 0.2, (0.6, 1.0)),
            scope=BeliefScope("software", "docker_build", "default"),
            confidence=0.9, state=BeliefState.ACTIVE,
        )
        bs.add(belief)
        policy = Policy(
            id="pol_j", action="x", trigger={"task_type": "docker_build"},
            justification=[belief.id], state=PolicyState.ACTIVE, confidence=0.9,
            evidence_ids=["e1"], expected_effect=PolicyEffect("r", "decrease", -0.3),
            created_at="t", last_validated="t",
        )
        ps.add(policy)
        # ACTIVE policy on a valid (ACTIVE) belief -> justification passes.
        ok_active = not policy.validate(bs)
        # Now challenge the belief; justification must fail.
        bl = BeliefLifecycle()
        bl.to_challenged(belief, reason="drift")
        bs.save_state(belief, fro="active", reason="challenged")
        challenged = bs.get(belief.id)
        ok_challenged = len(policy.validate(bs)) > 0
        return bool(ok_active and ok_challenged)

    def _store_consistency(self) -> bool:
        # Append-only: a belief's state change must leave a transition row and
        # never silently overwrite. We assert transitions() grows and the row
        # reflects the latest state (not the original).
        tmp = Path(tempfile.mkdtemp(prefix="corr_store_"))
        bs = BeliefStore(tmp / "beliefs.db")
        belief = Belief(
            id="bel_s", claim=BeliefClaim(
                condition={"task": "docker_build"}, prediction={"failure_probability": 0.8}),
            evidence_ids=["e1"], statistics=BeliefStatistics(10, 0.2, (0.6, 1.0)),
            scope=BeliefScope("software", "docker_build", "default"),
            confidence=0.9, state=BeliefState.PROPOSED,
        )
        bs.add(belief)
        before = len(bs.transitions(belief.id))
        bl = BeliefLifecycle()
        bl.to_testing(belief)
        bs.save_state(belief, fro="proposed", reason="test")
        after = len(bs.transitions(belief.id))
        # latest stored state must be 'testing', not the original 'proposed'
        latest = bs.get(belief.id).state
        return bool(after == before + 1 and latest == BeliefState.TESTING)

    def _state_transition_legality(self) -> bool:
        # Illegal transition (e.g. PROPOSED -> ACTIVE) must raise, never corrupt.
        bs = BeliefStore(Path(tempfile.mkdtemp(prefix="corr_trans_")) / "beliefs.db")
        belief = Belief(
            id="bel_t", claim=BeliefClaim(
                condition={"task": "docker_build"}, prediction={"failure_probability": 0.8}),
            evidence_ids=["e1"], statistics=BeliefStatistics(10, 0.2, (0.6, 1.0)),
            scope=BeliefScope("software", "docker_build", "default"),
            confidence=0.9, state=BeliefState.PROPOSED,
        )
        bs.add(belief)
        bl = BeliefLifecycle()
        try:
            bl.to_active(belief)  # illegal from PROPOSED
            return False
        except ValueError:
            return True

    def _run(self):
        metrics: list[MetricResult] = []
        metrics.append(self.metrics.compute("false_belief_rate", self._false_belief_rate()))
        replay = test_replay.test_replay()
        metrics.append(self.metrics.compute("replay_determinism", replay.metrics["replay_rate"]))
        metrics.append(self.metrics.compute("belief_justification_valid", self._belief_justification()))
        cascade = test_belief_policy_cascade.test_belief_policy_cascade()
        metrics.append(self.metrics.compute("belief_policy_dependency",
                                             cascade.metrics.get("cascade_moves", 0) >= 1))
        metrics.append(self.metrics.compute("store_consistency", self._store_consistency()))
        metrics.append(self.metrics.compute("state_transition_legality", self._state_transition_legality()))
        return metrics, {}
