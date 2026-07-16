"""Phase 3: the BeliefEngine — the closed cognitive loop orchestrator.

    Act -> Observe -> Explain -> Hypothesize -> Experiment -> Learn -> Act better

Concretely:
    1. synthesize()        scans validated experiences -> PROPOSED candidate
                            beliefs (observation-based, no causal interpretation).
    2. lifecycle.to_testing()  moves each into quarantine (TESTING).
    3. BeliefTester.run()  runs the discriminator experiment (Group A vs B).
    4. verdict drives the transition (the EXPERIMENT decides):
         SUPPORT      -> TESTING -> SUPPORTED -> ACTIVE
         CHALLENGE    -> TESTING -> CHALLENGED -> RETIRED
         INSUFFICIENT -> held in TESTING (not promoted, not retired)
    5. every state change is recorded as an immutable transition + the
       experiment result is stored in belief_tests.

No LLM. The experiment, not assertion, earns promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cog.experience.store import ExperienceStore
from cog.learning.belief.lifecycle import BeliefLifecycle
from cog.learning.belief.model import Belief, BeliefState
from cog.learning.belief.store import BeliefStore
from cog.learning.belief.synthesis import SynthesisResult, synthesize
from cog.learning.belief.testing import BeliefTester, ExperimentResult
from cog.learning.belief.policy import is_broad

@dataclass
class BeliefCase:
    belief: Belief
    experiment: ExperimentResult
    final_state: BeliefState


class BeliefEngine:
    def __init__(self, beliefs: BeliefStore, experiences: ExperienceStore,
                 tester: BeliefTester | None = None) -> None:
        self.beliefs = beliefs
        self.experiences = experiences
        self.tester = tester or BeliefTester(experiences)
        # The engine owns the lifecycle; transitions are persisted via
        # beliefs.save_state (which records an immutable transition row).
        self.lifecycle = BeliefLifecycle()

    def _apply_verdict(self, belief: Belief, exp: ExperimentResult) -> None:
        if exp.verdict == "SUPPORT":
            self.lifecycle.to_supported(belief, reason=f"experiment lift={exp.lift:.3f}")
            self.beliefs.save_state(belief, fro="testing", reason=f"supported lift={exp.lift:.3f}")
            # Governance gate (governance-v0.2): a SUPPORTED belief that is broad
            # (high blast radius) or conflicts with a strong belief must be held
            # for review, NOT auto-promoted to ACTIVE. This is the safety boundary
            # before any autonomous research loop is allowed to modify Cog.
            if is_broad(belief):
                self.lifecycle.to_pending_review(
                    belief, reason="broad rule held for review (governance gate)")
                self.beliefs.save_state(belief, fro="supported", reason="PENDING_REVIEW: broad rule")
                return
            self.lifecycle.to_active(belief, reason="promoted after supported")
            self.beliefs.save_state(belief, fro="supported", reason="promoted to active")
        elif exp.verdict == "CHALLENGE":
            self.lifecycle.to_challenged(belief, reason=f"intervention worsened (lift={exp.lift:.3f})")
            self.beliefs.save_state(belief, fro="testing", reason="challenged")
            self.lifecycle.retire(belief, reason="challenged by experiment")
            self.beliefs.save_state(belief, fro="challenged", reason="retired")
        else:
            # INSUFFICIENT: remain TESTING (held, not promoted, not retired)
            self.beliefs.save_state(belief, fro="testing", reason=f"verdict={exp.verdict}")

    def process_candidate(self, belief: Belief) -> BeliefCase:
        # PROPOSED -> TESTING (enter quarantine)
        self.lifecycle.to_testing(belief, reason="enter quarantine/validation")
        self.beliefs.save_state(belief, fro="proposed", reason="enter quarantine")
        # run the experiment
        result = self.tester.run(belief)
        # persist the experiment record (immutable)
        self.beliefs.record_test(belief.id, kind="discriminator", result=result.to_dict())
        # experiment decides the transition
        self._apply_verdict(belief, result)
        return BeliefCase(belief=belief, experiment=result, final_state=belief.state)

    def run(self, scope_domain: str = "software", min_evidence: int = 10) -> list[BeliefCase]:
        synth_result: SynthesisResult = synthesize(
            self.experiences, scope_domain=scope_domain, min_evidence=min_evidence
        )
        cases: list[BeliefCase] = []
        for belief in synth_result.candidates:
            self.beliefs.add(belief)
            cases.append(self.process_candidate(belief))
        return cases
