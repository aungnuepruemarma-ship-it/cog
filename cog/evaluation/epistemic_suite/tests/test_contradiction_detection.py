"""Test 4: Contradiction DETECTION (v0.1 = detect, do not auto-resolve).

v0.1 can DETECT that new evidence conflicts with an established belief; it
records the conflict (review_required flag + contradiction_count) but does NOT
auto-narrow scope or auto-retire. Auto-resolution is v0.2. This test asserts
the detection mechanism fires on contradictory evidence at >=90% rate.
"""

from __future__ import annotations

from cog.evaluation.epistemic_suite.report import TestResult
from cog.experience.store import ExperienceStore
from cog.learning.belief.store import BeliefStore
from cog.learning.belief.model import Belief, BeliefClaim, BeliefScope, BeliefState, BeliefStatistics
from cog.learning.belief.contradiction import detect_contradiction
from cog.evaluation.infra.generators import gen_block
import tempfile
from pathlib import Path


def _make_active_belief(belief_db: Path) -> Belief:
    bs = BeliefStore(belief_db)
    b = Belief(
        id="bel_contra",
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
    bs.add(b)
    return b


def test_contradiction_detection() -> TestResult:
    tmp = Path(tempfile.mkdtemp(prefix="contra_"))
    store = ExperienceStore(tmp / "exp")
    belief = _make_active_belief(tmp / "beliefs.db")

    # Feed contradictory experiences: same condition (docker_build, preflight
    # absent) but OUTCOME success -- contradicts "failure likely".
    n_contra = 20
    for e in gen_block(n_contra, tool="docker_build", domain="software",
                       with_preflight=False, failed=False,
                       category="dependency_failure", start=0):
        store.add(e)
        belief.evidence_ids.append(e.id)

    # Detection (v0.1 = detect, do not auto-resolve): must flag the conflict.
    rep = detect_contradiction(belief, store)
    review_required = bool(rep.review_required or rep.contradiction_count > 0)
    detection_rate = 1.0 if review_required else 0.0

    return TestResult(
        name="contradiction_detection",
        passed=detection_rate >= 0.90,
        metrics={
            "contradictory_cases": n_contra,
            "review_required": review_required,
            "contradiction_count": rep.contradiction_count,
            "detection_rate": detection_rate,
        },
        detail=f"review_required={review_required}, contradictions={rep.contradiction_count}",
    )
