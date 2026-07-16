"""Governance-v0.2 invariant suites (COG_LEARNING_GOVERNANCE.md section 9, step 3F).

Four invariants the governance layer must hold:
  1. Promotion correctness  -- more evidence raises promotion probability
  2. Diversity protection    -- 10 independent > 100 repeated
  3. Contradiction recovery  -- new contradicting evidence is not silently overwritten
  4. Decay behavior          -- old unused belief loses privilege (never deleted)

Run: python -m cog.learning.belief.tests.test_governance
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from cog.learning.belief.consolidation.decay import tiered_decay
from cog.learning.belief.model import (
    Belief,
    BeliefClaim,
    BeliefScope,
    BeliefState,
)
from cog.learning.belief.policy import (
    ConfidencePolicy,
    PromotionScore,
    PromotionSignals,
    is_broad,
)

NOW = datetime.now(timezone.utc)


def _belief(statement_text: str = "", scope_domain: str = "software",
            task_type: str = "docker_build", confidence: float = 0.8,
            state: BeliefState = BeliefState.ACTIVE,
            condition: dict | None = None, prediction: dict | None = None) -> Belief:
    cond = condition if condition is not None else {"task": task_type, "preflight": False}
    pred = prediction if prediction is not None else {"failure_probability": 0.9}
    b = Belief(
        id="bel_test",
        claim=BeliefClaim(condition=cond, prediction=pred),
        scope=BeliefScope(domain=scope_domain, task_type=task_type, environment="default"),
        confidence=confidence,
        state=state,
    )
    # inject a statement override by monkeypatching .statement for textual checks
    if statement_text:
        b.statement = lambda: statement_text  # type: ignore[method-assign]
    return b


# --------------------------------------------------------------------------- #
# 1. Promotion correctness
# --------------------------------------------------------------------------- #
def test_promotion_correctness() -> None:
    base = PromotionSignals(
        confidence=0.6, evidence_quantity=PromotionScore.normalize_evidence_quantity(10),
        evidence_diversity=0.5, cross_session_consistency=0.5,
        runtime_success=0.6, contradiction_penalty=0.0,
    )
    low = PromotionScore.calculate(base)
    more = PromotionSignals(
        confidence=base.confidence, evidence_quantity=PromotionScore.normalize_evidence_quantity(40),
        evidence_diversity=base.evidence_diversity, cross_session_consistency=base.cross_session_consistency,
        runtime_success=base.runtime_success, contradiction_penalty=base.contradiction_penalty,
    )
    high = PromotionScore.calculate(more)
    assert high > low, f"more evidence should raise score: {low} -> {high}"
    # and the decision should reflect it (higher score -> not lower privilege)
    _RANK = {BeliefState.TESTING: 0, BeliefState.SUPPORTED: 1, BeliefState.ACTIVE: 2,
             BeliefState.PENDING_REVIEW: 1}
    low_dec = ConfidencePolicy.decide_state(low, _belief())
    high_dec = ConfidencePolicy.decide_state(high, _belief())
    assert _RANK[high_dec] >= _RANK[low_dec], \
        f"higher score must not reduce privilege: {low_dec.value}({_RANK[low_dec]}) vs {high_dec.value}({_RANK[high_dec]})"


# --------------------------------------------------------------------------- #
# 2. Diversity protection
# --------------------------------------------------------------------------- #
def test_diversity_protection() -> None:
    # 100 repeated observations from ONE scenario: high qty, low diversity
    repeated = PromotionSignals(
        confidence=0.6, evidence_quantity=PromotionScore.normalize_evidence_quantity(100),
        evidence_diversity=0.05, cross_session_consistency=0.05,
        runtime_success=0.6, contradiction_penalty=0.0,
    )
    # 10 independent observations across scenarios: lower qty, high diversity
    independent = PromotionSignals(
        confidence=0.6, evidence_quantity=PromotionScore.normalize_evidence_quantity(10),
        evidence_diversity=0.95, cross_session_consistency=0.95,
        runtime_success=0.6, contradiction_penalty=0.0,
    )
    s_rep = PromotionScore.calculate(repeated)
    s_ind = PromotionScore.calculate(independent)
    assert s_ind > s_rep, f"diversity must beat repetition: repeated={s_rep} independent={s_ind}"


# --------------------------------------------------------------------------- #
# 3. Contradiction recovery
# --------------------------------------------------------------------------- #
def test_contradiction_recovery_policy() -> None:
    bel = _belief(statement_text="use urllib for HTTP calls")
    # New strong contradicting evidence arrives -> must NOT silently promote/overwrite.
    decision = ConfidencePolicy.decide_state(0.95, bel, strong_conflict=True)
    assert decision == BeliefState.PENDING_REVIEW, f"expected PENDING_REVIEW, got {decision}"
    # And a broad rule also routes to review even at high score.
    broad = _belief(statement_text="always use urllib for all Python networking")
    assert is_broad(broad) is True
    assert ConfidencePolicy.decide_state(0.99, broad) == BeliefState.PENDING_REVIEW
    # While a narrow, non-conflicting high score belief promotes to ACTIVE.
    narrow = _belief(statement_text="use urllib inside Termux")
    assert is_broad(narrow) is False
    assert ConfidencePolicy.decide_state(0.95, narrow) == BeliefState.ACTIVE


# --------------------------------------------------------------------------- #
# 4. Decay behavior
# --------------------------------------------------------------------------- #
def test_decay_behavior() -> None:
    # old, unused, low-confidence ACTIVE belief -> loses privilege
    old_low = _belief(confidence=0.5, state=BeliefState.ACTIVE)
    old_low.last_used = (NOW - timedelta(days=30)).isoformat()
    old_low.confirmation_count = 0
    assert tiered_decay(old_low, now=NOW) in (BeliefState.TESTING, BeliefState.CHALLENGED)

    # old, unused, high-confidence (>0.95) ACTIVE belief -> extremely slow, stays
    old_high = _belief(confidence=0.98, state=BeliefState.ACTIVE)
    old_high.last_used = (NOW - timedelta(days=60)).isoformat()
    old_high.confirmation_count = 5
    assert tiered_decay(old_high, now=NOW) == BeliefState.ACTIVE

    # heavily contradicted belief -> fast-tracked to CHALLENGED (never deleted)
    contradicted = _belief(confidence=0.9, state=BeliefState.ACTIVE)
    contradicted.contradiction_count = 5
    assert tiered_decay(contradicted, now=NOW) == BeliefState.CHALLENGED

    # decay never returns RETIRED
    for conf in (0.1, 0.5, 0.9, 0.99):
        b = _belief(confidence=conf, state=BeliefState.ACTIVE)
        b.last_used = (NOW - timedelta(days=400)).isoformat()
        b.confirmation_count = 0
        assert tiered_decay(b, now=NOW) != BeliefState.RETIRED


def _run() -> int:
    tests = [
        test_promotion_correctness,
        test_diversity_protection,
        test_contradiction_recovery_policy,
        test_decay_behavior,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  [OK ] {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {t.__name__}: {e}")
    print(f"Governance invariants: {len(tests)-failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if _run() else 0)
