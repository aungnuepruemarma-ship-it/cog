"""Learning benchmark: epistemic quality of the BELIEF layer only.

A belief is a PREDICTIVE statement about the world ("under condition C, failures
are likely"), NOT a prescriptive intervention. This module therefore evaluates
beliefs against OBSERVATIONS, never against intervention labels.

We use the hidden truth labels only to (a) define the held-out eval split and
(b) optionally supply world_state for a genuine false_belief_rate. Intervention
labels (effective_interventions) are deliberately NOT consumed here -- that is
the runtime benchmark's job (see runtime/).

Reports (learner-only metrics):
    candidate_beliefs, active_beliefs, sample_efficiency,
    belief_accuracy, belief_precision, belief_recall, calibration_ece,
    belief_retention_rate, false_belief_rate (None unless world_state labels exist)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import mkdtemp

from cog.evaluation.infra.generators import GeneratedDataset, HiddenTruth, make_experience
from cog.evaluation.learning.calibration import calibration_ece
from cog.experience.store import ExperienceStore
from cog.learning.belief.engine import BeliefEngine
from cog.learning.belief.model import Belief, BeliefState
from cog.learning.belief.store import BeliefStore
from cog.learning.belief.synthesis import synthesize


@dataclass
class LearningReport:
    candidate_beliefs: int = 0
    active_beliefs: int = 0
    sample_efficiency: float = 0.0
    belief_accuracy: float | None = None
    belief_precision: float | None = None
    belief_recall: float | None = None
    calibration_ece: float | None = None
    belief_retention_rate: float | None = None
    false_belief_rate: float | None = None  # None unless world_state labels exist

    def to_dict(self) -> dict:
        return {
            "candidate_beliefs": self.candidate_beliefs,
            "active_beliefs": self.active_beliefs,
            "sample_efficiency": self.sample_efficiency,
            "belief_accuracy": self.belief_accuracy,
            "belief_precision": self.belief_precision,
            "belief_recall": self.belief_recall,
            "calibration_ece": self.calibration_ece,
            "belief_retention_rate": self.belief_retention_rate,
            "false_belief_rate": self.false_belief_rate,
        }


def _run_belief_engine(train_exps: list, belief_db: Path) -> list[Belief]:
    store = ExperienceStore(belief_db.parent / "train_exp")
    for e in train_exps:
        store.add(e)
    store_beliefs = BeliefStore(belief_db)
    engine = BeliefEngine(store_beliefs, store)
    # Real promotion path via the engine's internal discriminator.
    cases = engine.run(min_evidence=10)
    return [c.belief for c in cases if c.belief.state == BeliefState.ACTIVE]


def _preflight_absent(exp) -> bool:
    tools = [s.get("tool", "") for s in (exp.execution or [])]
    return not any("preflight" in t or "inspect" in t or "check" in t for t in tools)


def _belief_matches(exp, belief: Belief) -> bool:
    """True only for experiences inside the belief's SUPPORT: same domain,
    same failed tool (task_type), and preflight absent (the belief's condition).
    Evaluating outside the support would unfairly penalize a belief for
    situations it never claimed to describe."""
    if exp.domain != belief.scope.domain:
        return False
    if belief.scope.task_type and exp.execution:
        tools = [s.get("tool", "") for s in exp.execution]
        if belief.scope.task_type not in tools and exp.causal.failure_node != belief.scope.task_type:
            return False
    return _preflight_absent(exp)  # belief condition = preflight absent


def _belief_prediction_stats(belief: Belief, eval_exps: list,
                             threshold: float = 0.5) -> tuple[int, int, int, int]:
    """TP, FP, TN, FN of the belief's failure prediction on eval experiences
    matching its condition. Predicted failure = failure_probability >= threshold
    (classification view, documented default 0.5). Calibration (probabilistic)
    is computed SEPARATELY in calibration_ece and must not be conflated here."""
    tp = fp = tn = fn = 0
    pred_fail = (belief.claim.prediction.get("failure_probability", 0.0) >= threshold)
    for e in eval_exps:
        if not _belief_matches(e, belief):
            continue
        obs_fail = (e.outcome == "failure")
        if pred_fail and obs_fail:
            tp += 1
        elif pred_fail and not obs_fail:
            fp += 1
        elif (not pred_fail) and obs_fail:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def _retention_rate(active: list[Belief], train: list, seed: int,
                   noise_frac: float = 0.05) -> float | None:
    """Deterministic STABILITY check (benign-noise variant).

    The engine re-synthesizes from scratch every run and a belief must re-earn
    ACTIVE via a fresh discriminator experiment. So a belief that drops under
    DIRECT contradiction is CORRECT behavior, not fragility. Real fragility is:
    the belief drops when faced with BENIGN noise that does NOT reverse the
    underlying pattern.

    This injects benign noise -- flips `noise_frac` of the preflight-absent
    FAILURES to SUCCESS (still a dominant failure pattern) -- and measures the
    fraction of previously-active belief IDs that remain ACTIVE.

        retention = still_active_after_benign_noise / previously_active

    A well-behaved learner stays stable under benign noise (retention -> 1.0)
    while still dropping on genuine contradiction (covered by the correctness
    suite's contradiction_detection gate).
    """
    if not active:
        return None
    prev_ids = {b.id for b in active}
    db = Path(mkdtemp(prefix="retain_")) / "beliefs.db"
    store = ExperienceStore(db.parent / "train_exp")
    # copy train, flipping a deterministic minority of preflight-absent failures
    flipped = 0
    n = len(train)
    for i, e in enumerate(train):
        if (e.outcome == "failure" and not _has_preflight(e)
                and (i % 100) < int(noise_frac * 100)):
            store.add(_flip_outcome(e, "success"))
            flipped += 1
        else:
            store.add(e)
    bstore = BeliefStore(db)
    engine = BeliefEngine(bstore, store)
    post_cases = engine.run(min_evidence=10)
    # The engine re-synthesizes from scratch each run (stateless across runs),
    # so belief IDs differ. Compare by CONDITION, not ID: a belief is "retained"
    # if an ACTIVE belief covering the same condition still exists after benign
    # noise. This measures stability under noise, independent of ID churn.
    def _cond(b: Belief) -> frozenset:
        return frozenset(b.claim.condition.items())
    prev_conditions = {_cond(b) for b in active}
    post_active = {_cond(c.belief) for c in post_cases
                   if c.belief.state == BeliefState.ACTIVE}
    retained = sum(1 for c in prev_conditions if c in post_active)
    return round(retained / len(prev_conditions), 3)


def _has_preflight(exp) -> bool:
    return any("preflight" in s.get("tool", "") for s in (exp.execution or []))


def _flip_outcome(exp, new_outcome: str):
    from cog.experience.record import FailureInfo
    failed = new_outcome == "failure"
    failure = FailureInfo(category="dependency_failure",
                          error_signature="missing_package") if failed else FailureInfo()
    return exp.__class__(
        id=exp.id, task_id=exp.task_id, goal=exp.goal, purpose=exp.purpose,
        domain=exp.domain, difficulty=exp.difficulty, constraints=exp.constraints,
        success_criteria=exp.success_criteria, context=exp.context,
        reality_delta=exp.reality_delta, workspace=exp.workspace, reasoning=exp.reasoning,
        execution=exp.execution,
        verification={"verified": not failed, "confidence": 0.0 if failed else 0.95},
        metrics=exp.metrics, failure=failure, causal=exp.causal,
        replay=exp.replay, outcome=new_outcome,
    )


def evaluate_learning(dataset: GeneratedDataset) -> LearningReport:
    split = len(dataset.experiences) - (len(dataset.experiences) // 5)  # last 20% = eval
    train, eval_split = dataset.experiences[:split], dataset.experiences[split:]
    db = Path(mkdtemp(prefix="learn_eval_")) / "beliefs.db"
    active = _run_belief_engine(train, db)

    # --- prediction vs observation (genuine belief metrics) --- #
    tp = fp = tn = fn = 0
    for b in active:
        t, f, n, mis = _belief_prediction_stats(b, eval_split)
        tp += t; fp += f; tn += n; fn += mis
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None

    # --- calibration on held-out eval --- #
    cal = calibration_ece(active, eval_split)
    ece = cal.expected_calibration_error if cal.bins else None

    # --- sample efficiency: experiences per candidate belief --- #
    cand = max(len(synthesize(ExperienceStore(db.parent / "train_exp")).candidates), 1)
    sample_eff = len(train) / cand if cand else 0.0

    # --- false_belief_rate: only if world_state labels exist --- #
    fbr: float | None = None
    if dataset.labels and any(getattr(v, "world_state", None) for v in dataset.labels.values()):
        # world_state present -> compare belief's claimed condition to ground truth
        pass  # v0.1 dataset has no world_state; left None by design

    # --- retention: re-run with deterministic contradictory evidence --- #
    retention = _retention_rate(active, train, seed=42)

    return LearningReport(
        candidate_beliefs=cand,
        active_beliefs=len(active),
        sample_efficiency=round(sample_eff, 3),
        belief_accuracy=accuracy,
        belief_precision=precision,
        belief_recall=recall,
        calibration_ece=ece,
        belief_retention_rate=retention,
        false_belief_rate=fbr,
    )
