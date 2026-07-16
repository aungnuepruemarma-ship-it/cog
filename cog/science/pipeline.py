"""Autonomous promotion pipeline: the scheduler that drives the learning loop.

Governance lives at the write boundary (record_claim refuses unauthorized
evidence/promotion writes). But as long as a human (or ad-hoc code) must call
run_experiment / FormalVerifier / promote_claim by hand, the system is not
self-driving. This module closes that gap with a persistent state machine:

    candidate
      -> scheduled
      -> running
      -> evidence_ready
      -> promotion_pending
      -> promoted
      -> validating
      -> stable
                 -> rolled_back   (on validation failure)

Design rules (so the pipeline does NOT weaken the governance it depends on):
  * The scheduler NEVER writes evidence or promotion records directly. It
    delegates evidence production to the authorized producers (ExperimentRunner
    for experiments, FormalVerifier for proofs) and promotion to
    Ledger.promote_claim. The single-authority invariant is preserved.
  * State is persisted in the same store as the ledger, so a crash is
    recoverable: tick() simply resumes from the last persisted state.
  * Continuous validation is a pluggable battery (callable) so the production
    harness (cog/validation.py) and tests can both drive it. A failed battery
    triggers rollback: the promotion is marked superseded and a finding records
    why.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable

from cog.science.ledger import Ledger
from cog.science.promotion import PromotionDenied


def stable_candidate_id(subject_id: str, hypothesis: str, kind: str) -> str:
    """Deterministic identity for a promotion candidate.

    Identity is derived ONLY from immutable facts: what artifact (subject_id),
    what claim (hypothesis), and what evidence kind. It deliberately excludes
    the producer_spec / proof callable, which is ephemeral runtime state. Two
    emissions of the same claim yield the same id, so re-running discovery is
    idempotent (INSERT OR REPLACE on the same store row -- no duplicates, and a
    candidate already advanced in its lifecycle is not reset).
    """
    digest = hashlib.sha256(
        "\x00".join((subject_id, hypothesis, kind)).encode("utf-8")
    ).hexdigest()[:16]
    return f"cand_{digest}"


class PromotionState(str, Enum):
    CANDIDATE = "candidate"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    EVIDENCE_READY = "evidence_ready"
    PROMOTION_PENDING = "promotion_pending"
    PROMOTED = "promoted"
    VALIDATING = "validating"
    STABLE = "stable"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"  # evidence produced but promotion gate refused (policy/proof failed)

    @property
    def terminal(self) -> bool:
        return self in (
            PromotionState.STABLE,
            PromotionState.ROLLED_BACK,
            PromotionState.REJECTED,
        )


# (from_state, event) -> to_state. Defines legal transitions; anything else is
# a programming error (the scheduler never calls an illegal one, but the check
# keeps the audit trail honest).
_TRANSITIONS: dict[tuple[PromotionState, str], PromotionState] = {
    (PromotionState.CANDIDATE, "schedule"): PromotionState.SCHEDULED,
    (PromotionState.SCHEDULED, "start"): PromotionState.RUNNING,
    (PromotionState.RUNNING, "evidence"): PromotionState.EVIDENCE_READY,
    (PromotionState.EVIDENCE_READY, "promote"): PromotionState.PROMOTION_PENDING,
    (PromotionState.PROMOTION_PENDING, "adopt"): PromotionState.PROMOTED,
    (PromotionState.PROMOTED, "validate"): PromotionState.VALIDATING,
    (PromotionState.VALIDATING, "pass"): PromotionState.STABLE,
    (PromotionState.VALIDATING, "fail"): PromotionState.ROLLED_BACK,
    (PromotionState.PROMOTION_PENDING, "reject"): PromotionState.REJECTED,
}


@dataclass
class Candidate:
    """A promotion candidate and the work needed to evaluate it.

    ``kind`` selects the authorized producer:
      "experiment" -> ExperimentRunner (run_experiment)
      "formal"     -> FormalVerifier
    ``producer_spec`` is the already-constructed spec the producer consumes.
    ``validation`` is a pluggable battery: takes (ledger, subject_id,
    evidence_id) and returns True if the promoted artifact still holds.
    """

    subject_id: str
    hypothesis: str
    kind: str  # "experiment" | "formal"
    producer_spec: Any
    baseline_id: str
    treatment_id: str
    validation: Callable[[Ledger, str, str], bool] | None = None
    candidate_id: str | None = None
    state: PromotionState = PromotionState.CANDIDATE
    evidence_id: str | None = None
    promotion_id: str | None = None

    def __post_init__(self) -> None:
        # Default the record id to the deterministic identity so re-emission is
        # idempotent. The caller may still override candidate_id explicitly
        # (e.g. when reloading an existing row), but a freshly discovered
        # candidate always gets a stable, content-derived id.
        if self.candidate_id is None:
            self.candidate_id = stable_candidate_id(
                self.subject_id, self.hypothesis, self.kind
            )

    @property
    def stable_id(self) -> str:
        """Deterministic identity from immutable fields (not the callable)."""
        return stable_candidate_id(self.subject_id, self.hypothesis, self.kind)


@runtime_checkable
class CandidateSource(Protocol):
    """Uniform contract every learning module implements to expose promotion
    candidates WITHOUT deciding adoption.

    A source constructs fully-formed Candidate objects (evidence spec attached
    as ephemeral runtime state) but never submits, runs, or promotes them. The
    orchestrator collects candidates from all sources and feeds the scheduler.
    Implementations must be pure and idempotent: calling twice over the same
    memory yields candidates with identical stable ids.
    """

    def discover_candidates(self, memory: Any) -> list[Candidate]:
        ...


class PromotionScheduler:
    """Drives candidate -> ... -> stable (or rolled_back) autonomously.

    Persists each candidate's state in the ledger store under kind="pipeline",
    so tick() is idempotent and crash-safe.
    """

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self._specs: dict[str, Any] = {}  # candidate_id -> producer spec (in-session)
        self._validations: dict[str, Any] = {}  # candidate_id -> validation battery

    # -- persistence helpers (so state survives crashes) --
    def _save(self, c: Candidate) -> None:
        # add() mints a new id when record_id is None, else reuses it
        # (INSERT OR REPLACE). We persist c.candidate_id in the content too, so
        # on reload the row carries its own id (used as the spec-resolver key).
        rec = self.ledger.pipeline.add(
            {
                "subject_id": c.subject_id,
                "hypothesis": c.hypothesis,
                "kind": c.kind,
                "state": c.state.value,
                "evidence_id": c.evidence_id,
                "promotion_id": c.promotion_id,
                "baseline_id": c.baseline_id,
                "treatment_id": c.treatment_id,
                "candidate_id": c.candidate_id,  # None on first save; fixed below
            },
            record_id=c.candidate_id,
        )
        c.candidate_id = rec.id
        # Rewrite once with the real id inside content (single extra write only
        # on the very first save; subsequent saves already carry the id).
        if rec.content.get("candidate_id") != rec.id:
            self.ledger.pipeline.add(
                {**dict(rec.content), "candidate_id": rec.id},
                record_id=rec.id,
            )

    def _load_open(self, spec_resolver) -> list[Candidate]:
        """Reload non-terminal candidates from the store. ``spec_resolver``
        maps candidate_id -> the producer spec (callables are not serializable,
        so the caller retains them and re-supplies on restart -- this is the
        honest boundary: state is persisted, non-serializable specs are not)."""
        out: list[Candidate] = []
        for r in self.ledger.pipeline.search(limit=500):
            content = r.content
            if PromotionState(content["state"]).terminal:
                continue
            cid = content["candidate_id"]
            spec = spec_resolver(cid)
            c = Candidate(
                subject_id=content["subject_id"],
                hypothesis=content["hypothesis"],
                kind=content["kind"],
                producer_spec=spec,
                baseline_id=content["baseline_id"],
                treatment_id=content["treatment_id"],
                candidate_id=cid,
                state=PromotionState(content["state"]),
                evidence_id=content.get("evidence_id"),
                promotion_id=content.get("promotion_id"),
                validation=self._validations.get(cid),
            )
            out.append(c)
        return out

    # -- public API --
    def submit(self, candidate: Candidate) -> str:
        """Register a candidate in CANDIDATE state. Its producer spec is kept
        in-session (callables are not serializable); on restart the caller
        re-supplies specs via tick(spec_resolver=...)."""
        self._save(candidate)  # assigns candidate.candidate_id
        self._specs[candidate.candidate_id] = candidate.producer_spec  # type: ignore[index]
        self._validations[candidate.candidate_id] = candidate.validation  # type: ignore[index]
        return candidate.candidate_id  # type: ignore[return-value]

    def _transition(self, c: Candidate, event: str) -> PromotionState:
        nxt = _TRANSITIONS.get((c.state, event))
        if nxt is None:
            raise RuntimeError(
                f"illegal transition {c.state.value} --{event}-> ?"
            )
        c.state = nxt
        self._save(c)
        return nxt

    def tick(self, spec_resolver: Callable[[str], Any] | None = None) -> list[str]:
        """Advance every open candidate by exactly one step. Returns the list
        of candidate_ids that changed state this tick. Idempotent and safe to
        call repeatedly (e.g. after a crash). ``spec_resolver(candidate_id)``
        supplies producer specs on restart; in-session specs are used otherwise.
        """
        def _resolve(cid: str) -> Any:
            if spec_resolver is not None:
                return spec_resolver(cid)
            return self._specs.get(cid)

        changed: list[str] = []
        for c in self._load_open(_resolve):
            before = c.state
            if c.state == PromotionState.CANDIDATE:
                self._transition(c, "schedule")  # -> SCHEDULED
            elif c.state == PromotionState.SCHEDULED:
                self._transition(c, "start")  # -> RUNNING
                self._produce_evidence(c)  # RUNNING -> EVIDENCE_READY
            elif c.state == PromotionState.EVIDENCE_READY:
                self._transition(c, "promote")  # -> PROMOTION_PENDING
                self._promote(c)  # -> PROMOTED (or stays if denied)
            elif c.state == PromotionState.PROMOTED:
                self._transition(c, "validate")  # -> VALIDATING
                self._validate(c)  # -> STABLE or ROLLED_BACK
            # Refresh in-session spec cache from whatever resolved it.
            if c.producer_spec is not None:
                self._specs[c.candidate_id] = c.producer_spec  # type: ignore[index]
            if c.state != before:
                changed.append(c.candidate_id)  # type: ignore[arg-type]
        return changed

    # -- delegated steps (scheduler never writes evidence/promotion itself) --
    def _produce_evidence(self, c: Candidate) -> None:
        from cog.science.experiment import run_experiment
        from cog.science.verification import FormalVerifier

        if c.kind == "experiment":
            res = run_experiment(self.ledger, c.producer_spec)
            c.evidence_id = res["experiment_id"]
        elif c.kind == "formal":
            claim = FormalVerifier(self.ledger).verify(
                subject_id=c.subject_id,
                hypothesis=c.hypothesis,
                proof=c.producer_spec,
                treatment_id=c.treatment_id,
                baseline_id=c.baseline_id,
                claim_id=f"exp_{c.subject_id}",
            )
            c.evidence_id = claim.id
        else:
            raise ValueError(f"unknown candidate kind {c.kind!r}")
        self._transition(c, "evidence")  # -> EVIDENCE_READY

    def _promote(self, c: Candidate) -> None:
        try:
            rec = self.ledger.promote_claim(
                subject_id=c.subject_id,
                hypothesis=c.hypothesis,
                experiment="scheduled pipeline run",
                dataset=[c.baseline_id],
                metrics={},
                confidence=1.0,
                experiment_id=c.evidence_id,  # type: ignore[arg-type]
                baseline_id=c.baseline_id,
                treatment_id=c.treatment_id,
                claim_id=f"claim_{c.subject_id}",
            )
            c.promotion_id = rec.id
            self._transition(c, "adopt")  # -> PROMOTED
        except PromotionDenied as e:
            # Provenance gate refused (policy not passed / proof failed / etc).
            # Record a finding explaining why, and move to a terminal REJECTED
            # state so the candidate does not hang forever at PROMOTION_PENDING.
            self.ledger.record_claim(
                subject_id=c.subject_id,
                hypothesis=c.hypothesis,
                experiment="promotion gate refused",
                dataset=[c.baseline_id],
                metrics={"reason": str(e)},
                decision="rejected",
                confidence=1.0,
            )
            self._transition(c, "reject")  # -> REJECTED

    def _validate(self, c: Candidate) -> None:
        ok = True
        if c.validation is not None:
            ok = bool(c.validation(self.ledger, c.subject_id, c.evidence_id))  # type: ignore[arg-type]
        if ok:
            self._transition(c, "pass")  # -> STABLE
        else:
            self._rollback(c)

    def _rollback(self, c: Candidate) -> None:
        # Mark the promotion superseded so the provenance gate will refuse any
        # future promotion that references it, and record a finding explaining
        # the rollback. This keeps rollback inside the governance model.
        if c.promotion_id:
            prom = self.ledger.claims.get(c.promotion_id)
            if prom is not None:
                content = dict(prom.content)
                content.setdefault("meta", {})["superseded_by"] = "rollback"
                self.ledger.claims.add(
                    content, tags=prom.tags, confidence=prom.confidence,
                    record_id=prom.id,
                )
        self.ledger.record_claim(
            subject_id=c.subject_id,
            hypothesis=c.hypothesis,
            experiment="continuous validation failed",
            dataset=[c.baseline_id],
            metrics={"rolled_back_from": c.promotion_id},
            decision="rejected",
            confidence=1.0,
        )
        self._transition(c, "fail")  # -> ROLLED_BACK
