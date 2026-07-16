"""The Scientific Ledger: every claim becomes evidence.

Whenever Cog adopts (or rejects) an artifact — a compiled skill, a
compression, a searched representation, an experiment outcome — the
decision is recorded as a *claim*: hypothesis, experiment, dataset,
metrics, decision, confidence, reproducibility. The ledger is what lets
Cog answer "why do we believe this skill?" or "why did this abstraction
replace another?" with a traceable chain instead of a shrug.

Claims live in the same inspectable SQLite records table (kind="claim"),
linked into the knowledge graph: claim→subject (``about``) and
claim→evidence (``dataset``).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from cog.memory.base import BaseStore, MemoryRecord
from cog.memory.router import MemoryRouter
from cog.science.promotion import PromotionDenied

# Authorized evidence producers. A claim_type listed here may ONLY be written
# by a producer that passes record_claim(_evidence_authority=<key>). To add a
# new evidence source (e.g. deterministic benchmark, simulation validation),
# call register_evidence_producer() ONCE at import time (before the registry is
# locked). The registry IS part of the trusted computing base -- so it is
# frozen at module load: after lock_evidence_registry() runs (implicitly at the
# end of this module), no code can add/overwrite a producer. This prevents a
# module from self-registering and then self-authorizing, which would defeat
# the single-authority invariant.
_EVIDENCE_REGISTRY: dict[str, set[str]] = {
    "experiment_runner": {"experiment"},
    "formal_verifier": {"formal_verification"},
}
_EVIDENCE_REGISTRY_LOCKED = False


class RegistryLocked(Exception):
    """Raised when code tries to mutate the evidence-producer registry after
    it has been locked at startup."""


def register_evidence_producer(key: str, claim_types: set[str]) -> None:
    """Register an authorized evidence producer.

    May ONLY be called before the registry is locked (i.e. at import time, by
    the module that owns the producer). After lock, raises RegistryLocked.
    Idempotent for the same key+claim_types; overwriting an existing key with
    different claim_types is refused (duplicate registration prevented).
    """
    if _EVIDENCE_REGISTRY_LOCKED:
        raise RegistryLocked(
            f"evidence producer registry is locked; cannot register {key!r}"
        )
    if key in _EVIDENCE_REGISTRY and _EVIDENCE_REGISTRY[key] != set(claim_types):
        raise RegistryLocked(
            f"evidence producer {key!r} already registered with "
            f"{sorted(_EVIDENCE_REGISTRY[key])}; duplicate/conflicting "
            f"registration refused"
        )
    _EVIDENCE_REGISTRY[key] = set(claim_types)


def lock_evidence_registry() -> None:
    """Freeze the registry. Called once at the end of this module's import."""
    global _EVIDENCE_REGISTRY_LOCKED
    _EVIDENCE_REGISTRY_LOCKED = True


# The public, immutable view of the registry. Code reads this; it cannot mutate.
EVIDENCE_PRODUCERS = MappingProxyType(dict(_EVIDENCE_REGISTRY))
# Flat set of all protected evidence claim types, for the guard lookup.
_PROTECTED_EVIDENCE_KINDS = {
    kind for _keys in EVIDENCE_PRODUCERS.values() for kind in _keys
}

# Freeze the registry: from here on, no producer can be added or changed.
lock_evidence_registry()


class ClaimStore(BaseStore):
    kind = "claim"


class PipelineStore(BaseStore):
    """Persistent state for the autonomous promotion pipeline (cog.science.pipeline).
    kind="pipeline". Each row is one candidate's current lifecycle state, so
    tick() can resume after a crash."""


class Ledger:
    def __init__(self, memory: MemoryRouter) -> None:
        self.memory = memory
        self.claims = ClaimStore(memory.conn)
        self.pipeline = PipelineStore(memory.conn)

    def record_claim(
        self,
        *,
        subject_id: str,
        hypothesis: str,
        experiment: str,
        dataset: list[str],
        metrics: dict[str, Any],
        decision: str,  # "adopted" | "rejected" (conclusion about the subject)
        confidence: float,
        reproducible: bool | None = None,
        claim_type: str = "finding",  # "finding" | "experiment" | "promotion"
        meta: dict[str, Any] | None = None,  # experiment governance fields (status, passed_policy, ...)
        claim_id: str | None = None,
        _via_promotion_gate: bool = False,  # private: only promote_claim may set this
        _evidence_authority: str | None = None,  # private: authorized evidence producer key
    ) -> MemoryRecord:
        # Single source of authority. No module may construct a promotion or
        # an evidence record directly -- only the authorized producer may:
        #   promotion  -> ONLY promote_claim (runs the provenance gate)
        #   evidence    -> ONLY a producer registered in EVIDENCE_PRODUCERS
        #                  (e.g. "experiment_runner", "formal_verifier")
        # Hand-crafting any of these via record_claim bypasses governance, so
        # it is refused. Adding a new evidence producer is a one-line registry
        # entry -- record_claim itself never changes again.
        if claim_type == "promotion":
            if not _via_promotion_gate:
                raise PromotionDenied(
                    "promotion claims must go through Ledger.promote_claim() "
                    "(which runs the provenance gate) -- direct record_claim is refused"
                )
        elif claim_type in _PROTECTED_EVIDENCE_KINDS:
            allowed = EVIDENCE_PRODUCERS.get(_evidence_authority)
            if allowed is None or claim_type not in allowed:
                raise PromotionDenied(
                    f"claim_type='{claim_type}' is an evidence record and must be "
                    f"emitted by its authorized producer (got authority="
                    f"{_evidence_authority!r}); registered producers="
                    f"{sorted(EVIDENCE_PRODUCERS)} -- provenance must stay trustworthy"
                )
        record = self.claims.add(
            {
                "subject_id": subject_id,
                "hypothesis": hypothesis,
                "experiment": experiment,
                "dataset": list(dataset),
                "metrics": metrics,
                "decision": decision,
                "claim_type": claim_type,
                "reproducible": reproducible,
                "meta": meta or {},
            },
            tags=["claim", decision, claim_type],
            confidence=confidence,
            record_id=claim_id,
        )
        self.memory.add_edge(record.id, subject_id, "about")
        for evidence_id in dataset:
            self.memory.add_edge(record.id, evidence_id, "dataset")
        # A claim is a discrete, atomic scientific event. BaseStore.add does NOT
        # commit (callers like CogRuntime.run batch-commit per cycle), but the
        # experiment path has no enclosing commit -- so without this the claim
        # INSERT is lost when the connection closes. Commit per claim (one
        # commit, not per-row-in-a-loop) to keep claim persistence durable.
        self.memory.conn.commit()
        return record

    def claims_about(self, subject_id: str) -> list[MemoryRecord]:
        return [
            record
            for record in self.claims.search(limit=500)
            if record.content.get("subject_id") == subject_id
        ]

    def why(self, subject_id: str) -> list[dict[str, Any]]:
        """The traceable answer to "why do we believe this?"."""
        return [
            {
                "claim_id": record.id,
                "hypothesis": record.content["hypothesis"],
                "experiment": record.content["experiment"],
                "dataset": record.content["dataset"],
                "metrics": record.content["metrics"],
                "decision": record.content["decision"],
                "confidence": record.confidence,
                "reproducible": record.content.get("reproducible"),
                "recorded_at": record.created_at,
            }
            for record in self.claims_about(subject_id)
        ]

    def mark_reproducible(self, claim_id: str, reproducible: bool) -> None:
        record = self.claims.get(claim_id)
        if record is None:
            raise KeyError(claim_id)
        content = dict(record.content)
        content["reproducible"] = reproducible
        self.claims.add(
            content, tags=record.tags, confidence=record.confidence, record_id=record.id
        )

    def promote_claim(
        self,
        *,
        subject_id: str,
        hypothesis: str,
        experiment: str,
        dataset: list[str],
        metrics: dict[str, Any],
        confidence: float,
        experiment_id: str,
        baseline_id: str,
        treatment_id: str,
        claim_id: str | None = None,
    ) -> Any:
        """The ONLY sanctioned path for adopting an artifact into the runtime.

        Provenance-only: verifies the referenced experiment claim exists,
        completed, passed policy, treatment matches, and is not superseded.
        On success records a claim_type="promotion" claim. On failure raises
        PromotionDenied -- the caller should record a FINDING instead of
        adopting. This keeps governance out of record_claim (which stays a
        dumb ledger primitive for findings/experiments).
        """
        from cog.science.promotion import promote_claim as _promote

        return _promote(
            self,
            subject_id=subject_id,
            hypothesis=hypothesis,
            experiment=experiment,
            dataset=dataset,
            metrics=metrics,
            confidence=confidence,
            experiment_id=experiment_id,
            baseline_id=baseline_id,
            treatment_id=treatment_id,
            claim_id=claim_id,
        )
