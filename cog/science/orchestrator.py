"""Candidate Orchestrator: the bridge from learning to governed adoption.

This is the layer that closes the gap the candidate-emission contract opened:

    Learning modules (CandidateSource)
          |  discover_candidates(memory) -> list[Candidate]
          v
    CandidateOrchestrator          <-- owns WORKFLOW POLICY
          |  submit + tick
          v
    PromotionScheduler
          |
          v
    Evidence producers (ExperimentRunner / FormalVerifier)
          |
          v
    Promotion gate (promote_claim)

Design boundaries this respects:
- Learning modules DISCOVER candidates; they do not decide adoption and never
  touch the scheduler. The orchestrator is the only thing that submits.
- Workflow policy (which sources to poll, when to submit, how far to advance
  the scheduler each cycle) lives HERE, not in the learners and not in the
  scheduler (which is a mechanical state machine).
- Submission is idempotent: candidates carry a deterministic stable id, so
  re-polling a source and re-submitting an already-known candidate is a no-op
  (the scheduler's persisted state is preserved, not reset).

It does NOT introduce a second promotion path: it only submits candidates and
advances the existing scheduler. The single adoption authority remains
promote_claim, reached exclusively through the scheduler's authorized producers.
"""

from __future__ import annotations

from typing import Any, Callable

from cog.science.ledger import Ledger
from cog.science.pipeline import Candidate, PromotionScheduler

# A source is anything that, given memory, yields candidates. Both the
# CandidateSource protocol objects and the module-level discover_* functions
# satisfy this via a thin adapter below.
SourceFn = Callable[[Any], list[Candidate]]


class CandidateOrchestrator:
    """Polls candidate sources, submits new candidates, advances the scheduler.

    ``sources`` are callables ``memory -> list[Candidate]`` (the module-level
    ``discover_*_candidates`` functions, or ``CandidateSource.discover_candidates``
    bound methods). ``max_ticks`` bounds how far each cycle advances the
    scheduler so a single cycle cannot run unboundedly.
    """

    def __init__(
        self,
        ledger: Ledger,
        sources: list[SourceFn],
        scheduler: PromotionScheduler | None = None,
        max_ticks: int = 16,
    ) -> None:
        self.ledger = ledger
        self.sources = sources
        self.scheduler = scheduler or PromotionScheduler(ledger)
        self.max_ticks = max_ticks

    def collect(self, memory: Any) -> list[Candidate]:
        """Poll every source. Pure with respect to governance -- sources are
        required to be observational (no ledger writes, no promotion)."""
        found: list[Candidate] = []
        for source in self.sources:
            found.extend(source(memory))
        return found

    def submit_new(self, memory: Any) -> list[str]:
        """Collect + submit, deduping by stable candidate id. Returns the ids
        that were newly submitted this call (already-known ids are skipped)."""
        known = {
            r.content.get("candidate_id") for r in self.scheduler.ledger.pipeline.search(limit=500)
        }
        submitted: list[str] = []
        for c in self.collect(memory):
            if c.candidate_id in known:
                continue  # idempotent: an already-tracked candidate is not reset
            self.scheduler.submit(c)
            submitted.append(c.candidate_id)  # type: ignore[arg-type]
            known.add(c.candidate_id)
        return submitted

    def cycle(self, memory: Any) -> dict[str, Any]:
        """One orchestration cycle: submit new candidates, then advance the
        scheduler until quiescent (no state changed) or max_ticks reached.

        Returns a small report: submitted ids, ticks spent, and the final state
        counts across the pipeline store."""
        submitted = self.submit_new(memory)
        ticks = 0
        prev_signature: tuple | None = None
        for _ in range(self.max_ticks):
            self.scheduler.tick()
            ticks += 1
            sig = self._signature()
            if sig == prev_signature:
                break  # quiescent: nothing advanced this tick
            prev_signature = sig
        return {
            "submitted": submitted,
            "ticks": ticks,
            "states": self._state_counts(),
        }

    def _signature(self) -> tuple:
        rows = sorted(
            (r.content.get("candidate_id"), r.content.get("state"))
            for r in self.scheduler.ledger.pipeline.search(limit=500)
        )
        return tuple(rows)

    def _state_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.scheduler.ledger.pipeline.search(limit=500):
            st = r.content.get("state", "?")
            counts[st] = counts.get(st, 0) + 1
        return counts
