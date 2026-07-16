"""Belief revision: notice when Cog's own memory contradicts itself, and fix it.

Every fact in Cog's memory is written through the verification gate — but the
gate only checks an output against a *declared* expectation. A goal solved with
no expectation (a skill replay, a drifted tool) auto-passes the output check, so
two verified facts can end up asserting different answers to the *same* goal.
Nothing downstream notices: retrieval would happily hand a planner both "Compute
2 + 2 → 4" and "Compute 2 + 2 → 5" as equally trustworthy context. An
evidence-driven system that never audits its own store for self-contradiction is
quietly broken.

The Belief Revision Engine closes that hole. It groups the fact store by goal,
flags any goal that carries two or more distinct answers as a **contradiction**,
and resolves it by *weight of evidence* — the same principle the rest of Cog
runs on:

1. **support** — how many independent verified facts assert each answer (more
   confirmations win);
2. **confidence** — the strongest single fact, as a tie-breaker;
3. **recency** — the most recent evidence, as a last resort (newer supersedes).

The losing answers are marked ``superseded`` (retrieval then ignores them, so
Cog stops believing them) and the resolution is filed as a Scientific-Ledger
claim with a ``supersedes`` edge from winner to loser — reproducible, because it
is a deterministic function of the evidence. Belief revision is not deletion:
the superseded facts stay on disk as the audit trail of what Cog used to think
and why it changed its mind.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from cog.memory.base import MemoryRecord
from cog.memory.router import MemoryRouter

SUPERSEDED_TAG = "superseded"


def _answer_key(output: object) -> str:
    """Canonical, order-insensitive key for an answer value."""
    try:
        return json.dumps(output, sort_keys=True, default=str)
    except TypeError:
        return repr(output)


@dataclass
class Belief:
    """One candidate answer to a goal, and the evidence backing it."""

    answer_key: str
    records: list[MemoryRecord]

    @property
    def support(self) -> int:
        return len(self.records)

    @property
    def confidence(self) -> float:
        return max(r.confidence for r in self.records)

    @property
    def latest(self) -> str:
        return max(r.created_at for r in self.records)

    def strongest(self) -> MemoryRecord:
        """The single record that best represents this belief."""
        return max(self.records, key=lambda r: (r.confidence, r.created_at))

    def _rank(self) -> tuple[int, float, str]:
        return (self.support, self.confidence, self.latest)


@dataclass
class Contradiction:
    """A goal the fact store answers two or more incompatible ways."""

    goal: str
    beliefs: list[Belief]  # sorted best-first by the resolution policy
    winner: Belief
    losers: list[Belief]

    @property
    def answer(self) -> object:
        return self.winner.strongest().content.get("output")

    @property
    def superseded_ids(self) -> list[str]:
        return [r.id for belief in self.losers for r in belief.records]

    @property
    def evidence_ids(self) -> list[str]:
        return [r.id for belief in self.beliefs for r in belief.records]

    def hypothesis(self) -> str:
        answers = ", ".join(
            f"{b.strongest().content.get('output')!r} (support {b.support}, "
            f"conf {b.confidence:.2f})"
            for b in self.beliefs
        )
        winning = self.winner.strongest().content.get("output")
        return (
            f"Goal {self.goal!r} had conflicting answers [{answers}]; "
            f"revised to {winning!r} by weight of evidence."
        )


class BeliefRevisionEngine:
    def detect(self, memory: MemoryRouter) -> list[Contradiction]:
        """Find every goal the (non-superseded) fact store answers >1 way."""
        by_goal: dict[str, dict[str, Belief]] = {}
        for record in memory.facts.search(limit=1000):
            if SUPERSEDED_TAG in record.tags:
                continue
            goal = record.content.get("goal")
            if goal is None:
                continue
            key = _answer_key(record.content.get("output"))
            belief = by_goal.setdefault(goal, {}).setdefault(key, Belief(key, []))
            belief.records.append(record)

        contradictions: list[Contradiction] = []
        for goal, beliefs_by_key in by_goal.items():
            if len(beliefs_by_key) < 2:
                continue
            # Best-first: more support, then higher confidence, then more recent.
            ranked = sorted(beliefs_by_key.values(), key=lambda b: b._rank(), reverse=True)
            contradictions.append(
                Contradiction(goal=goal, beliefs=ranked, winner=ranked[0], losers=ranked[1:])
            )
        # Surface the most-evidenced contradictions first.
        contradictions.sort(key=lambda c: -len(c.evidence_ids))
        return contradictions

    def revise(self, memory: MemoryRouter, contradiction: Contradiction) -> None:
        """Enact one resolution: mark losing facts superseded and link them to
        the winner in the knowledge graph."""
        winner_id = contradiction.winner.strongest().id
        for record in (r for belief in contradiction.losers for r in belief.records):
            if SUPERSEDED_TAG in record.tags:
                continue
            memory.facts.add(
                record.content,
                tags=[*record.tags, SUPERSEDED_TAG],
                confidence=record.confidence,
                record_id=record.id,  # INSERT OR REPLACE: rewrites in place
            )
            memory.add_edge(winner_id, record.id, "supersedes")


def revise_and_store(memory: MemoryRouter, record_claims: bool = True) -> list[Contradiction]:
    """Detect self-contradictions, resolve each by weight of evidence, and file
    the resolution as a reproducible Scientific-Ledger claim."""
    engine = BeliefRevisionEngine()
    contradictions = engine.detect(memory)
    if not contradictions:
        return []

    ledger = None
    if record_claims:
        from cog.science.ledger import Ledger

        ledger = Ledger(memory)

    for contradiction in contradictions:
        engine.revise(memory, contradiction)
        if ledger is not None:
            ledger.record_claim(
                subject_id=contradiction.winner.strongest().id,
                hypothesis=contradiction.hypothesis(),
                experiment="group verified facts by goal; resolve conflicting "
                "answers by support, then confidence, then recency",
                dataset=contradiction.evidence_ids,
                metrics={
                    "answers": len(contradiction.beliefs),
                    "winning_support": contradiction.winner.support,
                    "superseded_facts": len(contradiction.superseded_ids),
                },
                decision="adopted",
                confidence=contradiction.winner.confidence,
                reproducible=True,
            )
    return contradictions
