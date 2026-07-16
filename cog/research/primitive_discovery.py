"""RESEARCH-GRADE. Phase 11: primitives are never invented — they are earned.

A searched representation may be *promoted* to a primitive only when the
evidence says so, scored on five axes computed from the ledger and the
knowledge graph:

- **compression** — experiences explained per artifact,
- **generality** — distinct emergent domains its members span,
- **transfer** — held-out predictive coverage beyond the null baseline,
- **predictive value** — holdout coverage itself,
- **simplicity** — inverse definition size.

The default thresholds are deliberately out of reach for today's evidence
base. That is the point: when a promotion happens, it will mean something.
"""

from __future__ import annotations

from dataclasses import dataclass

from cog.memory.router import MemoryRouter
from cog.science.ledger import Ledger

DEFAULT_THRESHOLDS = {
    "compression": 25.0,  # >= 25 experiences per artifact
    "generality": 3,  # verified in >= 3 emergent domains
    "transfer": 0.5,  # holdout beats null by >= 0.5
    "predictive": 0.9,  # >= 90% holdout coverage
    "simplicity": 0.2,  # definition of <= 5 elements
}


@dataclass
class PrimitiveScore:
    candidate_id: str
    name: str
    compression: float
    generality: int
    transfer: float
    predictive: float
    simplicity: float

    def qualifies(self, thresholds: dict[str, float]) -> bool:
        return (
            self.compression >= thresholds["compression"]
            and self.generality >= thresholds["generality"]
            and self.transfer >= thresholds["transfer"]
            and self.predictive >= thresholds["predictive"]
            and self.simplicity >= thresholds["simplicity"]
        )


class PrimitiveDiscovery:
    def __init__(self, memory: MemoryRouter, thresholds: dict[str, float] | None = None) -> None:
        self.memory = memory
        self.thresholds = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
        self.ledger = Ledger(memory)

    def score_representations(self) -> list[PrimitiveScore]:
        scores: list[PrimitiveScore] = []
        for record in self.memory.concepts.search(tags=["searched"], limit=200):
            content = record.content
            metrics = content.get("metrics", {})
            members = content.get("members", [])
            domains = self._domains_of(members)
            holdout = float(metrics.get("holdout_coverage", 0.0))
            null = float(metrics.get("null_coverage", 0.0))
            definition_size = int(metrics.get("definition_size", 0)) or 1
            scores.append(
                PrimitiveScore(
                    candidate_id=record.id,
                    name=content.get("name", record.id),
                    compression=float(metrics.get("compression_ratio", len(members))),
                    generality=len(domains),
                    transfer=max(0.0, holdout - null),
                    predictive=holdout,
                    simplicity=1.0 / definition_size,
                )
            )
        return scores

    def promote(self) -> list[str]:
        """Promote qualifying representations to primitives. Expected result
        with today's evidence base: nothing promotes — and that is recorded."""
        promoted: list[str] = []
        for score in self.score_representations():
            if not score.qualifies(self.thresholds):
                continue
            primitive_id = score.candidate_id.replace("rep_s_", "prim_")
            self.memory.concepts.add(
                {
                    "level": "primitive",
                    "name": score.name,
                    "promoted_from": score.candidate_id,
                    "scores": vars(score),
                },
                tags=["primitive"],
                confidence=score.predictive,
                record_id=primitive_id,
            )
            self.memory.add_edge(primitive_id, score.candidate_id, "promoted_from")
            self.ledger.record_claim(
                subject_id=primitive_id,
                hypothesis=f"{score.name} is a reasoning primitive",
                experiment="five-axis evidence scoring against promotion thresholds",
                dataset=[score.candidate_id],
                metrics={**vars(score), "thresholds": self.thresholds},
                decision="adopted",
                confidence=score.predictive,
                claim_id=f"claim_{primitive_id}_promoted",
            )
            promoted.append(primitive_id)
        return promoted

    def _domains_of(self, member_ids: list[str]) -> set[str]:
        members = set(member_ids)
        domains: set[str] = set()
        for record in self.memory.concepts.search(tags=["domain"], limit=200):
            if members & set(record.content.get("members", [])):
                domains.add(record.content.get("name", record.id))
        return domains
