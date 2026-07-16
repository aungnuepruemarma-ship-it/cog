"""Representation Search: Cog searches for better internal representations.

The pipeline everything upstream of primitives depends on — with **no human
labeling** anywhere:

    Experiences → Cluster → Compress → Represent → Benchmark → Accept/Reject

- **Cluster**: greedy agglomeration over behavioral feature sets (tool-flow
  shape, reasoning genes, outcome) by Jaccard similarity.
- **Compress**: a cluster's candidate representation is the structure its
  members share (feature-set intersection).
- **Benchmark**: the candidate is built from a *train* split only, then
  judged on whether it covers the held-out members better than it covers
  random non-members (the null baseline). Prediction, not description.
- **Accept/Reject**: both outcomes become ledger claims — negative results
  are evidence too.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from cog.experience.record import Experience
from cog.learning.genome import genome_from_experience
from cog.memory.router import MemoryRouter
from cog.science.ledger import Ledger


def behavioral_features(experience: Experience) -> frozenset[str]:
    """Label-free description of what a run *did*."""
    features = {f"outcome:{experience.outcome}"}
    tools = [step["tool"] for step in experience.execution]
    if tools:
        features.add("flow:" + ">".join(tools))
    for gene in genome_from_experience(experience).genes:
        features.add(f"gene:{gene}")
    return frozenset(features)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass
class Candidate:
    members: list[Experience]
    definition: frozenset[str] = frozenset()  # shared structure from the TRAIN split
    train_ids: list[str] = field(default_factory=list)
    holdout_ids: list[str] = field(default_factory=list)
    holdout_coverage: float = 0.0
    null_coverage: float = 0.0
    accepted: bool = False
    reason: str = ""

    @property
    def id(self) -> str:
        digest = hashlib.sha1("|".join(sorted(self.definition)).encode()).hexdigest()[:12]
        return f"rep_s_{digest}"

    @property
    def compression_ratio(self) -> float:
        return float(len(self.members))  # n experiences explained by 1 structure

    def name(self) -> str:
        core = sorted(f for f in self.definition if f.startswith(("flow:", "gene:")))
        return "searched: " + (" + ".join(core[:4]) or "trivial")


@dataclass
class SearchReport:
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def accepted(self) -> list[Candidate]:
        return [c for c in self.candidates if c.accepted]

    @property
    def rejected(self) -> list[Candidate]:
        return [c for c in self.candidates if not c.accepted]


class RepresentationSearch:
    def __init__(
        self,
        min_members: int = 4,
        similarity: float = 0.6,
        min_holdout_coverage: float = 0.8,
        min_definition_size: int = 2,
    ) -> None:
        self.min_members = min_members
        self.similarity = similarity
        self.min_holdout_coverage = min_holdout_coverage
        self.min_definition_size = min_definition_size

    def cluster(self, experiences: list[Experience]) -> list[list[Experience]]:
        clusters: list[tuple[frozenset[str], list[Experience]]] = []
        for experience in sorted(experiences, key=lambda e: e.id):  # deterministic
            fs = behavioral_features(experience)
            for anchor, members in clusters:
                if _jaccard(anchor, fs) >= self.similarity:
                    members.append(experience)
                    break
            else:
                clusters.append((fs, [experience]))
        return [members for _anchor, members in clusters]

    def evaluate(self, members: list[Experience], others: list[Experience]) -> Candidate:
        """Compress a cluster on its train split, benchmark on its holdout."""
        ordered = sorted(members, key=lambda e: e.id)
        train, holdout = ordered[0::2], ordered[1::2]  # deterministic split
        definition = frozenset.intersection(*(behavioral_features(e) for e in train))
        candidate = Candidate(
            members=members,
            definition=definition,
            train_ids=[e.id for e in train],
            holdout_ids=[e.id for e in holdout],
        )

        def covers(experience: Experience) -> bool:
            return definition <= behavioral_features(experience)

        candidate.holdout_coverage = (
            sum(covers(e) for e in holdout) / len(holdout) if holdout else 0.0
        )
        candidate.null_coverage = sum(covers(e) for e in others) / len(others) if others else 0.0

        if len(definition) < self.min_definition_size:
            candidate.reason = "definition too trivial to predict anything"
        elif candidate.holdout_coverage < self.min_holdout_coverage:
            candidate.reason = "failed holdout coverage"
        elif candidate.null_coverage >= candidate.holdout_coverage:
            candidate.reason = "no better than the null baseline"
        else:
            candidate.accepted = True
            candidate.reason = "holdout coverage beats null baseline"
        return candidate

    def search(self, experiences: list[Experience]) -> SearchReport:
        report = SearchReport()
        clusters = self.cluster(experiences)
        for members in clusters:
            if len(members) < self.min_members:
                continue
            member_ids = {e.id for e in members}
            others = [e for e in experiences if e.id not in member_ids]
            report.candidates.append(self.evaluate(members, others))
        return report


def search_and_store(
    memory: MemoryRouter, min_members: int = 4, similarity: float = 0.6
) -> SearchReport:
    experiences = [Experience.from_dict(r.content) for r in memory.experiences.search(limit=500)]
    report = RepresentationSearch(min_members=min_members, similarity=similarity).search(
        experiences
    )
    ledger = Ledger(memory)
    for candidate in report.candidates:
        metrics = {
            "compression_ratio": candidate.compression_ratio,
            "holdout_coverage": round(candidate.holdout_coverage, 4),
            "null_coverage": round(candidate.null_coverage, 4),
            "definition_size": len(candidate.definition),
        }
        if candidate.accepted:
            memory.concepts.add(
                {
                    "level": "representation",
                    "name": candidate.name(),
                    "definition": sorted(candidate.definition),
                    "members": sorted(e.id for e in candidate.members),
                    "metrics": metrics,
                },
                tags=["representation", "searched"],
                confidence=candidate.holdout_coverage,
                record_id=candidate.id,
            )
            for experience in candidate.members:
                memory.add_edge(candidate.id, experience.id, "covers")
        ledger.record_claim(
            subject_id=candidate.id,
            hypothesis=(
                f"the structure {sorted(candidate.definition)} explains"
                f" {len(candidate.members)} experiences"
            ),
            experiment="train/holdout coverage vs null baseline (label-free)",
            dataset=candidate.train_ids + candidate.holdout_ids,
            metrics=metrics,
            decision="adopted" if candidate.accepted else "rejected",
            confidence=candidate.holdout_coverage,
            claim_id=f"claim_{candidate.id}_searched",
        )
    return report
