"""Representation Competition: many theories for the same evidence, one survives.

``RepresentationSearch`` (Phase 8) tests a *single* candidate — the full feature
intersection of a cluster — against a null baseline. That answers "does this one
structure beat noise?" It does not answer the deeper, more scientific question:
**of all the ways to explain the same experiences, which explanation is best?**

This engine does. For one cluster of experiences it enumerates several competing
*theories* — different framings built from different *aspects* of the shared
structure (the tool-flow alone, the reasoning genes alone, flow+outcome, the full
conjunction, …) — and makes them compete head to head on the two things a good
theory must do:

- **predict** — cover held-out members of the cluster it claims to explain
  (``holdout_coverage``), net of how often it also covers non-members
  (``null_coverage``); their difference is the theory's *discrimination*;
- **compress** — say it in as few commitments as possible (a shorter
  ``definition`` is a shorter description; Occam's razor as an MDL tiebreak).

Theories are ranked lexicographically: most discriminating first, then most
predictive, then *simplest*. The winner survives and is stored as a
representation; every competitor — winner and losers alike — is filed in the
**Theory Ledger** with its scores and the reason it won or lost, so the record
holds not just what Cog believes but the alternatives it *considered and
rejected*. This is model selection over Cog's own execution logs: no new model,
no labels, only evidence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cog.experience.record import Experience
from cog.learning.representation_search import RepresentationSearch, behavioral_features
from cog.memory.router import MemoryRouter
from cog.science.ledger import Ledger

if TYPE_CHECKING:
    from cog.science.pipeline import Candidate

FeatureFn = Callable[[Experience], frozenset[str]]


def _kind(feature: str) -> str:
    """The aspect a feature describes: 'flow', 'gene', 'outcome', ..."""
    return feature.split(":", 1)[0]


def candidate_definitions(shared: frozenset[str]) -> list[frozenset[str]]:
    """Competing framings of one shared structure: every non-empty combination
    of the *aspects* (feature kinds) it contains, plus the full conjunction."""
    by_kind: dict[str, set[str]] = {}
    for feature in shared:
        by_kind.setdefault(_kind(feature), set()).add(feature)

    kinds = sorted(by_kind)
    definitions: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    # Non-empty subsets of the aspect set -> a theory that commits to just those
    # aspects. Up to 2**len(kinds)-1 framings (kinds are few: flow/gene/outcome).
    for mask in range(1, 1 << len(kinds)):
        definition: set[str] = set()
        for i, kind in enumerate(kinds):
            if mask & (1 << i):
                definition |= by_kind[kind]
        frozen = frozenset(definition)
        if frozen and frozen not in seen:
            seen.add(frozen)
            definitions.append(frozen)
    # Widest (most discriminating) first, shortest last — a stable, deterministic
    # order so equal-ranked theories break ties reproducibly.
    definitions.sort(key=lambda d: (-len(d), sorted(d)))
    return definitions


@dataclass
class Theory:
    """One candidate explanation of a cluster, and how well it did."""

    definition: frozenset[str]
    holdout_coverage: float = 0.0
    null_coverage: float = 0.0

    @property
    def discrimination(self) -> float:
        return round(self.holdout_coverage - self.null_coverage, 4)

    @property
    def description_length(self) -> int:
        return len(self.definition)

    @property
    def rank_key(self) -> tuple[float, float, int]:
        # More discriminating, then more predictive, then *simpler* (Occam).
        return (self.discrimination, round(self.holdout_coverage, 4), -self.description_length)

    @property
    def id(self) -> str:
        digest = hashlib.sha1("|".join(sorted(self.definition)).encode()).hexdigest()[:12]
        return f"theory_{digest}"

    def name(self) -> str:
        core = sorted(self.definition)
        return "theory: " + (" ∧ ".join(core[:4]) or "trivial")


@dataclass
class Competition:
    """The head-to-head result over one cluster's evidence."""

    theories: list[Theory]  # ranked best-first
    train_ids: list[str] = field(default_factory=list)
    holdout_ids: list[str] = field(default_factory=list)
    member_ids: list[str] = field(default_factory=list)
    min_prediction: float = 0.8
    min_definition_size: int = 1

    @property
    def winner(self) -> Theory | None:
        return self.theories[0] if self.theories else None

    @property
    def losers(self) -> list[Theory]:
        return self.theories[1:]

    @property
    def survivor(self) -> Theory | None:
        """The winner, but only if it actually earned survival: it must beat the
        null baseline and clear the prediction bar. Otherwise no theory won."""
        top = self.winner
        if top is None:
            return None
        if (
            top.discrimination > 0.0
            and top.holdout_coverage >= self.min_prediction
            and top.description_length >= self.min_definition_size
        ):
            return top
        return None

    def margin(self) -> float:
        """How decisively the winner beat the runner-up (0 if uncontested)."""
        if len(self.theories) < 2:
            return 0.0
        return round(self.theories[0].discrimination - self.theories[1].discrimination, 4)


class RepresentationCompetition:
    def __init__(
        self,
        min_members: int = 4,
        similarity: float = 0.6,
        min_prediction: float = 0.8,
        min_definition_size: int = 1,
        feature_fn: FeatureFn = behavioral_features,
    ) -> None:
        self.min_members = min_members
        self.similarity = similarity
        self.min_prediction = min_prediction
        self.min_definition_size = min_definition_size
        self.feature_fn = feature_fn

    def compete(self, members: list[Experience], others: list[Experience]) -> Competition:
        ordered = sorted(members, key=lambda e: e.id)
        train, holdout = ordered[0::2], ordered[1::2]  # deterministic split
        shared = frozenset.intersection(*(self.feature_fn(e) for e in train))

        def coverage(definition: frozenset[str], pool: list[Experience]) -> float:
            if not pool:
                return 0.0
            return sum(definition <= self.feature_fn(e) for e in pool) / len(pool)

        theories = [
            Theory(
                definition=definition,
                holdout_coverage=coverage(definition, holdout),
                null_coverage=coverage(definition, others),
            )
            for definition in candidate_definitions(shared)
            if len(definition) >= self.min_definition_size
        ]
        theories.sort(key=lambda t: t.rank_key, reverse=True)
        return Competition(
            theories=theories,
            train_ids=[e.id for e in train],
            holdout_ids=[e.id for e in holdout],
            member_ids=[e.id for e in ordered],
            min_prediction=self.min_prediction,
            min_definition_size=self.min_definition_size,
        )

    def run(self, experiences: list[Experience]) -> list[Competition]:
        searcher = RepresentationSearch(min_members=self.min_members, similarity=self.similarity)
        competitions: list[Competition] = []
        for members in searcher.cluster(experiences):
            if len(members) < self.min_members:
                continue
            member_ids = {e.id for e in members}
            others = [e for e in experiences if e.id not in member_ids]
            competition = self.compete(members, others)
            if competition.theories:
                competitions.append(competition)
        return competitions

    def discover_candidates(self, memory: MemoryRouter) -> list["Candidate"]:
        """CandidateSource contract: emit an EXPERIMENT candidate per survivor.

        Purely observational -- reruns the competition (no store writes; `run`
        does not persist, only `run_competition_and_store` does) and, for each
        survivor theory, packages a statistical A/B experiment: does the
        survivor's definition classify member vs non-member experiences better
        than a trivial baseline? Success per task = prediction matches truth.

        This is the honest "real experiment" the module's own comment says a
        survivor requires before runtime adoption (see GOV-REP-001). It does NOT
        promote, record, or mutate anything -- the orchestrator submits the
        candidate; ExperimentRunner produces the evidence; the gate decides.
        """
        from cog.learning.representation_search import behavioral_features
        from cog.science.experiment import EvidenceClass, ExperimentSpec
        from cog.science.pipeline import Candidate

        experiences = [
            Experience.from_dict(r.content) for r in memory.experiences.search(limit=500)
        ]
        by_id = {e.id: e for e in experiences}
        out: list[Candidate] = []
        for competition in self.run(experiences):
            survivor = competition.survivor
            if survivor is None:
                continue
            member_ids = set(competition.member_ids)
            # Tasks: every experience, labelled member/non-member. The classifier
            # under test predicts membership from behavioral features.
            tasks = [(e, e.id in member_ids) for e in experiences if e.id in by_id]
            definition = survivor.definition

            def _treatment(task, _def=definition):
                exp, is_member = task
                predicted = _def <= behavioral_features(exp)
                return {"verified": predicted == is_member}

            def _baseline(task):
                # Trivial classifier: predict "member" for everything (the null
                # theory that commits to nothing). Beats the survivor only when
                # the definition carries no real discrimination.
                _exp, is_member = task
                return {"verified": is_member is True}

            spec = ExperimentSpec(
                subject_id=survivor.id,
                hypothesis=(
                    f"{sorted(definition)} classifies membership of "
                    f"{len(member_ids)} experiences better than the null theory"
                ),
                baseline=_baseline,
                treatment=_treatment,
                tasks=tasks,
                evidence_class=EvidenceClass.STATISTICAL,
                baseline_id="null_theory",
                treatment_id=survivor.id,
            )
            out.append(
                Candidate(
                    subject_id=survivor.id,
                    hypothesis=spec.hypothesis,
                    kind="experiment",
                    producer_spec=spec,
                    baseline_id="null_theory",
                    treatment_id=survivor.id,
                )
            )
        return out


def run_competition_and_store(
    memory: MemoryRouter, min_members: int = 4, similarity: float = 0.6
) -> list[Competition]:
    """Run representation competition over stored experiences; store each
    survivor as a ``competed`` representation and file the whole field of
    competitors (winner + losers) in the Theory Ledger."""
    experiences = [Experience.from_dict(r.content) for r in memory.experiences.search(limit=500)]
    competitions = RepresentationCompetition(min_members=min_members, similarity=similarity).run(
        experiences
    )
    ledger = Ledger(memory)

    for competition in competitions:
        survivor = competition.survivor
        winner = competition.winner
        if survivor is not None:
            memory.concepts.add(
                {
                    "level": "representation",
                    "name": survivor.name(),
                    "definition": sorted(survivor.definition),
                    "members": competition.member_ids,
                    "metrics": {
                        "holdout_coverage": round(survivor.holdout_coverage, 4),
                        "null_coverage": round(survivor.null_coverage, 4),
                        "discrimination": survivor.discrimination,
                        "description_length": survivor.description_length,
                        "win_margin": competition.margin(),
                        "field_size": len(competition.theories),
                    },
                },
                tags=["representation", "competed"],
                confidence=survivor.holdout_coverage,
                record_id=survivor.id,
            )
            for member_id in competition.member_ids:
                memory.add_edge(survivor.id, member_id, "covers")

        # The Theory Ledger: record every competitor, won and lost alike.
        # These are CANDIDATE FINDINGS, not promotions. A survivor's win is
        # a scientific conclusion about which theory best explains the
        # data -- it does NOT enter the runtime until a real experiment
        # (claim_type="experiment") passes policy and promote_claim() is
        # called. Search output stays out of governance.
        for theory in competition.theories:
            won = theory is survivor
            reason = (
                f"best of {len(competition.theories)} theories (margin {competition.margin()})"
                if won
                else (
                    f"out-competed by {winner.id if winner else 'n/a'} "
                    f"(discrimination {theory.discrimination} vs "
                    f"{winner.discrimination if winner else 0.0})"
                )
            )
            ledger.record_claim(
                subject_id=theory.id,
                hypothesis=f"{sorted(theory.definition)} explains "
                f"{len(competition.member_ids)} experiences",
                experiment=f"representation competition: {reason}",
                dataset=competition.train_ids + competition.holdout_ids,
                metrics={
                    "holdout_coverage": round(theory.holdout_coverage, 4),
                    "null_coverage": round(theory.null_coverage, 4),
                    "discrimination": theory.discrimination,
                    "description_length": theory.description_length,
                },
                decision="adopted" if won else "rejected",
                confidence=theory.holdout_coverage,
                reproducible=True,
                claim_id=f"claim_{theory.id}_competed",
            )
            if winner is not None and not won:
                memory.add_edge(winner.id, theory.id, "won_over")
    return competitions


def discover_representation_candidates(
    memory: MemoryRouter, min_members: int = 4, similarity: float = 0.6
) -> list["Candidate"]:
    """Module-level CandidateSource entry point (mirrors *_and_store helpers)."""
    return RepresentationCompetition(
        min_members=min_members, similarity=similarity
    ).discover_candidates(memory)
