"""The Organizational Runtime: reason about *structures of reasoning*.

Skills chain tools; primitives merge fragments. An **organization** is the next
rung: a named, ordered structure of reasoning stages — Scientific Method
(``hypothesize ▸ test ▸ verify``), Debugging (``reproduce ▸ localize ▸ fix ▸
verify``), Proof (``state ▸ decompose ▸ prove ▸ conclude``). Cog does not just
*use* an organization; it can hold several, **compare** them on a task family by
the outcomes they actually produce, and **evolve** better ones from proven
primitives.

This is deliberately a clean layer *above* primitives, not a second promotion
heuristic. It rests on three contracts:

1. **schema** — an ``Organization`` is an ordered tuple of stage roles; an
   experience *follows* it when its tool-flow matches those stages;
2. **comparison** — ``OrganizationComparator`` scores an organization on a task
   family by the reasoning it produced: **verified-rate** first (does this
   structure actually work here?), then **coverage** (how much of the family it
   explains), then **cost** (mean actions) — and ranks organizations head to head;
3. **evolution** — ``OrganizationEvolver`` proposes recombinations and
   primitive-insertions, and accepts a candidate **only if it strictly improves
   the verified-rate** on the family. Nothing is adopted because it looks tidy;
   an organization is adopted because the evidence says it reasons better.

Everything is scored over a fixed set of experiences, so it is deterministic and
fixture-backed — the same evidence discipline as every other engine.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from cog.experience.record import Experience

FlowFn = Callable[[Experience], tuple[str, ...]]


def _default_flow(experience: Experience) -> tuple[str, ...]:
    return tuple(step["tool"] for step in experience.execution if step.get("tool"))


# --- Contract 1: schema -----------------------------------------------------


@dataclass(frozen=True)
class Organization:
    """A named, ordered structure of reasoning stages."""

    name: str
    stages: tuple[str, ...]
    description: str = ""

    @property
    def id(self) -> str:
        digest = hashlib.sha1("|".join(self.stages).encode()).hexdigest()[:12]
        return f"org_{digest}"

    def follows(self, flow: tuple[str, ...]) -> bool:
        """An experience realizes this organization when its flow *is* the
        organization's stage sequence."""
        return tuple(flow) == self.stages

    def signature(self) -> str:
        return " ▸ ".join(self.stages)


# A few seed organizations, expressed in Cog's concrete tool vocabulary. These
# are fixtures — starting structures Cog can compare and improve, not truths.
SEED_ORGANIZATIONS: tuple[Organization, ...] = (
    Organization("direct", ("calculator",), "compute straight away"),
    Organization("record-then-compute", ("note", "calculator"), "state intent, then compute"),
    Organization("transform-then-refine", ("text", "text"), "transform, then refine the result"),
)


# --- Contract 2: comparison -------------------------------------------------


@dataclass
class OrganizationScore:
    """How one organization performed on a task family."""

    organization: Organization
    followers: int  # experiences whose flow matched the structure
    verified: int  # of those, how many verified
    family_size: int  # total experiences in the family
    total_cost: int  # summed actions over followers

    @property
    def coverage(self) -> float:
        return self.followers / self.family_size if self.family_size else 0.0

    @property
    def verified_rate(self) -> float:
        return self.verified / self.followers if self.followers else 0.0

    @property
    def mean_cost(self) -> float:
        return self.total_cost / self.followers if self.followers else 0.0

    @property
    def rank_key(self) -> tuple[float, float, float]:
        # works here first, then explains more of the family, then is cheaper.
        return (
            round(self.verified_rate, 4),
            round(self.coverage, 4),
            -round(self.mean_cost, 4),
        )


class OrganizationComparator:
    def __init__(self, flow_fn: FlowFn = _default_flow) -> None:
        self.flow_fn = flow_fn

    def score(self, organization: Organization, family: Iterable[Experience]) -> OrganizationScore:
        family = list(family)
        followers = [e for e in family if organization.follows(self.flow_fn(e))]
        return OrganizationScore(
            organization=organization,
            followers=len(followers),
            verified=sum(bool(e.verified) for e in followers),
            family_size=len(family),
            total_cost=sum(len(self.flow_fn(e)) for e in followers),
        )

    def rank(
        self, organizations: Iterable[Organization], family: Iterable[Experience]
    ) -> list[OrganizationScore]:
        family = list(family)
        scores = [self.score(o, family) for o in organizations]
        scores.sort(key=lambda s: s.rank_key, reverse=True)
        return scores

    def compare(
        self, a: Organization, b: Organization, family: Iterable[Experience]
    ) -> Organization:
        """Return whichever organization reasons better on the family."""
        family = list(family)
        sa, sb = self.score(a, family), self.score(b, family)
        return a if sa.rank_key >= sb.rank_key else b


# --- Contract 3: evolution --------------------------------------------------


@dataclass
class EvolutionResult:
    incumbent: Organization
    candidate: Organization | None  # the best proposal considered
    accepted: bool
    incumbent_rate: float
    candidate_rate: float
    reason: str = ""
    proposals: list[Organization] = field(default_factory=list)


class OrganizationEvolver:
    """Propose better organizations from proven primitives and existing ones,
    and adopt one only when it strictly improves verified outcomes."""

    def __init__(self, comparator: OrganizationComparator | None = None) -> None:
        self.comparator = comparator or OrganizationComparator()

    def propose(
        self,
        parents: Iterable[Organization],
        primitives: Iterable[tuple[str, ...]] = (),
    ) -> list[Organization]:
        parents = list(parents)
        proposed: dict[tuple[str, ...], Organization] = {}

        def add(stages: tuple[str, ...], name: str, why: str) -> None:
            if stages and stages not in proposed and stages not in {p.stages for p in parents}:
                proposed[stages] = Organization(name, stages, why)

        # A proven primitive *is* a candidate organization on its own, and can be
        # spliced onto the front or back of each parent.
        for prim in primitives:
            prim = tuple(prim)
            add(prim, f"primitive::{'-'.join(prim)}", "a promoted primitive as a structure")
            for parent in parents:
                add(prim + parent.stages, f"{parent.name}+pre", "primitive prepended")
                add(parent.stages + prim, f"{parent.name}+post", "primitive appended")
        # Recombine pairs of parents (head of one, tail of another).
        for a in parents:
            for b in parents:
                if a is not b:
                    add(a.stages + b.stages, f"{a.name}>{b.name}", "two organizations composed")
        return list(proposed.values())

    def evolve(
        self,
        incumbent: Organization,
        family: Iterable[Experience],
        primitives: Iterable[tuple[str, ...]] = (),
        parents: Iterable[Organization] = (),
    ) -> EvolutionResult:
        family = list(family)
        candidates = self.propose([incumbent, *parents], primitives)
        base = self.comparator.score(incumbent, family)

        best: Organization | None = None
        best_rate = base.verified_rate
        for candidate in candidates:
            score = self.comparator.score(candidate, family)
            # Strict improvement in verified-rate, and it must actually apply
            # (coverage > 0) so we never "win" by explaining nothing.
            if score.followers and score.verified_rate > best_rate:
                best, best_rate = candidate, score.verified_rate

        if best is None:
            return EvolutionResult(
                incumbent=incumbent,
                candidate=None,
                accepted=False,
                incumbent_rate=base.verified_rate,
                candidate_rate=base.verified_rate,
                reason="no proposal strictly improved verified-rate",
                proposals=candidates,
            )
        return EvolutionResult(
            incumbent=incumbent,
            candidate=best,
            accepted=True,
            incumbent_rate=base.verified_rate,
            candidate_rate=best_rate,
            reason=f"verified-rate {base.verified_rate:.2f} -> {best_rate:.2f}",
            proposals=candidates,
        )
