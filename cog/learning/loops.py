"""The continuous learning loops (docs/08-learning-loops.md), evidence-gated.

``run_learning_cycle`` does one pass of the hourly/daily/weekly work over a
memory store: compile skills, discover patterns, reduce them to
representations, cluster emergent domains, and refresh gene statistics.
Loops are triggered by evidence thresholds — with too few experiences the
cycle reports itself skipped rather than hallucinating structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cog.learning.abstractions import merge_records
from cog.learning.beliefs import revise_and_store
from cog.learning.calibration import store_calibration
from cog.learning.compression import compress_and_store
from cog.learning.corrections import mine_and_store
from cog.learning.domains import discover_and_store_domains
from cog.learning.genome import store_gene_records, store_gene_stats
from cog.learning.patterns import discover_and_store
from cog.learning.primitives import discover_primitives_and_store
from cog.learning.representation_competition import run_competition_and_store
from cog.learning.representation_search import search_and_store
from cog.learning.representations import reduce_and_store
from cog.learning.skill_compiler import compile_and_store
from cog.learning.stats import StatReport, proportion_ci, report_from_counts
from cog.memory.router import MemoryRouter


@dataclass
class LoopReport:
    experiences_seen: int = 0
    skills_compiled: int = 0
    skills_compressed: int = 0
    patterns_found: int = 0
    representations_built: int = 0
    representations_searched: int = 0
    representations_accepted: int = 0
    abstractions_merged: int = 0
    theories_competed: int = 0
    theories_survived: int = 0
    primitives_promoted: int = 0
    domains_discovered: int = 0
    genes_tracked: int = 0
    corrections_learned: int = 0
    contradictions_resolved: int = 0
    calibration_ece: float = 0.0
    confidence_reliable: bool = True
    skipped: bool = False
    # Statistical evidence packets (one per inductive engine)
    skill_stats: StatReport | None = None
    representation_stats: StatReport | None = None
    primitive_stats: StatReport | None = None
    org_selection_stats: StatReport | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "experiences_seen": self.experiences_seen,
            "skills_compiled": self.skills_compiled,
            "skills_compressed": self.skills_compressed,
            "patterns_found": self.patterns_found,
            "representations_built": self.representations_built,
            "representations_searched": self.representations_searched,
            "representations_accepted": self.representations_accepted,
            "abstractions_merged": self.abstractions_merged,
            "theories_competed": self.theories_competed,
            "theories_survived": self.theories_survived,
            "primitives_promoted": self.primitives_promoted,
            "domains_discovered": self.domains_discovered,
            "genes_tracked": self.genes_tracked,
            "corrections_learned": self.corrections_learned,
            "contradictions_resolved": self.contradictions_resolved,
            "calibration_ece": self.calibration_ece,
            "confidence_reliable": self.confidence_reliable,
            "skipped": self.skipped,
        }
        if self.skill_stats:
            d["skill_stats"] = self.skill_stats.to_dict()
        if self.representation_stats:
            d["representation_stats"] = self.representation_stats.to_dict()
        if self.primitive_stats:
            d["primitive_stats"] = self.primitive_stats.to_dict()
        if self.org_selection_stats:
            d["org_selection_stats"] = self.org_selection_stats.to_dict()
        return d


def _verified_rate(memory: MemoryRouter) -> tuple[int, int]:
    """Return (verified_count, total) over experiences (capped at 500 for cost)."""
    exps = memory.experiences.search(limit=500)
    if not exps:
        return (0, 0)
    verified = sum(1 for r in exps if r.content.get("verification", {}).get("verified"))
    return (verified, len(exps))


def _skill_stat_report(memory: MemoryRouter, skills: list) -> StatReport:
    """Compute verification stats for compiled-skill source experiences.

    Each skill is backed by ``source_experiences``; we pull those experiences
    and measure their verified-rate as evidence the skill will generalize.
    """
    if not skills:
        return StatReport()
    source_ids = set()
    for s in skills:
        source_ids.update(s.source_experiences)
    if not source_ids:
        return StatReport()
    # Count verified among the skill's supporting experiences
    exps = memory.experiences.search(limit=500)
    supporting = [r for r in exps if r.id in source_ids]
    if not supporting:
        return StatReport()
    verified = sum(1 for r in supporting if r.content.get("verification", {}).get("verified"))
    return report_from_counts(verified, len(supporting))


def _representation_stat_report(memory: MemoryRouter) -> StatReport:
    """Verification rate among experiences that drove representation search."""
    verified, total = _verified_rate(memory)
    if total == 0:
        return StatReport()
    return report_from_counts(verified, total)


def run_learning_cycle(
    memory: MemoryRouter, min_support: int = 2, min_experiences: int = 2
) -> LoopReport:
    experiences_seen = memory.experiences.count()
    # Belief revision is a self-audit, not structure-induction — it never
    # hallucinates from sparse data, it only reconciles contradictions that are
    # already on disk. So it runs even below the evidence gate that guards the
    # inductive engines.
    contradictions = revise_and_store(memory)
    if experiences_seen < min_experiences:
        notes = [f"evidence gate: {experiences_seen} < {min_experiences} experiences"]
        notes += [f"belief revised: {c.hypothesis()}" for c in contradictions]
        return LoopReport(
            experiences_seen=experiences_seen,
            contradictions_resolved=len(contradictions),
            skipped=True,
            notes=notes,
        )

    skills = compile_and_store(memory, min_support=min_support)
    compressions = compress_and_store(memory)
    patterns = discover_and_store(memory, min_support=min_support)
    representations = reduce_and_store(memory, min_merge=min_support)
    abstractions = merge_records(memory, min_merge=min_support)
    search_report = search_and_store(memory)
    competitions = run_competition_and_store(memory)
    domains = discover_and_store_domains(memory, min_members=min_support)
    primitives = discover_primitives_and_store(memory)  # after domains: needs them
    genes = store_gene_stats(memory)
    store_gene_records(memory)
    corrections = mine_and_store(memory, min_support=min_support)
    calibration = store_calibration(memory)

    # --- Statistical evidence packets (same format across all engines) ---
    skill_stats = _skill_stat_report(memory, skills)
    representation_stats = _representation_stat_report(memory)
    primitive_stats = StatReport(n=len(primitives)) if primitives else StatReport()
    # Org selection: verified-rate among experiences (proxy for org-controlled runs)
    v, t = _verified_rate(memory)
    org_selection_stats = report_from_counts(v, t) if t > 0 else StatReport()

    notes = []
    if skills:
        notes.append(f"compiled skills: {[s.name for s in skills]}")
    notes.extend(compressions)
    for accepted in search_report.accepted:
        notes.append(f"representation accepted: {accepted.name()}")
    for primitive in primitives:
        notes.append(f"primitive PROMOTED (all six gates): {primitive.name()}")
    survivors = [c.survivor for c in competitions if c.survivor is not None]
    for competition in competitions:
        if competition.survivor is not None:
            notes.append(
                f"theory won ({len(competition.theories)} competed, "
                f"margin {competition.margin()}): {competition.survivor.name()}"
            )
    for rule in corrections:
        notes.append(f"correction learned: {rule.hypothesis()}")
    for contradiction in contradictions:
        notes.append(f"belief revised: {contradiction.hypothesis()}")
    notes.append(
        f"confidence calibration: ECE {calibration.ece} over {calibration.n} experiences "
        f"({'reliable' if calibration.reliable else 'MISCALIBRATED'})"
    )
    memory.conn.commit()  # single transactional commit for the whole learn cycle
    return LoopReport(
        experiences_seen=experiences_seen,
        skills_compiled=len(skills),
        skills_compressed=len(compressions),
        patterns_found=len(patterns),
        representations_built=len(representations),
        representations_searched=len(search_report.candidates),
        representations_accepted=len(search_report.accepted),
        abstractions_merged=len(abstractions),
        theories_competed=sum(len(c.theories) for c in competitions),
        theories_survived=len(survivors),
        primitives_promoted=len(primitives),
        domains_discovered=len(domains),
        genes_tracked=len(genes),
        corrections_learned=len(corrections),
        contradictions_resolved=len(contradictions),
        calibration_ece=calibration.ece,
        confidence_reliable=calibration.reliable,
        skill_stats=skill_stats,
        representation_stats=representation_stats,
        primitive_stats=primitive_stats,
        org_selection_stats=org_selection_stats,
        notes=notes,
    )