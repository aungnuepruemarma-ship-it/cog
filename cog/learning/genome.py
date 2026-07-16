"""Phase 12 (seed): reasoning genes extracted from real traces.

Instead of storing workflows, store which reasoning *stages* a run actually
exercised — observe (used memories), gather (used skills), hypothesize
(carried failure hypotheses), plan, execute, verify. Aggregated over the
graph, gene statistics show which stages correlate with verified outcomes.

Genes are **scientific objects**, not prompt templates: each stored gene
record carries its assumptions (part of the gene's definition), its
evidence (the experience ids it was observed in), its confidence (verified
rate when present), and its transfer_domains (the emergent domains — Phase
19 clustering — where it has *verified* evidence). `derived_from` stays
empty until Cog can honestly derive one gene from another.
"""

from __future__ import annotations

from cog.experience.record import Experience
from cog.learning.artifacts import Genome
from cog.learning.domains import DomainDiscovery
from cog.memory.base import MemoryRecord
from cog.memory.router import MemoryRouter

GENOME_STATS_ID = "concept_genome_stats"

# Assumptions are part of a gene's definition (what must hold for the stage
# to be meaningful), not evidence — evidence is mined from traces below.
GENE_ASSUMPTIONS: dict[str, list[str]] = {
    "observe": ["relevant memories are retrievable for the goal"],
    "gather": ["a compiled skill or prior workflow is applicable"],
    "hypothesize": ["a failure signal from a previous attempt is available"],
    "plan": ["the goal decomposes into tool-executable steps"],
    "execute": ["registered tools can effect the plan"],
    "verify": ["success is checkable against declared expectations"],
}


def genome_from_experience(experience: Experience) -> Genome:
    genes: list[str] = []
    workspace = experience.workspace
    if workspace.get("memories"):
        genes.append("observe")
    if workspace.get("skills") or experience.strategy == "skill_replay":
        genes.append("gather")
    if workspace.get("hypotheses"):
        genes.append("hypothesize")
    if experience.reasoning.get("raw_plan"):
        genes.append("plan")
    if experience.execution:
        genes.append("execute")
    genes.append("verify")  # every run passes through the pipeline
    return Genome(genes=genes, fitness=experience.confidence)


def gene_stats(experiences: list[Experience]) -> dict[str, dict[str, float]]:
    """Per-gene: how often present, and verified-rate when present."""
    stats: dict[str, dict[str, float]] = {}
    for experience in experiences:
        genome = genome_from_experience(experience)
        for gene in genome.genes:
            entry = stats.setdefault(gene, {"present": 0, "verified": 0})
            entry["present"] += 1
            entry["verified"] += int(experience.verified)
    for entry in stats.values():
        entry["verified_rate"] = round(entry["verified"] / entry["present"], 4)
    return stats


def store_gene_stats(memory: MemoryRouter) -> dict[str, dict[str, float]]:
    experiences = [Experience.from_dict(r.content) for r in memory.experiences.search(limit=500)]
    stats = gene_stats(experiences)
    if stats:
        memory.concepts.add(
            {"level": "genome", "genes": stats},
            tags=["genome"],
            record_id=GENOME_STATS_ID,
        )
    return stats


def store_gene_records(memory: MemoryRouter) -> list[MemoryRecord]:
    """Persist each gene as a scientific object with evidence, confidence,
    and transfer_domains — all mined from real traces, none hand-authored."""
    experiences = [Experience.from_dict(r.content) for r in memory.experiences.search(limit=500)]
    if not experiences:
        return []

    domain_of: dict[str, str] = {}
    for name, member_ids in DomainDiscovery().cluster(experiences).items():
        for member_id in member_ids:
            domain_of[member_id] = name

    evidence: dict[str, list[Experience]] = {}
    for experience in experiences:
        for gene in genome_from_experience(experience).genes:
            evidence.setdefault(gene, []).append(experience)

    records: list[MemoryRecord] = []
    for gene, observed_in in evidence.items():
        verified = [e for e in observed_in if e.verified]
        transfer_domains = sorted({domain_of[e.id] for e in verified if e.id in domain_of})
        confidence = len(verified) / len(observed_in)
        record = memory.concepts.add(
            {
                "level": "gene",
                "name": gene,
                "assumptions": GENE_ASSUMPTIONS.get(gene, []),
                "evidence": [e.id for e in observed_in],
                "transfer_domains": transfer_domains,
                "derived_from": [],  # stays empty until derivation is honest
            },
            tags=["gene"],
            confidence=round(confidence, 4),
            record_id=f"gene_{gene}",
        )
        for experience in observed_in:
            memory.add_edge(record.id, experience.id, "evidence")
        records.append(record)
    return records
