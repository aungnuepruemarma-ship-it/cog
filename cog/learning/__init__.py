"""Phases 6–13: the learning engine stack (see docs/05-learning-engine.md).

Implemented: SkillCompiler (6), PatternEngine (7), RepresentationEngine (8),
CompressionEngine (9, skill subsumption), DomainDiscovery (19), genome
statistics (12 seed), and the evidence-gated learning cycle. Interface-only:
Abstraction, Primitive, ReasoningGenome assembly, Organization.
"""

from cog.learning.artifacts import (
    Abstraction,
    Genome,
    Pattern,
    Primitive,
    Representation,
    Skill,
)
from cog.learning.beliefs import (
    BeliefRevisionEngine,
    Contradiction,
    revise_and_store,
)
from cog.learning.calibration import (
    CalibrationEngine,
    CalibrationReport,
    calibrated_confidence,
    evaluate_memory,
    store_calibration,
)
from cog.learning.compression import CompressionEngine, compress_and_store
from cog.learning.corrections import (
    CorrectionEngine,
    CorrectionRule,
    corrections_for_goal,
    mine_and_store,
)
from cog.learning.curiosity import (
    CuriosityEngine,
    ExperimentProposal,
    ExplorationOutcome,
)
from cog.learning.domains import DomainDiscovery, discover_and_store_domains
from cog.learning.engines import (
    AbstractionEngine,
    OrganizationEngine,
    PrimitiveEngine,
    ReasoningGenome,
)
from cog.learning.genome import (
    gene_stats,
    genome_from_experience,
    store_gene_records,
    store_gene_stats,
)
from cog.learning.loops import LoopReport, run_learning_cycle
from cog.learning.organizations import (
    SEED_ORGANIZATIONS,
    EvolutionResult,
    Organization,
    OrganizationComparator,
    OrganizationEvolver,
    OrganizationScore,
)
from cog.learning.patterns import PatternEngine, discover_and_store
from cog.learning.primitives import (
    DEFAULT_THRESHOLDS,
    GateResult,
    PrimitiveCandidate,
    PrimitiveDiscoveryEngine,
    discover_primitives_and_store,
)
from cog.learning.representation_competition import (
    Competition,
    RepresentationCompetition,
    Theory,
    candidate_definitions,
    run_competition_and_store,
)
from cog.learning.representation_search import (
    RepresentationSearch,
    SearchReport,
    behavioral_features,
    search_and_store,
)
from cog.learning.representations import RepresentationEngine, reduce_and_store
from cog.learning.skill_compiler import (
    SkillCompiler,
    compile_and_store,
    instantiate_plan,
    match_skill,
)

__all__ = [
    "Abstraction",
    "AbstractionEngine",
    "BeliefRevisionEngine",
    "CalibrationEngine",
    "CalibrationReport",
    "Competition",
    "CompressionEngine",
    "DEFAULT_THRESHOLDS",
    "EvolutionResult",
    "GateResult",
    "Organization",
    "OrganizationComparator",
    "OrganizationEvolver",
    "OrganizationScore",
    "SEED_ORGANIZATIONS",
    "PrimitiveCandidate",
    "PrimitiveDiscoveryEngine",
    "Contradiction",
    "CorrectionEngine",
    "CorrectionRule",
    "CuriosityEngine",
    "DomainDiscovery",
    "ExperimentProposal",
    "ExplorationOutcome",
    "Genome",
    "LoopReport",
    "OrganizationEngine",
    "Pattern",
    "PatternEngine",
    "Primitive",
    "PrimitiveEngine",
    "ReasoningGenome",
    "Representation",
    "RepresentationCompetition",
    "RepresentationEngine",
    "RepresentationSearch",
    "SearchReport",
    "Skill",
    "Theory",
    "SkillCompiler",
    "behavioral_features",
    "calibrated_confidence",
    "candidate_definitions",
    "compile_and_store",
    "discover_primitives_and_store",
    "compress_and_store",
    "corrections_for_goal",
    "discover_and_store",
    "evaluate_memory",
    "discover_and_store_domains",
    "gene_stats",
    "genome_from_experience",
    "instantiate_plan",
    "match_skill",
    "mine_and_store",
    "reduce_and_store",
    "revise_and_store",
    "run_competition_and_store",
    "run_learning_cycle",
    "search_and_store",
    "store_calibration",
    "store_gene_records",
    "store_gene_stats",
]
