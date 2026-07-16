"""Phases 10–13: the not-yet-implemented learning engines (typed interfaces).

Implemented engines live in their own modules: SkillCompiler
(``skill_compiler.py``), PatternEngine (``patterns.py``),
RepresentationEngine (``representations.py``), CompressionEngine
(``compression.py``), DomainDiscovery (``domains.py``), and the genome seed
(``genome.py``). The contracts below stay fixed until their implementations
land, each behind a benchmark (Phase 9 / Phase 17 discipline).
"""

from __future__ import annotations

from typing import Any

from cog.learning.artifacts import Abstraction, Genome, Primitive, Representation
from cog.workspace.workspace import TaskWorkspace


class AbstractionEngine:
    """Phase 10: can multiple representations merge?"""

    def merge(self, representations: list[Representation]) -> list[Abstraction]:
        raise NotImplementedError("M3 — see docs/09-roadmap.md")


class PrimitiveEngine:
    """Phase 11 (monthly, slow): the smallest verified primitive explaining
    many abstractions. Primitives are never manually written — they emerge."""

    def emerge(self, abstractions: list[Abstraction]) -> list[Primitive]:
        raise NotImplementedError("M5+ — see docs/09-roadmap.md")


class ReasoningGenome:
    """Phase 12: assemble reasoning genes per task; genomes evolve."""

    def assemble(self, workspace: TaskWorkspace) -> Genome:
        raise NotImplementedError("M5+ — see docs/09-roadmap.md")


class OrganizationEngine:
    """Phase 13: store organization, not objects. North-star hypothesis —
    stays experimental until Phase 17 experiments show it beats object-level
    storage (see docs/00-vision.md)."""

    def organize(self, concepts: list[Any]) -> list[Any]:
        raise NotImplementedError("M5+ — see docs/09-roadmap.md")
