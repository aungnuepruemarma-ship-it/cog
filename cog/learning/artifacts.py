"""Phases 6–13: the artifacts the learning engines produce.

These dataclasses fix the data contracts now, so implementing an engine
later never requires re-architecting the layers below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Skill:
    """Phase 6 output: a compiled, parameterized, executable workflow."""

    id: str
    name: str
    steps: list[dict[str, Any]]  # PlanStep dicts with {param} placeholders
    goal_template: str = ""  # e.g. "Compute {p0}"; params bind from new goals
    parameters: list[str] = field(default_factory=list)
    source_experiences: list[str] = field(default_factory=list)
    benchmark_score: float = 0.0


@dataclass
class Pattern:
    """Phase 7 output: a repeated failure/success/reasoning/tool-usage regularity."""

    id: str
    kind: str  # "failure" | "success" | "reasoning" | "tool_usage"
    description: str
    subject: str = ""  # the tool/workflow the regularity is about (merge key)
    support: list[str] = field(default_factory=list)  # experience ids as evidence


@dataclass
class Representation:
    """Phase 8 output: the smallest representation explaining several patterns."""

    id: str
    name: str  # e.g. "Dependency Resolution"
    merges: list[str] = field(default_factory=list)  # pattern ids it explains
    benchmark_score: float = 0.0


@dataclass
class Abstraction:
    """Phase 10 output: a merge of multiple representations."""

    id: str
    name: str  # e.g. "Constraint Satisfaction"
    merges: list[str] = field(default_factory=list)  # representation ids


@dataclass
class Primitive:
    """Phase 11 output. Never manually written — primitives emerge."""

    id: str
    name: str  # e.g. "Search under Constraints"
    explains: list[str] = field(default_factory=list)  # abstraction ids
    evidence_count: int = 0


@dataclass
class Genome:
    """Phase 12: an ordered assembly of reasoning genes for one task."""

    genes: list[str] = field(default_factory=list)
    # canonical gene pool: observe, gather, compare, cluster, hypothesize,
    # execute, verify, generalize, compress
    fitness: float = 0.0
