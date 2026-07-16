"""Phase 4: the five memory systems.

Facts, Experiences, Skills, Concepts, Documents — same persistence, different
kinds and retrieval affordances.
"""

from __future__ import annotations

from cog.memory.base import BaseStore, MemoryRecord


class FactStore(BaseStore):
    """Small verified statements, e.g. {"statement": ..., "source_experience": ...}."""

    kind = "fact"


class ExperienceStore(BaseStore):
    """Complete task records (Phase 5 mirrors Experience nodes here)."""

    kind = "experience"


class SkillStore(BaseStore):
    """Compiled, executable workflows (Phase 6 output)."""

    kind = "skill"


class ConceptStore(BaseStore):
    """Patterns, representations, abstractions, primitives (Phases 7–11 output)."""

    kind = "concept"


class DocumentStore(BaseStore):
    """Raw reference material. Keyword scoring stands in for RAG; swap in an
    embedding-backed retriever behind the same interface later."""

    kind = "document"

    def add_document(
        self, text: str, title: str = "", tags: list[str] | None = None
    ) -> MemoryRecord:
        return self.add({"title": title, "text": text}, tags=tags)

    def retrieve(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        return self.search(query=query, limit=limit)


__all__ = [
    "FactStore",
    "ExperienceStore",
    "SkillStore",
    "ConceptStore",
    "DocumentStore",
    "MemoryRecord",
]
