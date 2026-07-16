"""Phase 5: the Experience Graph.

Nodes are Experience records (mirrored into the ExperienceStore so the
Memory Router retrieves them like any memory); edges are typed links —
the bottom layers of the knowledge graph (Phase 15).
"""

from __future__ import annotations

from dataclasses import dataclass

from cog.experience.record import Experience
from cog.memory.router import MemoryRouter


@dataclass
class Edge:
    src: str
    dst: str
    kind: str  # e.g. "produced_fact", "used_skill", "similar_goal", "retry_of"


class ExperienceGraph:
    def __init__(self, memory: MemoryRouter) -> None:
        self.memory = memory  # edges live in the same inspectable memory.db

    def record(self, experience: Experience, links: list[Edge] | None = None) -> None:
        self.memory.experiences.add(
            experience.to_dict(),
            tags=[experience.outcome],
            confidence=experience.confidence,
            record_id=experience.id,
        )
        for edge in links or []:
            self.add_edge(edge)

    def add_edge(self, edge: Edge) -> None:
        self.memory.add_edge(edge.src, edge.dst, edge.kind)

    def get(self, experience_id: str) -> Experience | None:
        record = self.memory.experiences.get(experience_id)
        if record is None:
            return None
        return Experience.from_dict(record.content)

    def all_experiences(self, limit: int = 500) -> list[Experience]:
        return [
            Experience.from_dict(r.content) for r in self.memory.experiences.search(limit=limit)
        ]

    def edges_from(self, experience_id: str, kind: str | None = None) -> list[Edge]:
        return [Edge(*row) for row in self.memory.edges_from(experience_id, kind)]

    def count(self) -> int:
        return self.memory.experiences.count()
