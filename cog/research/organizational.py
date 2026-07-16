"""RESEARCH-GRADE. Phase 13: Organizational Mathematics as a question.

We do not implement an "organization layer". We let Cog repeatedly ask:

    "Can I replace these structures with something simpler?"

and answer it from evidence: an accepted searched representation counts as
a simpler alternative for a set of structures only when it covers all the
evidence those structures rest on, with a smaller definition. If, over
time, the answers keep converging on organization-shaped representations,
the hypothesis graduates from philosophy to empirical result — see
docs/00-vision.md.
"""

from __future__ import annotations

from typing import Any

from cog.memory.router import MemoryRouter


class OrganizationalMathematics:
    def __init__(self, memory: MemoryRouter) -> None:
        self.memory = memory

    def simpler_alternative(self, structure_ids: list[str]) -> dict[str, Any] | None:
        """Is there ONE accepted searched representation that explains all
        the evidence behind these structures? Simplicity is measured the
        honest way available today: structure count (n -> 1), with the
        definition size reported for transparency."""
        if len(structure_ids) < 2:
            return None  # nothing to simplify

        evidence: set[str] = set()
        for structure_id in structure_ids:
            record = self.memory.concepts.get(structure_id)
            if record is None:
                return None
            content = record.content
            evidence |= set(content.get("support", []) or content.get("members", []))
        if not evidence:
            return None

        for candidate in self.memory.concepts.search(tags=["searched"], limit=200):
            if candidate.id in structure_ids:
                continue
            covered = set(candidate.content.get("members", []))
            if evidence <= covered:
                return {
                    "candidate": candidate.id,
                    "name": candidate.content.get("name", candidate.id),
                    "replaces": list(structure_ids),
                    "definition_size": len(candidate.content.get("definition", [])),
                    "reduction": f"{len(structure_ids)} structures -> 1 representation",
                }
        return None
