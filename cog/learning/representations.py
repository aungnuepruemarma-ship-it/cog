"""Phase 8: the Representation Engine (weekly loop).

Asks "what is the smallest representation?" — patterns that are *about the
same subject* (import error / dependency error / package error are all
about dependency handling) merge into one representation. Compression is
structural and measurable: len(representations) < len(patterns) merged.
"""

from __future__ import annotations

import hashlib

from cog.learning.artifacts import Pattern, Representation
from cog.memory.router import MemoryRouter


def _representation_id(subject: str) -> str:
    return "rep_" + hashlib.sha1(subject.encode()).hexdigest()[:12]


class RepresentationEngine:
    def __init__(self, min_merge: int = 2) -> None:
        self.min_merge = min_merge

    def reduce(self, patterns: list[Pattern]) -> list[Representation]:
        by_subject: dict[str, list[Pattern]] = {}
        for pattern in patterns:
            if pattern.subject:
                by_subject.setdefault(pattern.subject, []).append(pattern)

        representations: list[Representation] = []
        for subject, group in by_subject.items():
            if len(group) < self.min_merge:
                continue  # nothing to compress yet
            kinds = sorted({p.kind for p in group})
            representations.append(
                Representation(
                    id=_representation_id(subject),
                    name=f"competence: {subject} ({'/'.join(kinds)})",
                    merges=sorted(p.id for p in group),
                )
            )
        return representations


def reduce_and_store(memory: MemoryRouter, min_merge: int = 2) -> list[Representation]:
    pattern_records = memory.concepts.search(tags=["pattern"], limit=500)
    patterns = [
        Pattern(
            id=r.id,
            kind=r.content.get("kind", ""),
            description=r.content.get("description", ""),
            subject=r.content.get("subject", ""),
            support=r.content.get("support", []),
        )
        for r in pattern_records
    ]
    representations = RepresentationEngine(min_merge=min_merge).reduce(patterns)
    for representation in representations:
        memory.concepts.add(
            {
                "level": "representation",
                "name": representation.name,
                "merges": representation.merges,
                "compression": f"{len(representation.merges)} patterns -> 1 representation",
            },
            tags=["representation"],
            record_id=representation.id,
        )
        for pattern_id in representation.merges:
            memory.add_edge(representation.id, pattern_id, "merges")  # Phase 15
    return representations
