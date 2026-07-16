"""Phase 4: the Memory Router.

Callers ask for what they need; the router fans out to the right stores and
merges ranked results. Writes derived from a task go through
``write_from_experience`` — the mechanical enforcement of the verification
gate: an unverified experience produces no fact/skill writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from cog._util import utc_now
from cog.memory.base import MemoryRecord, open_db
from cog.memory.stores import (
    ConceptStore,
    DocumentStore,
    ExperienceStore,
    FactStore,
    SkillStore,
)

if TYPE_CHECKING:
    from cog.experience.record import Experience

_DEFAULT_KINDS = ("fact", "skill", "concept", "experience", "document")


class MemoryRouter:
    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = Path(storage_dir)
        self.conn = open_db(self.storage_dir / "memory.db")
        self.facts = FactStore(self.conn)
        self.experiences = ExperienceStore(self.conn)
        self.skills = SkillStore(self.conn)
        self.concepts = ConceptStore(self.conn)
        self.documents = DocumentStore(self.conn)
        self._stores = {
            "fact": self.facts,
            "experience": self.experiences,
            "skill": self.skills,
            "concept": self.concepts,
            "document": self.documents,
        }

    def retrieve(
        self, query: str, kinds: tuple[str, ...] | None = None, limit: int = 5
    ) -> list[MemoryRecord]:
        """Fan out to the requested stores, merge by relevance score."""
        merged: list[MemoryRecord] = []
        for kind in kinds or _DEFAULT_KINDS:
            store = self._stores[kind]
            merged.extend(store.search(query=query, limit=limit))
        # Belief revision (Phase: beliefs) marks contradicted facts "superseded";
        # they stay on disk as an audit trail but must never be retrieved as
        # trustworthy context again.
        merged = [r for r in merged if "superseded" not in r.tags]
        merged.sort(key=lambda r: -r.score)
        return merged[:limit]

    def add_edge(self, src: str, dst: str, kind: str) -> None:
        """Typed link in the knowledge graph (Phase 15). Artifacts link to the
        evidence below them: skill→experience, pattern→experience,
        representation→pattern, general skill→compressed skill."""
        self.conn.execute(
            "INSERT OR IGNORE INTO edges (src, dst, kind, created_at) VALUES (?, ?, ?, ?)",
            (src, dst, kind, utc_now()),
        )
        # No per-edge commit: batched with other writes, committed once per
        # run()/learn() cycle (see MemoryRouter / CogRuntime).

    def edges_from(self, src: str, kind: str | None = None) -> list[tuple[str, str, str]]:
        if kind is None:
            rows = self.conn.execute(
                "SELECT src, dst, kind FROM edges WHERE src = ?", (src,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT src, dst, kind FROM edges WHERE src = ? AND kind = ?", (src, kind)
            ).fetchall()
        return [tuple(row) for row in rows]

    def write_from_experience(self, experience: Experience) -> list[MemoryRecord]:
        """The verification gate. Only a verified experience writes facts."""
        if not experience.verified:
            return []
        written = [
            self.facts.add(
                {
                    "statement": f"Goal {experience.goal!r} was solved; output: "
                    f"{experience.output!r}",
                    "goal": experience.goal,
                    "output": experience.output,
                    "source_experience": experience.id,
                },
                tags=["derived"],
                confidence=experience.confidence,
            )
        ]
        return written

    def close(self) -> None:
        self.conn.close()
