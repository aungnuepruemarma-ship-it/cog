"""Phase 19: Domain Discovery — no manual labels.

Experiences cluster by their *behavioral signature* (the set of tools the
task actually needed), not by human curriculum. Whatever the evidence
clusters into IS the domain. As the tool surface grows (sandbox, browser,
terminal), the same clustering yields richer emergent domains without a
single label being written by hand.
"""

from __future__ import annotations

import hashlib

from cog.experience.record import Experience
from cog.memory.router import MemoryRouter


class DomainDiscovery:
    def cluster(self, experiences: list[Experience]) -> dict[str, list[str]]:
        domains: dict[str, list[str]] = {}
        for experience in experiences:
            tools = sorted({step["tool"] for step in experience.execution})
            name = "+".join(tools) if tools else "abstract"
            domains.setdefault(name, []).append(experience.id)
        return domains


def discover_and_store_domains(memory: MemoryRouter, min_members: int = 2) -> dict[str, list[str]]:
    experiences = [Experience.from_dict(r.content) for r in memory.experiences.search(limit=500)]
    domains = {
        name: members
        for name, members in DomainDiscovery().cluster(experiences).items()
        if len(members) >= min_members
    }
    for name, members in domains.items():
        domain_id = "dom_" + hashlib.sha1(name.encode()).hexdigest()[:12]
        memory.concepts.add(
            {"level": "domain", "name": name, "members": sorted(members)},
            tags=["domain"],
            confidence=min(1.0, len(members) / 10),
            record_id=domain_id,
        )
        for member_id in members:
            memory.add_edge(domain_id, member_id, "clusters")  # Phase 15
    return domains
