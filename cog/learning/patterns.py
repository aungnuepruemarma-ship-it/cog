"""Phase 7: the Pattern Engine (daily loop).

Discovers regularities in the Experience Graph with minimum evidence
support: repeated failures (grouped by failing tool + error type), repeated
successes (grouped by tool workflow), and repeated tool usage. Failures
were stored with full fidelity precisely so this engine could find them.
"""

from __future__ import annotations

import hashlib

from cog.experience.record import Experience
from cog.learning.artifacts import Pattern
from cog.memory.router import MemoryRouter


def _pattern_id(kind: str, key: str) -> str:
    return "pat_" + hashlib.sha1(f"{kind}|{key}".encode()).hexdigest()[:12]


class PatternEngine:
    def __init__(self, min_support: int = 2) -> None:
        self.min_support = min_support

    def discover(self, experiences: list[Experience]) -> list[Pattern]:
        failures: dict[tuple[str, str], list[str]] = {}
        successes: dict[tuple[str, ...], list[str]] = {}
        tool_usage: dict[str, set[str]] = {}

        for experience in experiences:
            tools = [step["tool"] for step in experience.execution]
            for tool in set(tools):
                tool_usage.setdefault(tool, set()).add(experience.id)

            if experience.verified:
                if tools:
                    successes.setdefault(tuple(tools), []).append(experience.id)
                continue

            errored = [s for s in experience.execution if s.get("error")]
            if errored:
                step = errored[0]
                error_type = str(step["error"]).split(":", 1)[0]
                key = (step["tool"], error_type)
            else:
                key = ("<plan>", "NoExecutableSteps" if not tools else "WrongOutput")
            failures.setdefault(key, []).append(experience.id)

        patterns: list[Pattern] = []
        for (tool, error_type), support in failures.items():
            if len(support) < self.min_support:
                continue
            description = f"repeated failure: {tool} fails with {error_type}"
            patterns.append(
                Pattern(
                    id=_pattern_id("failure", f"{tool}|{error_type}"),
                    kind="failure",
                    description=description,
                    subject=tool,
                    support=sorted(support),
                )
            )
        for workflow, support in successes.items():
            if len(support) < self.min_support:
                continue
            description = f"repeated success: workflow {' -> '.join(workflow)}"
            patterns.append(
                Pattern(
                    id=_pattern_id("success", "|".join(workflow)),
                    kind="success",
                    description=description,
                    subject=workflow[-1],  # the workflow's decisive tool
                    support=sorted(support),
                )
            )
        for tool, ids in tool_usage.items():
            if len(ids) < self.min_support:
                continue
            description = f"repeated tool usage: {tool} across {len(ids)} experiences"
            patterns.append(
                Pattern(
                    id=_pattern_id("tool_usage", tool),
                    kind="tool_usage",
                    description=description,
                    subject=tool,
                    support=sorted(ids),
                )
            )
        return patterns


def discover_and_store(memory: MemoryRouter, min_support: int = 2) -> list[Pattern]:
    experiences = [Experience.from_dict(r.content) for r in memory.experiences.search(limit=500)]
    patterns = PatternEngine(min_support=min_support).discover(experiences)
    for pattern in patterns:
        memory.concepts.add(
            {
                "level": "pattern",
                "kind": pattern.kind,
                "description": pattern.description,
                "subject": pattern.subject,
                "support": pattern.support,
            },
            tags=["pattern", pattern.kind],
            confidence=min(1.0, len(pattern.support) / 10),  # more evidence, more trust
            record_id=pattern.id,
        )
        for experience_id in pattern.support:
            memory.add_edge(pattern.id, experience_id, "supports")  # Phase 15
    return patterns
