"""Causal credit assignment: learn *why* a retry worked, apply it preemptively.

Cog's retry loop turns a failed attempt into a success by replanning under a
failure hypothesis. Each such failure→success transition (a ``retry_of`` edge
from the success back to the failure) is a natural experiment: holding the
goal fixed, one plan failed and a changed plan succeeded. The minimal change
that flipped the outcome is a *causal correction*.

This engine mines those transitions for the change that **generalizes** — a
tool-selection fix (the retry switched which tools it used) shared across
several goals with a common feature word. It emits a ``CorrectionRule``
keyed by that feature. The WorkspaceBuilder then injects the rule as a
hypothesis *before the first attempt* of a matching new goal, so Cog avoids
the known mistake preemptively instead of paying for the retry again.

Arg-only fixes (same tools, different argument values) are deliberately
ignored: the right value is goal-specific and does not generalize, whereas
"for goals about X, use tool T not tool U" does.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from cog.experience.record import Experience
from cog.memory.router import MemoryRouter

_STOPWORDS = frozenset(
    {"the", "a", "an", "of", "to", "and", "then", "for", "with", "from", "into", "that", "this"}
)


def goal_tokens(goal: str) -> frozenset[str]:
    """Salient goal feature words — lowercased, no digits, no stopwords."""
    tokens = (t for t in re.split(r"[^a-z0-9]+", goal.lower()) if t)
    return frozenset(t for t in tokens if len(t) > 2 and not t.isdigit() and t not in _STOPWORDS)


def _tools(experience: Experience) -> tuple[str, ...]:
    return tuple(step["tool"] for step in experience.execution)


@dataclass
class CorrectionRule:
    triggers: tuple[str, ...]  # goal feature words that activate the rule (all required)
    prefer: tuple[str, ...]  # tools the successful retry used that the failure did not
    avoid: tuple[str, ...]  # tools the failure used that the successful retry dropped
    support: int = 0  # number of failure→success transitions that agree
    sources: list[str] = field(default_factory=list)  # success experience ids

    @property
    def id(self) -> str:
        key = f"{sorted(self.triggers)}|{self.prefer}|{self.avoid}"
        return "corr_" + hashlib.sha1(key.encode()).hexdigest()[:12]

    def hypothesis(self) -> str:
        parts = [
            f"Learned correction (from {self.support} past fix{'es' if self.support != 1 else ''})"
        ]
        parts.append(f"for goals about {', '.join(sorted(self.triggers))}:")
        if self.prefer:
            parts.append(f"prefer the {', '.join(self.prefer)} tool(s)")
        if self.avoid:
            parts.append(f"avoid the {', '.join(self.avoid)} tool(s)")
        return " ".join(parts)


class CorrectionEngine:
    def __init__(self, min_support: int = 2) -> None:
        self.min_support = min_support

    def mine(
        self, experiences: list[Experience], retry_targets: dict[str, list[str]]
    ) -> list[CorrectionRule]:
        by_id = {e.id: e for e in experiences}

        # Group transitions by the (avoid, prefer) tool-change they represent.
        groups: dict[tuple[tuple[str, ...], tuple[str, ...]], list[Experience]] = {}
        for success in experiences:
            if not success.verified:
                continue
            for failed_id in retry_targets.get(success.id, []):
                failed = by_id.get(failed_id)
                if failed is None or failed.verified:
                    continue
                wrong, right = _tools(failed), _tools(success)
                if not right or wrong == right:
                    continue  # empty fix, or an arg-only fix that does not generalize
                prefer = tuple(t for t in right if t not in wrong)
                avoid = tuple(t for t in wrong if t not in right)
                if not prefer:
                    continue
                groups.setdefault((avoid, prefer), []).append(success)

        rules: list[CorrectionRule] = []
        for (avoid, prefer), successes in groups.items():
            if len(successes) < self.min_support:
                continue
            # The rule triggers on the goal words ALL these fixes share.
            shared = frozenset.intersection(*(goal_tokens(s.goal) for s in successes))
            if not shared:
                continue
            rules.append(
                CorrectionRule(
                    triggers=tuple(sorted(shared)),
                    prefer=prefer,
                    avoid=avoid,
                    support=len(successes),
                    sources=sorted(s.id for s in successes),
                )
            )
        return rules


def mine_and_store(memory: MemoryRouter, min_support: int = 2) -> list[CorrectionRule]:
    experiences = [Experience.from_dict(r.content) for r in memory.experiences.search(limit=500)]
    retry_targets = {
        e.id: [dst for _src, dst, _kind in memory.edges_from(e.id, "retry_of")] for e in experiences
    }
    rules = CorrectionEngine(min_support=min_support).mine(experiences, retry_targets)
    for rule in rules:
        memory.concepts.add(
            {
                "level": "correction",
                "triggers": list(rule.triggers),
                "prefer": list(rule.prefer),
                "avoid": list(rule.avoid),
                "support": rule.support,
                "hypothesis": rule.hypothesis(),
            },
            tags=["correction"],
            confidence=min(1.0, rule.support / 5),
            record_id=rule.id,
        )
        for source in rule.sources:
            memory.add_edge(rule.id, source, "corrects")  # Phase 15: link to evidence
    return rules


def corrections_for_goal(memory: MemoryRouter, goal: str, limit: int = 3) -> list[str]:
    """Hypotheses to inject preemptively: every stored correction whose trigger
    words are all present in this goal."""
    tokens = goal_tokens(goal)
    hits: list[tuple[int, str]] = []
    for record in memory.concepts.search(tags=["correction"], limit=200):
        triggers = record.content.get("triggers", [])
        if triggers and set(triggers) <= tokens:
            hits.append((record.content.get("support", 0), record.content["hypothesis"]))
    hits.sort(key=lambda h: -h[0])
    return [hypothesis for _support, hypothesis in hits[:limit]]
