"""Phase 6: the Skill Compiler — program induction from verified traces.

Repeated verified workflows compress into *parameterized* skills:

- a group of experiences with the same tool/arg-shape signature is found;
- arg values that vary across the group must be recoverable from the goal
  text, and become parameters;
- the goals, with those values substituted out, must collapse to one shared
  goal template.

The compiled skill carries a goal regex. A future task whose goal matches
the regex can be solved by replaying the skill's steps with the captured
parameter values — **zero model calls**. That is the compression promise
made concrete: instead of replaying long traces (or re-planning), Cog
replays one induced program.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from cog.execution.planner import Plan, PlanStep
from cog.experience.record import Experience
from cog.learning.artifacts import Skill
from cog.memory.base import MemoryRecord
from cog.memory.router import MemoryRouter
from cog.memory.stores import SkillStore
from cog.science.ledger import Ledger

_PLACEHOLDER_RE = re.compile(r"\{(p\d+)\}")

Signature = tuple[tuple[str, tuple[str, ...]], ...]


def _signature(experience: Experience) -> Signature | None:
    """Tool sequence + arg shape of a clean run; None if any step errored."""
    steps = experience.execution
    if not steps or any(step.get("error") for step in steps):
        return None
    return tuple((step["tool"], tuple(sorted(step.get("args", {})))) for step in steps)


def _skill_id(template: str, signature: Signature) -> str:
    digest = hashlib.sha1(f"{template}|{signature}".encode()).hexdigest()[:12]
    return f"skill_{digest}"


def _goal_shape(signature: Signature, experience: Experience) -> str:
    """The goal with every goal-recoverable arg value blanked out — the
    grouping key that separates parameterizable workflows from exact ones."""
    shape = experience.goal
    for step_index, (_tool, arg_keys) in enumerate(signature):
        for key in arg_keys:
            value = experience.execution[step_index]["args"][key]
            if isinstance(value, str) and value in shape:
                shape = shape.replace(value, "\x00", 1)
    return shape


def _template_to_regex(template: str) -> str:
    regex = re.escape(template)
    # re.escape escapes { and }; rewrite escaped placeholders as capture
    # groups. Non-greedy, so multi-parameter templates split at the literal
    # separators between placeholders instead of one group swallowing them.
    return re.sub(r"\\\{p\d+\\\}", "(.+?)", regex)


class SkillCompiler:
    """Phase 6 (hourly loop): experiences -> executable, parameterized skills."""

    def __init__(self, min_support: int = 2) -> None:
        self.min_support = min_support

    def compile(self, experiences: list[Experience]) -> list[Skill]:
        # Group by (signature, per-experience goal shape). The second key
        # keeps unrelated goals out of each other's groups, so one workflow
        # whose args never surface in its goal cannot poison the
        # parameterization of another that shares its tool signature.
        groups: dict[tuple[Signature, str], list[Experience]] = {}
        for experience in experiences:
            if not experience.verified:
                continue
            signature = _signature(experience)
            if signature is not None:
                shape = _goal_shape(signature, experience)
                groups.setdefault((signature, shape), []).append(experience)

        skills: list[Skill] = []
        for (signature, _shape), group in groups.items():
            if len(group) < self.min_support:
                continue
            skill = self._templatize(signature, group)
            if skill is not None:
                skills.append(skill)
        return skills

    def _templatize(self, signature: Signature, group: list[Experience]) -> Skill | None:
        param_index = 0
        goal_templates = [e.goal for e in group]
        step_templates: list[dict[str, Any]] = []

        for step_index, (tool, arg_keys) in enumerate(signature):
            template_args: dict[str, Any] = {}
            for key in arg_keys:
                values = [e.execution[step_index]["args"][key] for e in group]
                if all(v == values[0] for v in values):
                    template_args[key] = values[0]
                    continue
                # Varying value: it must be recoverable from each goal text.
                pairs = zip(values, group, strict=True)
                if not all(isinstance(v, str) and v in e.goal for v, e in pairs):
                    return None
                name = f"p{param_index}"
                param_index += 1
                for i, value in enumerate(values):
                    if value not in goal_templates[i]:
                        return None
                    goal_templates[i] = goal_templates[i].replace(value, "{" + name + "}", 1)
                template_args[key] = "{" + name + "}"
            step_templates.append({"tool": tool, "args": template_args, "description": ""})

        if len(set(goal_templates)) != 1:
            return None  # the goals do not share one template — not one skill
        template = goal_templates[0]
        parameters = _PLACEHOLDER_RE.findall(template)
        if param_index and len(parameters) != param_index:
            return None  # a parameter never surfaced in the goal template

        tools = "_".join(tool for tool, _ in signature)
        source_ids = [e.id for e in group]
        return Skill(
            id=_skill_id(template, signature),
            name=f"replay_{tools}",
            steps=step_templates,
            goal_template=template,
            parameters=parameters,
            source_experiences=source_ids,
            benchmark_score=sum(e.confidence for e in group) / len(group),
        )


def compile_and_store(memory: MemoryRouter, min_support: int = 2) -> list[Skill]:
    """Run the compiler over the ExperienceStore; upsert results into the
    SkillStore. Existing skills keep their evolved confidence and use count."""
    experiences = [Experience.from_dict(r.content) for r in memory.experiences.search(limit=500)]
    skills = SkillCompiler(min_support=min_support).compile(experiences)
    for skill in skills:
        existing = memory.skills.get(skill.id)
        content = {
            "name": skill.name,
            "goal_template": skill.goal_template,
            "goal_regex": _template_to_regex(skill.goal_template),
            "parameters": skill.parameters,
            "steps": skill.steps,
            "source_experiences": skill.source_experiences,
            "uses": existing.content.get("uses", 0) if existing else 0,
        }
        memory.skills.add(
            content,
            tags=list(existing.tags) if existing else ["compiled"],
            # Evolved confidence survives recompilation: evidence beats freshness.
            confidence=existing.confidence if existing else skill.benchmark_score,
            record_id=skill.id,
        )
        for source_id in skill.source_experiences:
            memory.add_edge(skill.id, source_id, "compiled_from")  # Phase 15: link to evidence
        if existing is None:  # a new belief enters Cog -> it gets a ledger claim
            Ledger(memory).record_claim(
                subject_id=skill.id,
                hypothesis=(
                    f"goals matching {skill.goal_template!r} are solved by replaying"
                    f" {[s['tool'] for s in skill.steps]}"
                ),
                experiment="skill compilation over verified experiences (min_support gate)",
                dataset=skill.source_experiences,
                metrics={
                    "support": len(skill.source_experiences),
                    "mean_confidence": round(skill.benchmark_score, 4),
                },
                decision="adopted",
                confidence=skill.benchmark_score,
                claim_id=f"claim_{skill.id}_compiled",
            )
    return skills


def match_skill(
    store: SkillStore, goal: str, min_confidence: float = 0.0
) -> tuple[MemoryRecord, dict[str, str]] | None:
    """Best non-retired skill whose goal regex matches, with bound params."""
    best: tuple[MemoryRecord, dict[str, str]] | None = None
    for record in store.search(limit=200):
        if {"retired", "compressed"} & set(record.tags) or record.confidence < min_confidence:
            continue
        regex = record.content.get("goal_regex")
        if not regex:
            continue
        matched = re.fullmatch(regex, goal)
        if not matched:
            continue
        bound = dict(zip(record.content.get("parameters", []), matched.groups(), strict=False))
        if best is None or record.confidence > best[0].confidence:
            best = (record, bound)
    return best


def instantiate_plan(record: MemoryRecord, bound: dict[str, str]) -> Plan:
    """Fill a skill's step templates with the values captured from the goal."""
    steps = []
    for step in record.content["steps"]:
        args = {key: _fill(value, bound) for key, value in step["args"].items()}
        steps.append(
            PlanStep(tool=step["tool"], args=args, description=step.get("description", ""))
        )
    return Plan(steps=steps, prompt="", raw=f"skill:{record.id}")


def _fill(value: Any, bound: dict[str, str]) -> Any:
    if isinstance(value, str):
        for name, captured in bound.items():
            value = value.replace("{" + name + "}", captured)
    return value
