"""Phase 3, Policy track: deterministic policy selector (no vector DB).

Given a task context, return the policies whose trigger matches. Matching is
exact field comparison — closer to a compiler optimization pass than RAG. No
embeddings, no fuzzy retrieval. Active/validated policies only; sorted by
confidence so the planner prefers the best-supported intervention.

Trigger fields understood:
    tool        -- the operation the policy applies to
    task_type   -- the task class
    domain      -- the domain
All present trigger keys must match the context. Extra context keys are ignored.
"""

from __future__ import annotations

from typing import Any

from cog.learning.policy.model import Policy, PolicyState
from cog.learning.policy.store import PolicyStore


def _ctx(task_type: str, tools: list[str], domain: str = "software") -> dict[str, Any]:
    return {"task_type": task_type, "tools": set(tools), "domain": domain}


def _matches(context: dict[str, Any], trigger: dict[str, Any]) -> bool:
    tools = context.get("tools", set())
    for key, value in trigger.items():
        if key == "tool":
            if value not in tools:
                return False
        elif key == "task_type":
            if context.get("task_type") != value:
                return False
        elif key == "domain":
            if context.get("domain") != value:
                return False
        else:
            # Unknown trigger key: require exact equality if present in context.
            if key in context and context[key] != value:
                return False
    return True


def select_policies(context: dict[str, Any], policy_store: PolicyStore) -> list[Policy]:
    eligible = policy_store.by_state(PolicyState.ACTIVE) + policy_store.by_state(PolicyState.VALIDATED)
    matched = [p for p in eligible if _matches(context, p.trigger)]
    matched.sort(key=lambda p: p.confidence, reverse=True)
    return matched


def select_for_task(task_type: str, tools: list[str], policy_store: PolicyStore,
                    domain: str = "software") -> list[Policy]:
    return select_policies(_ctx(task_type, tools, domain), policy_store)
