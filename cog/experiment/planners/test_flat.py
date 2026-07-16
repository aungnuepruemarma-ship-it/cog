"""Tests for the Exp3 baseline FlatPlanner.

Run:  python -m cog.experiment.planners.test_flat
These tests PROVE the control group has no dependency/decomposition capability,
so Exp3 cannot be contaminated by a "smart baseline". They need NO pytest.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from cog.experiment.planners.flat import FlatPlanner
from cog.execution.planner import Plan
from cog.workspace.workspace import TaskWorkspace
from cog.runtime.task import Budget


class _FixedAdapter:
    """Deterministic fake model: returns a preset raw plan, no reasoning."""

    def __init__(self, raw: str) -> None:
        self._raw = raw
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return self._raw


class _StubRouter:
    def specs(self):
        return [
            {"name": "calculator", "description": "evaluate an expression"},
            {"name": "text", "description": "text ops"},
        ]


RAW = (
    'step: calculator {"expression": "2 + 2"} -- add\n'
    'step: text {"op": "reverse", "value": "cog"} -- flip\n'
)


def _ws() -> TaskWorkspace:
    return TaskWorkspace(task_id="t_demo", goal="demo goal", purpose="demo", budget=Budget(max_actions=5))


def test_capability_flags_are_flat():
    cap = FlatPlanner.CAPABILITIES
    assert cap["hierarchy"] is False
    assert cap["decomposition"] is False
    assert cap["dependency_graph"] is False
    assert cap["lookahead"] is False
    assert cap["recovery_policy"] is False
    assert cap["linear_only"] is True
    print("PASS test_capability_flags_are_flat")


def test_plan_is_linear_no_structure():
    p = FlatPlanner(_FixedAdapter(RAW), _StubRouter())
    plan = p.plan(_ws())
    assert isinstance(plan, Plan)
    # exactly the two steps, in emitted order, no grouping
    assert [s.tool for s in plan.steps] == ["calculator", "text"]
    assert len(plan.steps) == 2
    # no step carries dependency/hierarchy metadata
    for s in plan.steps:
        assert "depends_on" not in s.args
        assert "subgoal" not in s.args
    print("PASS test_plan_is_linear_no_structure")


def test_deterministic_given_same_adapter():
    a = FlatPlanner(_FixedAdapter(RAW), _StubRouter())
    b = FlatPlanner(_FixedAdapter(RAW), _StubRouter())
    pa, pb = a.plan(_ws()), b.plan(_ws())
    assert [s.to_dict() for s in pa.steps] == [s.to_dict() for s in pb.steps]
    print("PASS test_deterministic_given_same_adapter")


def test_no_recovery_policy_on_failure():
    # FlatPlanner must NOT rewrite/retry planning on a malformed step; it just
    # records the rejected line and returns what it lexed.
    bad = 'step: calculator {"expression": 2 + 2} -- broken json\n'
    p = FlatPlanner(_FixedAdapter(bad), _StubRouter())
    plan = p.plan(_ws())
    assert len(plan.steps) == 0
    assert len(plan.rejected) == 1
    print("PASS test_no_recovery_policy_on_failure")


def test_behavior_hash_stable():
    h1 = FlatPlanner.behavior_hash()
    h2 = FlatPlanner.behavior_hash()
    assert h1 == h2
    assert len(h1) == 16
    print(f"PASS test_behavior_hash_stable ({h1})")


def main() -> int:
    test_capability_flags_are_flat()
    test_plan_is_linear_no_structure()
    test_deterministic_given_same_adapter()
    test_no_recovery_policy_on_failure()
    test_behavior_hash_stable()
    print("\nALL FLATPLANNER TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
