"""Integration test: real CogRuntime run emits a populated Experience.

Drives the actual executor -> verifier -> emitter wiring (no LLM; a
ScriptedAdapter supplies the plan). Proves the runtime now produces richly
populated, validating Experience records — not empty shells — which is the
exit criterion from the review before any PolicyLifecycle exists.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cog.experience.emitter import ExperienceEmitter
from cog.experience.store import ExperienceStore
from cog.execution.router import ToolRouter
from cog.runtime.adapter import ScriptedAdapter
from cog.runtime.core import CogRuntime
from cog.runtime.task import Budget, Task


class BoomTool:
    name = "boom"
    description = 'A tool that always fails: missing dependency. Args: {"x": 1}'

    def run(self, **kwargs) -> int:
        raise RuntimeError("missing package libxyz during build")


class OkTool:
    name = "ok"
    description = 'A tool that succeeds. Args: {"x": 1}'

    def run(self, **kwargs) -> int:
        return 42


def _runtime(tools) -> CogRuntime:
    adapter = ScriptedAdapter(
        script={
            # The planner prompt contains the goal; emit a one-step plan.
            "Deploy Flask API": "step: boom {} -- trigger missing dep",
            "Double the number": "step: ok {\"x\": 21} -- succeed",
        },
        default="step: ok {\"x\": 1} -- default",
    )
    rt = CogRuntime(
        adapter=adapter,
        storage_dir=Path(tempfile.mkdtemp()),
        tools=tools,
        verification_threshold=0.7,
    )
    return rt


def test_runtime_emits_populated_failure_experience() -> None:
    rt = _runtime([BoomTool()])
    task = Task(
        goal="Deploy Flask API",
        purpose="integration",
        success_criteria=["ran"],
        expected_output=None,
        budget=Budget(max_actions=8, max_seconds=30),
    )
    exp = rt.run(task)

    # The runtime produced an Experience via the new emitter path.
    assert exp is not None
    # It validates (deterministic gate).
    problems = exp.validate()
    assert problems == [], f"experience failed validation: {problems}"

    # Structured evidence is populated (not an empty shell):
    assert exp.outcome == "failure"
    assert exp.replay.environment_snapshot is not None  # reproducibility, every task
    assert exp.replay.seed is None  # task had no seed; explicitly None is fine
    assert exp.failure.category == "dependency_failure"
    assert exp.failure.error_signature == "BOOM_MISSING_LIBXYZ"
    assert exp.failure.failed_step == "s0"
    assert exp.reality_delta.unexpected_conditions  # expected vs actual captured
    assert exp.causal.failure_node == "s0"
    assert "dependency_failure" in exp.causal.caused_by

    # Store it and prove the canonical query + replay work on a real record.
    with tempfile.TemporaryDirectory() as tmp:
        store = ExperienceStore(Path(tmp))
        store.add(exp)  # would raise if invalid
        rows = store.failures_by("dependency_failure", "unspecified")
        assert len(rows) == 1
        replayed = store.replay(exp.id)
        assert replayed is not None
        assert replayed["failure"]["category"] == "dependency_failure"
    print("test_runtime_emits_populated_failure_experience: OK")


def test_runtime_emits_populated_success_experience() -> None:
    rt = _runtime([OkTool()])
    task = Task(
        goal="Double the number",
        purpose="integration",
        success_criteria=["ran"],
        expected_output=42,
        budget=Budget(max_actions=8, max_seconds=30),
    )
    exp = rt.run(task)
    assert exp is not None
    problems = exp.validate()
    assert problems == [], f"experience failed validation: {problems}"
    assert exp.outcome == "success"
    assert exp.failure.category is None  # success carries no failure classification
    assert exp.replay.environment_snapshot is not None
    assert exp.causal.failure_node is None
    print("test_runtime_emits_populated_success_experience: OK")


if __name__ == "__main__":
    test_runtime_emits_populated_failure_experience()
    test_runtime_emits_populated_success_experience()
    print("\nALL RUNTIME-WIRING INTEGRATION TESTS PASSED")
