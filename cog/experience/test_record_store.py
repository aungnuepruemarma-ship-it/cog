"""Deterministic end-to-end test for the Experience evidence pipeline (v0.1).

No LLM is invoked. We construct synthetic runtime objects (Task, Workspace,
Plan, ExecutionResult, VerificationReport), build Experiences the same way the
executor would, validate them, store them, and prove the canonical query +
replay work. This is the "replay 100 past executions" foundation the review
asks for *before* any policy extractor exists.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cog.experience.record import (
    BeliefState,
    CausalGraph,
    Experience,
    FailureInfo,
    RealityDelta,
    ReplayInfo,
    Resolution,
)
from cog.experience.store import ExperienceStore
from cog.execution.executor import ExecutionResult
from cog.execution.log import ActionLog, ActionRecord
from cog.execution.planner import Plan, PlanStep
from cog.runtime.task import Task
from cog.verification.pipeline import VerificationReport
from cog.workspace.workspace import TaskWorkspace


def _make_task(goal: str) -> Task:
    return Task(
        goal=goal,
        purpose="test",
        constraints=["no_network"],
        success_criteria=["container_running"],
        budget=__import__("cog.runtime.task", fromlist=["Budget"]).Budget(
            max_actions=16, max_seconds=60
        ),
    )


def _make_workspace(task: Task) -> TaskWorkspace:
    ws = TaskWorkspace(task_id=task.id, goal=task.goal, purpose=task.purpose)
    ws.hypotheses = []
    return ws


def _make_plan() -> Plan:
    return Plan(
        steps=[PlanStep(tool="docker_build", args={}, description="build image")],
        prompt="plan",
        raw="step: docker_build {} -- build image",
        rejected=[],
    )


def _failed_execution() -> ExecutionResult:
    log = ActionLog()
    rec = ActionRecord(index=0, tool="docker_build", args={})
    rec.error = "exit 127: missing package"
    rec.duration_s = 4.0
    log.append(rec)
    return ExecutionResult(log=log, completed=False, duration_s=4.0)


def _failed_report() -> VerificationReport:
    return VerificationReport(
        results=[],
        confidence=0.0,
        verified=False,
        threshold=0.7,
        required_failures=["container_running"],
    )


def _success_execution() -> ExecutionResult:
    log = ActionLog()
    rec = ActionRecord(index=0, tool="docker_build", args={}, result={"image": "ok"})
    rec.duration_s = 3.0
    log.append(rec)
    return ExecutionResult(log=log, completed=True, duration_s=3.0, output={"image": "ok"})


def _success_report() -> VerificationReport:
    return VerificationReport(
        results=[],
        confidence=0.95,
        verified=True,
        threshold=0.7,
        required_failures=[],
    )


def test_failure_experience_pipeline() -> None:
    task = _make_task("Deploy Flask API")
    ws = _make_workspace(task)
    plan = _make_plan()
    exec_res = _failed_execution()
    report = _failed_report()

    exp = Experience.from_run(
        task,
        ws,
        plan,
        exec_res,
        report,
        strategy="model_plan",
        domain="software",
        difficulty="medium",
        belief=BeliefState(assumptions=["python package available", "port is free"]),
        reality_delta=RealityDelta(unexpected_conditions=["package missing"]),
        failure=FailureInfo(
            category="dependency_failure",
            error_signature="missing_package",
            failed_step="docker_build",
        ),
        resolution=Resolution(
            attempted=True, action_taken="install_missing_package", result="success"
        ),
        causal=CausalGraph(
            failure_node="docker_build",
            caused_by=["missing_dependency", "invalid_assumption"],
        ),
        replay=ReplayInfo(
            seed=12345, environment_snapshot="sha256:abc", task_version="v1"
        ),
    )

    # 1) validation passes (the deterministic gate)
    problems = exp.validate()
    assert problems == [], f"unexpected validation problems: {problems}"

    # 2) serialise -> deserialise round trip is lossless
    restored = Experience.from_dict(exp.to_dict())
    assert restored.failure.category == "dependency_failure"
    assert restored.belief.assumptions == ["python package available", "port is free"]
    assert restored.causal.caused_by == ["missing_dependency", "invalid_assumption"]

    # 3) store + canonical query
    with tempfile.TemporaryDirectory() as tmp:
        store = ExperienceStore(Path(tmp))
        store.add(exp)
        assert store.count() == 1

        rows = store.failures_by("dependency_failure", "software")
        assert len(rows) == 1
        assert rows[0]["id"] == exp.id
        assert rows[0]["failure"]["category"] == "dependency_failure"

        # 4) replay returns the verbatim structured evidence
        replayed = store.replay(exp.id)
        assert replayed is not None
        assert replayed["replay"]["seed"] == 12345
        assert replayed["causal"]["failure_node"] == "docker_build"

        # 5) stats aggregate
        stats = store.stats()
        assert stats["total"] == 1
        assert stats["by_outcome"].get("failure") == 1

    print("test_failure_experience_pipeline: OK")


def test_success_experience_pipeline() -> None:
    task = _make_task("Deploy Flask API")
    ws = _make_workspace(task)
    plan = _make_plan()
    exec_res = _success_execution()
    report = _success_report()

    exp = Experience.from_run(
        task,
        ws,
        plan,
        exec_res,
        report,
        strategy="compiled_skill",
        domain="software",
        difficulty="easy",
    )
    problems = exp.validate()
    assert problems == [], f"unexpected validation problems: {problems}"
    assert exp.failure.category is None  # success carries no failure classification

    with tempfile.TemporaryDirectory() as tmp:
        store = ExperienceStore(Path(tmp))
        store.add(exp)
        rows = store.filter(domain="software", outcome="success", verified=True)
        assert len(rows) == 1
    print("test_success_experience_pipeline: OK")


def test_validation_rejects_noisy_records() -> None:
    # A "success" outcome with an unverified report must be rejected.
    task = _make_task("x")
    ws = _make_workspace(task)
    plan = _make_plan()
    exp = Experience.from_run(
        task, ws, plan, _failed_execution(), _failed_report(), domain="software"
    )
    # Force an inconsistent state the gate must catch.
    exp.outcome = "success"
    problems = exp.validate()
    assert any("verification.verified is False" in p for p in problems)

    # A failure without a failure_category must be rejected.
    exp2 = Experience.from_run(
        task, ws, plan, _failed_execution(), _failed_report(), domain="software"
    )
    exp2.failure.category = None
    exp2.failure.error_signature = None
    problems2 = exp2.validate()
    assert any("failure.category" in p for p in problems2)
    print("test_validation_rejects_noisy_records: OK")


if __name__ == "__main__":
    test_failure_experience_pipeline()
    test_success_experience_pipeline()
    test_validation_rejects_noisy_records()
    print("\nALL EXPERIENCE-PIPELINE TESTS PASSED")
