"""End-to-end test of the runtime evidence adapter (ExecutionTrace + Emitter).

Proves the wiring emits *populated* Experience records with the structured
fields the review requires, using a synthetic trace + verification report.
No LLM is involved anywhere in the adapter.
"""

from __future__ import annotations

from cog.experience.emitter import ExperienceEmitter, error_signature, failure_category
from cog.experience.record import Experience
from cog.execution.log import ActionLog, ActionRecord
from cog.execution.planner import Plan, PlanStep
from cog.runtime.task import Task
from cog.runtime.trace import ExecutionTrace, RecoveryAttempt, StepTrace, snapshot_environment
from cog.verification.pipeline import VerificationReport
from cog.workspace.workspace import TaskWorkspace


def _task() -> Task:
    return Task(goal="Deploy Flask API", purpose="test", constraints=["no_network"])


def _workspace(task: Task) -> TaskWorkspace:
    ws = TaskWorkspace(task_id=task.id, goal=task.goal)
    return ws


def _plan() -> Plan:
    return Plan(steps=[PlanStep(tool="docker_build", args={})], prompt="p", raw="raw")


def _failed_log() -> ActionLog:
    log = ActionLog()
    r = ActionRecord(index=0, tool="docker_build", args={})
    r.error = "Command 'docker build' failed: missing package libxyz"
    r.duration_s = 4.0
    log.append(r)
    return log


def _failed_report() -> VerificationReport:
    return VerificationReport(
        results=[], confidence=0.0, verified=False,
        threshold=0.7, required_failures=["container_running"],
    )


def test_emitter_populates_failure_fields() -> None:
    task = _task()
    ws = _workspace(task)
    plan = _plan()
    trace = ExecutionTrace(task_id=task.id, seed=1234, task_version="v1",
                           environment_snapshot=snapshot_environment(ws))
    # Executor would populate this; we simulate the factual trace directly.
    trace.add_step(StepTrace(
        step_id="s0", tool="docker_build", args={},
        expected_state={"container": "running"},
        observed_state={"exit_code": 127, "error": "missing package libxyz"},
        error="Command 'docker build' failed: missing package libxyz",
        duration_s=4.0,
        assumptions=["python3 installed", "port 8080 available"],
    ))
    trace.add_recovery(RecoveryAttempt(
        step_id="s0", action_taken="install_missing_package", result="success",
    ))

    emitter = ExperienceEmitter(default_domain="software", default_difficulty="medium")
    exp = emitter.emit(task, ws, plan, trace, _failed_log(), _failed_report(),
                       strategy="model_plan", attempt=1)

    # Validation gate passes on a populated record.
    assert exp.validate() == [], exp.validate()

    # Phase A: ReplayInfo on every task
    assert exp.replay.seed == 1234
    assert exp.replay.task_version == "v1"
    assert exp.replay.environment_snapshot is not None

    # Phase A: FailureInfo deterministic + categorised
    assert exp.failure.category == "dependency_failure"
    assert exp.failure.error_signature == "DOCKER_BUILD_MISSING_LIBXYZ"
    assert exp.failure.failed_step == "s0"

    # Phase A: RealityDelta populated on verification failure
    assert exp.reality_delta.unexpected_conditions, "reality delta must be set"
    assert "expected=" in exp.reality_delta.unexpected_conditions[0]

    # Phase B: BeliefState (operational assumptions only)
    assert exp.belief.assumptions == ["python3 installed", "port 8080 available"]

    # Phase B: CausalGraph deterministic
    assert exp.causal.failure_node == "s0"
    assert "dependency_failure" in exp.causal.caused_by
    assert "invalid_assumption" in exp.causal.caused_by

    # Phase B: Resolution from recovery
    assert exp.resolution.attempted is True
    assert exp.resolution.action_taken == "install_missing_package"
    assert exp.resolution.result == "success"

    print("test_emitter_populates_failure_fields: OK")


def test_emitter_success_has_no_failure_classification() -> None:
    task = _task()
    ws = _workspace(task)
    plan = _plan()
    trace = ExecutionTrace(task_id=task.id)
    trace.add_step(StepTrace(
        step_id="s0", tool="docker_build", args={},
        observed_state={"image": "ok"}, tool_output="ok",
    ))
    log = ActionLog()
    log.append(ActionRecord(index=0, tool="docker_build", args={}, result={"image": "ok"}))
    report = VerificationReport(results=[], confidence=0.95, verified=True,
                                threshold=0.7, required_failures=[])
    emitter = ExperienceEmitter()
    exp = emitter.emit(task, ws, plan, trace, log, report, strategy="compiled_skill")
    assert exp.validate() == [], exp.validate()
    assert exp.failure.category is None
    assert exp.causal.failure_node is None
    assert exp.replay.environment_snapshot is not None  # still captured on success
    print("test_emitter_success_has_no_failure_classification: OK")


def test_deterministic_signature_classification() -> None:
    # Signature must be stable + informative, never raw prose.
    assert error_signature("docker failed: missing package libxyz", "docker_build") \
        == "DOCKER_BUILD_MISSING_LIBXYZ"
    assert failure_category("permission denied: /root") == "permission_failure"
    assert failure_category("connection refused") == "network_failure"
    assert error_signature("port 8080 already in use", "docker_run") \
        == "DOCKER_RUN_PORT_IN_USE"
    print("test_deterministic_signature_classification: OK")


def test_env_snapshot_is_deterministic() -> None:
    ws = _workspace(_task())
    h1 = snapshot_environment(ws)
    h2 = snapshot_environment(ws)
    assert h1 == h2
    assert h1.startswith("sha256:")
    print("test_env_snapshot_is_deterministic: OK")


if __name__ == "__main__":
    test_emitter_populates_failure_fields()
    test_emitter_success_has_no_failure_classification()
    test_deterministic_signature_classification()
    test_env_snapshot_is_deterministic()
    print("\nALL EMITTER/TRACE TESTS PASSED")
