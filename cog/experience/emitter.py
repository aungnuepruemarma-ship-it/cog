"""Phase 5: the ExperienceEmitter — turns runtime facts into an Experience.

This is the ONLY place that maps an ExecutionTrace + VerificationReport onto
the structured Experience fields (BeliefState, RealityDelta, FailureInfo,
CausalGraph, ReplayInfo, Resolution). It contains no LLM calls: every
classification here is deterministic so the evidence layer stays trustworthy.

Priority order (per the review):
  Phase A (critical evidence)
    1. ReplayInfo      -- reproducibility, every task
    2. FailureInfo     -- deterministic category + error_signature, every failure
    3. RealityDelta    -- expected vs actual, every verification failure
  Phase B (richer cognition)
    4. BeliefState     -- explicit operational assumptions only
    5. CausalGraph     -- deterministic dependency links, no LLM root-cause yet
    6. Resolution      -- recovery outcome if a recovery was attempted
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from cog._util import new_id, utc_now
from cog.execution.planner import Plan
from cog.experience.record import (
    BeliefState,
    CausalGraph,
    Experience,
    ExperienceContext,
    ExperienceMetrics,
    FailureInfo,
    RealityDelta,
    ReplayInfo,
    Resolution,
)
from cog.runtime.task import Task
from cog.runtime.trace import ExecutionTrace, StepTrace, snapshot_environment
from cog.verification.pipeline import VerificationReport
from cog.workspace.workspace import TaskWorkspace


# --------------------------------------------------------------------------- #
# Deterministic classifiers (no model, no randomness)
# --------------------------------------------------------------------------- #
# Ordered, specific-first. Each (pattern, category, signature_template) maps a
# raw error string to a stable failure category + UPPER_SNAKE error signature.
_FAILURE_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"missing|no such|not found|command not found", re.I),
     "dependency_failure", "MISSING_{token}"),
    (re.compile(r"permission denied|eacces|denied", re.I),
     "permission_failure", "PERMISSION_DENIED"),
    (re.compile(r"connection refused|timeout|network is unreachable", re.I),
     "network_failure", "NETWORK_UNREACHABLE"),
    (re.compile(r"port .* (in use|already allocated)|address already in use", re.I),
     "resource_conflict", "PORT_IN_USE"),
    (re.compile(r"out of memory|oom|disk full|no space", re.I),
     "resource_exhaustion", "RESOURCE_EXHAUSTED"),
    (re.compile(r"syntax error|parse error|invalid syntax", re.I),
     "syntax_failure", "SYNTAX_ERROR"),
    # Generic exit-code 127 last: it is the least informative signal and
    # should only fire when no more specific pattern matched.
    (re.compile(r"exit code 127|127:", re.I),
     "dependency_failure", "MISSING_COMMAND"),
]


def error_signature(raw_error: str | None, tool: str = "") -> str:
    """Deterministic, comparable signature. Never the raw prose.

    Bad (what we avoid):  "docker failed"
    Good (what we emit):  DOCKER_BUILD_MISSING_PACKAGE_LIBXYZ
    """
    if not raw_error:
        return "UNKNOWN"
    text = raw_error.strip()
    for pattern, _cat, template in _FAILURE_RULES:
        m = pattern.search(text)
        if m:
            token = _extract_token(text, m)
            sig = template.format(token=token) if "{token}" in template else template
            prefix = tool.upper().replace("-", "_") or "RUN"
            return f"{prefix}_{sig}"
    # Fallback: stable, sanitised upper-snake from the first meaningful words.
    words = re.findall(r"[A-Za-z0-9]+", text)[:4]
    return "ERR_" + "_".join(w.upper() for w in words) or "ERR_UNKNOWN"


def failure_category(raw_error: str | None) -> str | None:
    if not raw_error:
        return None
    for pattern, category, _sig in _FAILURE_RULES:
        if pattern.search(raw_error):
            return category
    return "unclassified_failure"


def _extract_token(text: str, match: re.Match[str] | None = None) -> str:
    """Pull a package/file token for the signature.

    Prefers a token near the regex match (e.g. the word after 'missing'),
    so 'missing package libxyz' yields LIBXYZ rather than the first word.
    """
    if match is not None:
        # Look at the remainder of the string starting at the match.
        tail = text[match.start():]
        tokens = re.findall(r"[A-Za-z0-9_.\-]+", tail)
        # Skip the matched keyword itself; take the next meaningful token.
        for tok in tokens[1:]:
            if tok.lower() not in ("package", "command", "failed", "error"):
                return tok.upper()
    m = re.search(r"([A-Za-z0-9_.\-]+)", text)
    return (m.group(1) if m else "DEP").upper()


def build_causal(trace: ExecutionTrace, failed: StepTrace) -> CausalGraph:
    """Deterministic causal link for v0.1.

    Later versions can upgrade this with learned inference; for now the link
    is structural: the failed step depends on the immediately preceding step's
    output, so a prior failure propagates.
    """
    caused_by: list[str] = []
    if failed.error:
        cat = failure_category(failed.error)
        if cat:
            caused_by.append(cat)
    if failed.assumptions:
        caused_by.append("invalid_assumption")
    # If an earlier step also failed, that is the structural root.
    idx = trace.steps.index(failed) if failed in trace.steps else -1
    if idx > 0:
        prev = trace.steps[idx - 1]
        if prev.error:
            caused_by.append(f"depends_on:{prev.step_id}")
    return CausalGraph(failure_node=failed.step_id, caused_by=caused_by)


# --------------------------------------------------------------------------- #
# The emitter
# --------------------------------------------------------------------------- #
class ExperienceEmitter:
    """Converts a populated ExecutionTrace + verification into an Experience.

    Deterministic. The runtime owns facts; this owns the schema mapping.
    """

    def __init__(self, default_domain: str = "unspecified",
                 default_difficulty: str = "unspecified") -> None:
        self.default_domain = default_domain
        self.default_difficulty = default_difficulty

    def emit(
        self,
        task: Task,
        workspace: TaskWorkspace,
        plan: Plan,
        trace: ExecutionTrace,
        execution_log: Any,  # cog.execution.log.ActionLog
        report: VerificationReport,
        strategy: str = "model_plan",
        attempt: int = 1,
        *,
        domain: str | None = None,
        difficulty: str | None = None,
    ) -> Experience:
        failed = trace.failed_steps()
        first_failure = failed[0] if failed else None

        # 1. ReplayInfo — every task, for reproducibility
        # If the trace didn't carry a snapshot (e.g. older caller), derive one
        # deterministically from the workspace so replay is never empty.
        env_snapshot = trace.environment_snapshot or snapshot_environment(workspace)
        replay = ReplayInfo(
            seed=trace.seed,
            environment_snapshot=env_snapshot,
            task_version=trace.task_version,
        )

        # 2. FailureInfo — deterministic, every failure
        if first_failure is not None:
            raw = first_failure.error
            failure = FailureInfo(
                category=failure_category(raw),
                error_signature=error_signature(raw, first_failure.tool),
                root_cause=None,  # reserved for later learned inference
                failed_step=first_failure.step_id,
            )
        else:
            failure = FailureInfo()

        # 3. RealityDelta — expected vs actual on verification failure
        if not report.verified and first_failure is not None:
            reality_delta = RealityDelta(
                unexpected_conditions=[
                    f"expected={first_failure.expected_state or 'ok'} "
                    f"actual={first_failure.observed_state or first_failure.error}"
                ],
                note="; ".join(report.required_failures) or first_failure.error,
            )
        else:
            reality_delta = RealityDelta()

        # 4. BeliefState — explicit operational assumptions only
        assumptions: list[str] = []
        for s in trace.steps:
            assumptions.extend(s.assumptions)
        belief = BeliefState(assumptions=assumptions)

        # 5. CausalGraph — deterministic, only when something failed
        causal = build_causal(trace, first_failure) if first_failure is not None else CausalGraph()

        # 6. Resolution — only if a recovery was attempted
        if trace.recovery_attempts:
            last = trace.recovery_attempts[-1]
            resolution = Resolution(
                attempted=True,
                action_taken=last.action_taken,
                result=last.result,
            )
        else:
            resolution = Resolution()

        metrics = ExperienceMetrics(
            duration_s=round(sum(s.duration_s for s in trace.steps), 6),
            actions=len(trace.steps),
            errors=len(failed),
            budget_max_actions=task.budget.max_actions,
            confidence=report.confidence,
            attempt=attempt,
            model_calls=1 if strategy == "model_plan" else 0,
        )

        return Experience(
            id=new_id("exp"),
            task_id=task.id,
            goal=task.goal,
            purpose=task.purpose,
            domain=domain or self.default_domain,
            difficulty=difficulty or self.default_difficulty,
            constraints=list(task.constraints),
            success_criteria=list(task.success_criteria),
            context=ExperienceContext(
                available_tools=list(getattr(workspace, "available_tools", [])),
                constraints=list(task.constraints),
            ),
            belief=belief,
            reality_delta=reality_delta,
            workspace=workspace.persist_snapshot(),
            reasoning={
                "prompt": plan.prompt,
                "raw_plan": plan.raw,
                "rejected": plan.rejected,
                "strategy": strategy,
            },
            execution=execution_log.to_dicts(),
            verification=report.to_dict(),
            metrics=metrics,
            failure=failure,
            resolution=resolution,
            causal=causal,
            replay=replay,
            outcome="success" if report.verified else "failure",
            output=None,  # populated by caller if needed
            created_at=utc_now(),
        )
