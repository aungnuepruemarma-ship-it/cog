"""Phase 2: the executor.

Runs plan steps through the ToolRouter under the task budget. Failures are
recorded, never swallowed: an errored step stops execution (later steps
usually depend on it) and the whole trace remains in the log.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from cog.execution.log import ActionLog, ActionRecord
from cog.execution.planner import Plan, PlanStep
from cog.execution.router import ToolRouter
from cog.runtime.hooks import HookBus
from cog.runtime.task import Budget
from cog.runtime.trace import ExecutionTrace, StepTrace


@dataclass
class ExecutionResult:
    log: ActionLog
    output: Any = None
    completed: bool = False  # every step ran without error, within budget
    truncated_by_budget: bool = False
    duration_s: float = 0.0
    notes: list[str] = field(default_factory=list)


class Executor:
    def __init__(self, router: ToolRouter, hooks: HookBus | None = None) -> None:
        self.router = router
        self.hooks = hooks or HookBus()

    def run(
        self,
        plan: Plan,
        budget: Budget,
        trace: ExecutionTrace | None = None,
        workspace: object | None = None,
        context: "ExecutionContext | None" = None,
        reorder: "Callable[[list[PlanStep]], list[PlanStep]] | None" = None,
    ) -> ExecutionResult:
        from cog.runtime.context import ExecutionContext

        if context is None:
            task_id = (trace.task_id if trace is not None else "unknown")
            context = ExecutionContext(task_id=task_id, workspace=workspace)
        log = ActionLog()
        result = ExecutionResult(log=log)
        started = time.monotonic()

        # Opt-in step reordering (controlled rollout of ordering heuristics).
        # reorder is a callable(steps) -> steps supplied by the runtime ONLY
        # when an active/experimental policy enables it. Default None = no
        # change to historical behavior.
        steps = reorder(plan.steps) if reorder is not None else plan.steps

        if not steps:
            result.notes.append("empty plan: nothing to execute")

        for index, step in enumerate(steps):
            if index >= budget.max_actions or time.monotonic() - started > budget.max_seconds:
                result.truncated_by_budget = True
                result.notes.append(f"budget exhausted before step {index}")
                break

            record = ActionRecord(index=index, tool=step.tool, args=step.args,
                                  step_id=step.id)
            step_started = time.monotonic()
            try:
                record.result = self.router.dispatch(step.tool, step.args, context=context)
            except Exception as exc:  # noqa: BLE001 - failures are evidence
                record.error = f"{type(exc).__name__}: {exc}"
            record.duration_s = time.monotonic() - step_started
            log.append(record)
            context.record_step(step.tool)
            self.hooks.emit("action", {"record": record})

            # Populate the factual trace (no DB / schema knowledge here).
            if trace is not None:
                trace.add_step(
                    StepTrace(
                        step_id=f"s{index}",
                        tool=step.tool,
                        args=step.args,
                        expected_state={},
                        observed_state={
                            "result": record.result,
                            "error": record.error,
                        },
                        tool_output=(
                            None if record.error is not None else str(record.result)
                        ),
                        error=record.error,
                        duration_s=record.duration_s,
                    )
                )

            if record.error is not None:
                result.notes.append(f"step {index} failed; aborting remaining steps")
                break
            result.output = record.result

        result.duration_s = time.monotonic() - started
        result.completed = (
            bool(plan.steps)
            and not result.truncated_by_budget
            and len(log) == len(plan.steps)
            and not log.errors
        )
        return result
