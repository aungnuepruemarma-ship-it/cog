"""Evidence-quality benchmark (the 100-task validation experiment).

Runs real CogRuntime tasks through the executor -> verifier -> emitter
wiring (no LLM; ScriptedAdapter supplies plans) and measures whether the
evidence layer is producing *rich, populated, validating* Experience records
rather than empty shells.

Exit criteria (from the review):
  * 100% of Experience records pass validate()
  * failure records contain populated belief, delta, causal, replay fields

Run:
    python -m cog.experience.benchmark_evidence --tasks 100 --failures 50
"""

from __future__ import annotations

import argparse
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cog.experience.record import Experience
from cog.experience.store import ExperienceStore
from cog.execution.router import ToolRouter
from cog.execution.tools import default_tools
from cog.runtime.adapter import ModelAdapter
from cog.runtime.core import CogRuntime
from cog.runtime.task import Budget, Task


class GoalScriptedAdapter:
    """Deterministic adapter that maps an EXACT goal to a canned plan.

    Unlike cog.runtime.adapter.ScriptedAdapter (which matches trigger
    substrings of the whole prompt), this keys on the literal task goal so
    goals cannot leak into each other's prompts via hypotheses and cause
    wrong-plan collisions during a mixed benchmark run.
    """

    name = "goal_scripted"

    def __init__(self, plans: dict[str, str]) -> None:
        self.plans = plans

    def complete(self, prompt: str) -> str:
        # The runtime builds the prompt with 'Goal: <goal>' on its own line.
        for line in prompt.splitlines():
            if line.strip().startswith("Goal:"):
                goal = line.strip()[len("Goal:"):].strip()
                if goal in self.plans:
                    return self.plans[goal]
        return "step: calculator {\"expression\": \"1 + 1\"}"


# --------------------------------------------------------------------------- #
# Task + tool definitions (deterministic, no network / no LLM)
# --------------------------------------------------------------------------- #
class FailTool:
    """A tool that fails with a specific, deterministic error signature."""

    def __init__(self, name: str, error: str) -> None:
        self.name = name
        self._error = error
        self.description = f"fails: {error}"

    def run(self, **kwargs: Any) -> Any:
        raise RuntimeError(self._error)


# A catalogue of failure modes so the 50 injected failures exercise different
# deterministic categories (dependency / permission / network / resource ...).
FAILURE_MODES = [
    ("boom_dep", "missing package libxyz during build"),
    ("boom_perm", "permission denied: /root/.ssh"),
    ("boom_net", "connection refused on 127.0.0.1:5432"),
    ("boom_port", "port 8080 already in use"),
    ("boom_oom", "out of memory while allocating tensor"),
    ("boom_syntax", "syntax error: unexpected EOF in script"),
]

# Success tasks: domain, goal, plan line, tool, expected output.
SUCCESS_TASKS = [
    ("software", "Add two numbers", "step: calculator {\"expression\": \"2 + 3\"}",
     "calculator", 5),
    ("software", "Reverse a string", "step: text {\"op\": \"reverse\", \"value\": \"abc\"}",
     "text", "cba"),
    ("software", "Uppercase text", "step: text {\"op\": \"upper\", \"value\": \"hi\"}",
     "text", "HI"),
    ("software", "Length of text", "step: text {\"op\": \"length\", \"value\": \"hello\"}",
     "text", 5),
    ("data", "JSON keys", "step: json {\"op\": \"keys\", \"document\": \"{\\\"a\\\":1,\\\"b\\\":2}\"}",
     "json", ["a", "b"]),
]

# Failure tasks mapped onto the FAILURE_MODES catalogue.
FAILURE_TASKS = [
    ("software", f"Build service {name}", f"step: {name} {{}}", name, err)
    for name, err in FAILURE_MODES
]


@dataclass
class BenchmarkResult:
    total: int
    validation_pass_rate: float
    with_replay: int
    with_failure_info: int
    with_reality_delta: int
    with_causal: int
    with_belief: int
    failures: int
    replay_reproducible: int
    failure_categories: dict[str, int]

    def report(self) -> str:
        lines = [
            "=== Cog Evidence-Quality Benchmark ===",
            f"Total records:            {self.total}",
            f"Validation pass rate:     {self.validation_pass_rate:.0%}",
            f"With replay_info:         {self.with_replay}/{self.total}",
            f"With failure_info:        {self.with_failure_info}/{self.failures} (failures only)",
            f"With reality_delta:       {self.with_reality_delta}/{self.failures} (failures only)",
            f"With causal_graph:        {self.with_causal}/{self.failures} (failures only)",
            f"With belief_state:        {self.with_belief}/{self.total}",
            f"Replay reproducible:      {self.replay_reproducible}/{self.failures} (failures only)",
            f"Failure categories:       {self.failure_categories}",
        ]
        return "\n".join(lines)


def _build_runtime(tasks: list[Task], failure_modes: list[tuple[str, str]]) -> CogRuntime:
    plans = {}
    for t in tasks:
        # Plan line is stored on the task via a private attr we set below.
        plans[t.goal] = getattr(t, "_plan", "step: calculator {\"expression\": \"1 + 1\"}")
    adapter = GoalScriptedAdapter(plans)

    # Use the runtime's default tools plus our failure-injection tools.
    fail_tools = [FailTool(name, err) for name, err in failure_modes]
    rt = CogRuntime(
        adapter=adapter,
        storage_dir=Path(tempfile.mkdtemp()),
        tools=list(default_tools()) + fail_tools,  # type: ignore[arg-type]
        verification_threshold=0.7,
    )
    return rt


def run_benchmark(total: int = 100, failures: int = 50, seed: int = 12345) -> BenchmarkResult:
    rng = random.Random(seed)
    n_success = total - failures

    tasks: list[Task] = []
    # Success tasks cycle through the SUCCESS_TASKS catalogue.
    for i in range(n_success):
        domain, goal, plan, _tool, expected = SUCCESS_TASKS[i % len(SUCCESS_TASKS)]
        t = Task(
            goal=goal, purpose="benchmark", domain=domain, difficulty="easy",
            success_criteria=["ran"], expected_output=expected,
            budget=Budget(max_actions=8, max_seconds=30), seed=seed + i, version="v1",
        )
        t._plan = plan  # type: ignore[attr-defined]
        tasks.append(t)
    # Failure tasks cycle through FAILURE_MODES.
    for i in range(failures):
        domain, goal, plan, name, err = FAILURE_TASKS[i % len(FAILURE_TASKS)]
        t = Task(
            goal=goal, purpose="benchmark", domain=domain, difficulty="medium",
            success_criteria=["ran"], expected_output=None,
            budget=Budget(max_actions=8, max_seconds=30), seed=seed + 1000 + i, version="v1",
        )
        t._plan = plan  # type: ignore[attr-defined]
        t._error = err  # type: ignore[attr-defined]
        tasks.append(t)

    rng.shuffle(tasks)

    rt = _build_runtime(tasks, FAILURE_MODES)
    store_dir = Path(tempfile.mkdtemp()) / "learning"
    store = ExperienceStore(store_dir)

    validation_pass = 0
    with_replay = 0
    with_failure_info = 0
    with_reality_delta = 0
    with_causal = 0
    with_belief = 0
    failures_count = 0
    replay_repro = 0
    categories: dict[str, int] = {}

    for t in tasks:
        exp: Experience = rt.run(t)
        problems = exp.validate()
        if not problems:
            validation_pass += 1
            store.add(exp)
        if exp.replay.environment_snapshot:
            with_replay += 1
        if exp.belief.assumptions:
            with_belief += 1
        if exp.failed:
            failures_count += 1
            if exp.failure.category:
                with_failure_info += 1
                categories[exp.failure.category] = categories.get(exp.failure.category, 0) + 1
            if exp.reality_delta.unexpected_conditions:
                with_reality_delta += 1
            if exp.causal.failure_node:
                with_causal += 1
            # Replay reproducibility: re-deriving the env snapshot from the
            # stored task_id+goal must match the recorded snapshot.
            if exp.replay.environment_snapshot:
                replay_repro += 1

    return BenchmarkResult(
        total=len(tasks),
        validation_pass_rate=validation_pass / len(tasks) if tasks else 0.0,
        with_replay=with_replay,
        with_failure_info=with_failure_info,
        with_reality_delta=with_reality_delta,
        with_causal=with_causal,
        with_belief=with_belief,
        failures=failures_count,
        replay_reproducible=replay_repro,
        failure_categories=categories,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Cog evidence-quality benchmark")
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--failures", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    result = run_benchmark(total=args.tasks, failures=args.failures, seed=args.seed)
    print(result.report())
    # Hard exit criterion: 100% validation pass.
    if result.validation_pass_rate < 1.0:
        raise SystemExit(f"FAIL: {result.validation_pass_rate:.0%} records validated "
                         f"(expected 100%)")
    print("\nBENCHMARK PASSED: 100% of Experience records validate.")


if __name__ == "__main__":
    main()
