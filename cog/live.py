"""Run Cog against a live model.

    python -m cog.live "Compute 12 * 34"            # hosted Claude (needs a key)
    python -m cog.live --local "Compute 12 * 34"    # a local OpenAI-compatible server

Hosted mode needs the optional anthropic extra and credentials. Local
mode needs an OpenAI-compatible server (llama.cpp / Ollama / vLLM) at
``COG_OPENAI_BASE_URL`` (default http://127.0.0.1:8000/v1) and no key —
it drives Cog with an open-source model over the standard library alone.
Memory persists in ./.cog/ so verified experience accumulates across
invocations; the learning cycle runs after each task.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from cog import CogRuntime, Task
from cog.runtime.adapter import ModelAdapter
from cog.science.ledger import Ledger

DEFAULT_STORAGE = Path(".cog")

PLANNER_SYSTEM = (
    "You plan tool programs for the Cog intelligence runtime. Respond only "
    "with step lines in the exact wire format requested; no prose."
)


def build_runtime(adapter: ModelAdapter, storage_dir: Path | str = DEFAULT_STORAGE) -> CogRuntime:
    return CogRuntime(adapter, storage_dir=storage_dir)


def run_goal(runtime: CogRuntime, goal: str) -> None:
    experience = runtime.run(Task(goal=goal))
    report = runtime.learn()

    print(f"goal:       {goal}")
    print(f"strategy:   {experience.strategy}")
    print(f"outcome:    {experience.outcome} (confidence {experience.confidence})")
    print(f"output:     {experience.output!r}")
    print(f"actions:    {[step['tool'] for step in experience.execution]}")
    adapter = runtime.adapter
    if hasattr(adapter, "total_input_tokens"):
        print(
            f"tokens:     {adapter.total_input_tokens} in /"
            f" {adapter.total_output_tokens} out"
        )
    if report.skills_compiled:
        ledger = Ledger(runtime.memory)
        for record in runtime.memory.skills.search(limit=3):
            for claim in ledger.why(record.id):
                print(f"skill:      {record.content.get('name')} — {claim['hypothesis']}")


def _build_adapter(local: bool) -> ModelAdapter:
    if local:
        from cog.runtime.model_adapters import OpenAIAdapter

        return OpenAIAdapter(
            model=os.environ.get("COG_OPENAI_MODEL", "local"),
            base_url=os.environ.get("COG_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.environ.get("COG_OPENAI_API_KEY"),
            system=PLANNER_SYSTEM,
        )
    from cog.runtime.model_adapters import AnthropicAdapter

    return AnthropicAdapter(system=PLANNER_SYSTEM)


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    local = False
    if args and args[0] == "--local":
        local = True
        args = args[1:]
    if not args:
        print('usage: python -m cog.live [--local] "<goal>"')
        return 2
    try:
        adapter = _build_adapter(local)
    except ImportError as exc:
        print(f"cannot start live mode: {exc}")
        return 1

    runtime = build_runtime(adapter)
    try:
        run_goal(runtime, " ".join(args))
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
