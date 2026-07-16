"""Run one task through the entire Cog core loop and print every artifact.

python -m cog.demo
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from cog import CogRuntime, ScriptedAdapter, Task


def main() -> None:
    # A deterministic "model": when the planner prompt mentions the goal,
    # it answers with a two-step plan in Cog's wire format.
    adapter = ScriptedAdapter(
        script={
            "compound interest": (
                'step: calculator {"expression": "1000 * (1 + 0.05) ** 10"} '
                "-- 10 years of 5% growth on 1000\n"
                'step: note {"text": "principal 1000 at 5% for 10y"} -- record the assumption\n'
                'step: calculator {"expression": "1000 * (1 + 0.05) ** 10 - 1000"} '
                "-- interest earned\n"
            )
        }
    )

    with tempfile.TemporaryDirectory(prefix="cog-demo-") as tmp:
        runtime = CogRuntime(adapter, storage_dir=Path(tmp))
        runtime.hooks.on(
            "action",
            lambda _e, p: print(
                f"  action[{p['record'].index}] {p['record'].tool}"
                f" -> {p['record'].error or p['record'].result}"
            ),
        )

        task = Task(
            goal="Compute the compound interest earned on 1000 at 5% over 10 years",
            purpose="demonstrate the Cog core loop end to end",
            constraints=["arithmetic only", "stay within budget"],
            success_criteria=["final output is the interest earned"],
            expected_output=lambda out: abs(float(out) - 628.894626777442) < 1e-6,
        )

        print(f"goal: {task.goal}\n")
        print("executing:")
        experience = runtime.run(task)

        print("\nworkspace snapshot:")
        print(json.dumps({k: v for k, v in experience.workspace.items() if v}, indent=2)[:800])

        print("\nverification:")
        for result in experience.verification["results"]:
            print(
                f"  {result['name']}: passed={result['passed']} score={result['score']:.2f}"
                f" ({result['details']})"
            )
        print(f"  confidence={experience.confidence} verified={experience.verified}")

        print("\nmetrics:", experience.metrics)
        print(f"outcome: {experience.outcome}")
        print(f"experience id: {experience.id}")

        edges = runtime.graph.edges_from(experience.id)
        print(f"graph edges: {[(e.kind, e.dst) for e in edges]}")
        facts = runtime.memory.facts.search(query="compound interest", limit=3)
        print(f"facts now retrievable: {[f.content['statement'] for f in facts]}")
        runtime.close()


if __name__ == "__main__":
    main()
