"""Phase 2: the action log.

Every action is logged — tool, args, result or error, duration, timestamp —
success or failure alike. This log is raw material for the Pattern Engine
(Phase 7) and Reasoning Economics (Phase 20), so completeness beats brevity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cog._util import json_safe, utc_now


@dataclass
class ActionRecord:
    index: int
    tool: str
    args: dict[str, Any]
    result: Any = None
    error: str | None = None
    duration_s: float = 0.0
    timestamp: str = field(default_factory=utc_now)
    step_id: str | None = None  # logical plan step id (e.g. "s0"), set by executor

    @property
    def ok(self) -> bool:
        return self.error is None


class ActionLog:
    def __init__(self) -> None:
        self.records: list[ActionRecord] = []

    def append(self, record: ActionRecord) -> None:
        self.records.append(record)

    @property
    def errors(self) -> list[ActionRecord]:
        return [r for r in self.records if not r.ok]

    def __len__(self) -> int:
        return len(self.records)

    def to_dicts(self) -> list[dict[str, Any]]:
        out = []
        for record in self.records:
            data = asdict(record)
            data["result"] = json_safe(data["result"])
            data["args"] = json_safe(data["args"])
            out.append(data)
        return out
