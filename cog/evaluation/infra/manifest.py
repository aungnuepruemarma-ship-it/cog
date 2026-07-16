"""Evaluation infrastructure: experiment manifest (reproducibility).

Every suite run emits a manifest capturing cog version, eval version, seed,
configuration, dataset params, artifacts, and results. This is what makes
benchmark numbers comparable across time and across machines.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _cog_version() -> str:
    try:
        from cog import __version__ as v
        return str(v)
    except Exception:
        return "0.6.2"  # current working version


@dataclass
class Manifest:
    manifest_version: str = "1.0"
    eval_suite_version: str = "v1.0.0"
    cog_version: str = field(default_factory=_cog_version)
    experiment_id: str = ""
    seed: int = 42
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    platform: str = field(default_factory=lambda: platform.platform())
    configuration: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    baselines: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
