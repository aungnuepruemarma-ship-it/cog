"""Phase 0: session lifecycle.

A session scopes one runtime instance: where it stores memory, when it
started, and which tasks it has run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from cog._util import new_id, utc_now


@dataclass
class Session:
    storage_dir: Path
    id: str = field(default_factory=lambda: new_id("session"))
    started_at: str = field(default_factory=utc_now)
    task_ids: list[str] = field(default_factory=list)
