"""Phase 4: shared SQLite persistence for the five memory stores.

One inspectable ``memory.db`` file, one ``records`` table, one store class
per memory kind. Retrieval is structured (tags + keyword scoring); the
DocumentStore's keyword scoring is the MVP stand-in for RAG.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cog._util import new_id, utc_now

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_kind ON records(kind);

CREATE TABLE IF NOT EXISTS edges (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (src, dst, kind)
);
"""


@dataclass
class MemoryRecord:
    id: str
    kind: str
    content: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    created_at: str = field(default_factory=utc_now)
    score: float = 0.0  # retrieval-time relevance, not persisted


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    return conn


def _tokens(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


class BaseStore:
    kind = "record"

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def add(
        self,
        content: dict[str, Any],
        tags: list[str] | None = None,
        confidence: float = 1.0,
        record_id: str | None = None,
    ) -> MemoryRecord:
        record = MemoryRecord(
            id=record_id or new_id(self.kind),
            kind=self.kind,
            content=content,
            tags=list(tags or []),
            confidence=confidence,
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO records (id, kind, content, tags, confidence, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.kind,
                json.dumps(record.content),
                json.dumps(record.tags),
                record.confidence,
                record.created_at,
            ),
        )
        # No per-row commit: callers batch writes into one transaction and
        # commit once (see CogRuntime.run / run_learning_cycle). Committing per
        # add made runtime.run() O(n^2) as the DB grew (a checkpoint per write).
        return record

    def get(self, record_id: str) -> MemoryRecord | None:
        row = self.conn.execute(
            "SELECT id, kind, content, tags, confidence, created_at FROM records"
            " WHERE id = ? AND kind = ?",
            (record_id, self.kind),
        ).fetchone()
        return self._to_record(row) if row else None

    def search(
        self, query: str = "", tags: list[str] | None = None, limit: int = 10,
        _scan_cap: int = 256,
    ) -> list[MemoryRecord]:
        """Keyword-scored retrieval: rank by query-token hits, then recency.

        ``_scan_cap`` bounds how many rows are pulled from the DB per call
        (most-recent first) so retrieval stays O(cap) instead of O(table) --
        this is what keeps ``runtime.run()`` (which calls search for prior
        similar experiences) linear as memory grows, instead of quadratic.
        """
        fetch = max(limit, _scan_cap)
        rows = self.conn.execute(
            "SELECT id, kind, content, tags, confidence, created_at FROM records"
            " WHERE kind = ? ORDER BY created_at DESC LIMIT ?",
            (self.kind, fetch),
        ).fetchall()
        records = [self._to_record(row) for row in rows]
        if tags:
            wanted = set(tags)
            records = [r for r in records if wanted & set(r.tags)]
        query_tokens = _tokens(query)
        if query_tokens:
            for record in records:
                haystack = _tokens(json.dumps(record.content) + " " + " ".join(record.tags))
                record.score = sum(haystack.count(t) for t in query_tokens)
            records = [r for r in records if r.score > 0]
            # Stable sort keeps the SQL recency order as the tie-breaker.
            records.sort(key=lambda r: -r.score)
        return records[:limit]

    def count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM records WHERE kind = ?", (self.kind,)
        ).fetchone()
        return int(row[0])

    @staticmethod
    def _to_record(row: tuple) -> MemoryRecord:
        return MemoryRecord(
            id=row[0],
            kind=row[1],
            content=json.loads(row[2]),
            tags=json.loads(row[3]),
            confidence=row[4],
            created_at=row[5],
        )
