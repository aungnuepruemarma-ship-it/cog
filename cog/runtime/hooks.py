"""Phase 0: the hook system.

Observers (loggers, UIs, future learning engines) attach to lifecycle events
without coupling to the runtime. A failing handler never breaks the task —
its exception is captured on the bus instead.

Events emitted by the MVP loop: ``task_start``, ``plan_ready``, ``action``,
``verified``, ``task_end``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Handler = Callable[[str, dict[str, Any]], None]


class HookBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}
        self.history: list[tuple[str, dict[str, Any]]] = []
        self.handler_errors: list[tuple[str, str]] = []

    def on(self, event: str, handler: Handler) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def emit(self, event: str, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        self.history.append((event, payload))
        for handler in self._handlers.get(event, []) + self._handlers.get("*", []):
            try:
                handler(event, payload)
            except Exception as exc:  # noqa: BLE001 - observers must not break tasks
                self.handler_errors.append((event, repr(exc)))
