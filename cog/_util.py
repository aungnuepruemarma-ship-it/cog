"""Small shared helpers."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def json_safe(value: Any) -> Any:
    """Coerce a value into something json.dumps accepts, falling back to repr."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)
