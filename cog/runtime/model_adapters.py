"""Live model adapters — Cog's bridge out of the scripted world.

The runtime stays model-independent: everything here satisfies the same
``ModelAdapter`` protocol the deterministic ``ScriptedAdapter`` does, so
the executor, verification gate, ledger, and economics treat a live model
exactly like a scripted one. Refusals and empty responses become failed
experiences (evidence), never special cases.

``AnthropicAdapter`` uses the official ``anthropic`` SDK, which is an
optional dependency (``pip install cog-runtime[anthropic]``) — Cog's core
remains stdlib-only. ``OpenAIAdapter`` talks to any OpenAI-compatible chat
endpoint (a local llama.cpp / Ollama / vLLM server, or a hosted provider)
using only the standard library — no dependency at all. Tests inject a
fake client/transport; nothing here is exercised over the network in the
test suite.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Any

_INSTALL_HINT = (
    "the AnthropicAdapter requires the official SDK: pip install anthropic "
    "(or: pip install cog-runtime[anthropic]); credentials resolve from the "
    "environment (ANTHROPIC_API_KEY or an `ant auth login` profile)"
)

# Transport contract: (url, payload, headers, timeout) -> parsed JSON dict.
Transport = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]


def _urllib_transport(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    request = urllib.request.Request(  # noqa: S310 - fixed http(s) endpoint, not user input
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


class CallableAdapter:
    """Wrap any ``str -> str`` callable as a ModelAdapter — the minimal
    surface for embedding Cog in a host application."""

    def __init__(self, fn: Callable[[str], str], name: str = "callable") -> None:
        self.fn = fn
        self.name = name
        self.calls: list[str] = []

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.fn(prompt)


class AnthropicAdapter:
    """Drive Cog with a live Claude model via the Messages API.

    Defaults follow current API guidance: ``claude-opus-4-8``, adaptive
    thinking, no sampling parameters, and a deliberately small
    ``max_tokens`` (Cog plans are short line-based outputs).
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        max_tokens: int = 4096,
        system: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise ImportError(_INSTALL_HINT) from exc
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.system = system
        self.name = f"anthropic:{model}"
        self.calls: list[str] = []
        # Real cost data for reasoning economics (Phase 20).
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system is not None:
            request["system"] = self.system
        response = self.client.messages.create(**request)

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.total_input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.total_output_tokens += getattr(usage, "output_tokens", 0) or 0

        # Check the stop reason BEFORE reading content: a refusal may carry
        # empty or partial content. An empty completion yields an empty plan,
        # which fails verification and is recorded as failure evidence.
        if getattr(response, "stop_reason", None) == "refusal":
            return ""
        return "\n".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )


class OpenAIAdapter:
    """Drive Cog with any OpenAI-compatible chat endpoint — a local
    llama.cpp / Ollama / vLLM server, or a hosted provider — using only the
    standard library.

    This is how Cog runs against an open-source model with no API key: point
    ``base_url`` at a local server (default ``http://127.0.0.1:8000/v1``).
    ``temperature`` defaults to 0 so a small model produces Cog's line-based
    plan format as deterministically as it can; a malformed plan is recorded
    as failure evidence like any other, never a crash.
    """

    def __init__(
        self,
        model: str = "local",
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        system: str | None = None,
        timeout: float = 120.0,
        transport: Transport | None = None,
    ) -> None:
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system = system
        self.timeout = timeout
        self.transport = transport or _urllib_transport
        self.name = f"openai:{model}"
        self.calls: list[str] = []
        # Real cost data for reasoning economics (Phase 20).
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        messages: list[dict[str, str]] = []
        if self.system is not None:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = self.transport(self.url, payload, headers, self.timeout)

        usage = data.get("usage") or {}
        self.total_input_tokens += usage.get("prompt_tokens", 0) or 0
        self.total_output_tokens += usage.get("completion_tokens", 0) or 0

        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        if message.get("refusal"):  # some providers expose an explicit refusal field
            return ""
        return message.get("content") or ""
