"""Fake model client for offline tests (no network calls)."""

from __future__ import annotations

from typing import Any


class FakeChatClient:
    """Scripted chat client: returns answers or raises errors in order."""

    def __init__(self, script: list[str | Exception]) -> None:
        """Store the scripted answers and errors."""
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def chat(self, model: str, messages: list[dict[str, Any]]) -> str:
        """Record the call and play the next scripted item."""
        self.calls.append({"model": model, "messages": messages})
        if not self._script:
            raise AssertionError("FakeChatClient script exhausted.")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
