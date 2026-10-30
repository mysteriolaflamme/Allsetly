"""Fake model client for offline tests (no network calls)."""

from __future__ import annotations

from typing import Any

from temoin.model import ChatAnswer


class FakeChatClient:
    """Scripted chat client: returns answers or raises errors in order."""

    def __init__(self, script: list[str | ChatAnswer | Exception]) -> None:
        """Store the scripted answers and errors."""
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def chat(self, model: str, messages: list[dict[str, Any]]) -> str:
        """Record the call and play the next scripted item."""
        item = self._play(model, messages, None)
        if isinstance(item, ChatAnswer):
            return item.content
        return item

    def chat_with_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> ChatAnswer:
        """Record the call and play the next scripted item."""
        item = self._play(model, messages, tools)
        if isinstance(item, str):
            return ChatAnswer(content=item, tool_calls=[])
        return item

    def _play(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> str | ChatAnswer:
        """Record the call and pop the next scripted item."""
        self.calls.append({"model": model, "messages": messages, "tools": tools})
        if not self._script:
            raise AssertionError("FakeChatClient script exhausted.")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item
