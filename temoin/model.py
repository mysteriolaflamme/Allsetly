"""Single gateway to the vision model.

Handles timeouts, retries with progressive waits, disk cache and JSON
validation. Nothing else in the codebase talks to the model.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from temoin.prompts import COMPARE_PROMPT_VERSION, PROMPT_VERSION, build_json_retry_message

logger = logging.getLogger(__name__)

ResponseT = TypeVar("ResponseT", bound=BaseModel)
_T = TypeVar("_T")


class ModelError(Exception):
    """Raised when the model cannot produce a usable answer."""


@dataclass(frozen=True)
class ToolCall:
    """One function call requested by the model."""

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatAnswer:
    """A chat answer: final text plus optional tool calls."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatClient(Protocol):
    """Minimal interface of a chat-completion client."""

    def chat(self, model: str, messages: list[dict[str, Any]]) -> str:
        """Send messages and return the final text answer."""
        ...

    def chat_with_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> ChatAnswer:
        """Send messages and return the text answer plus tool calls."""
        ...


class OpenAIChatClient:
    """Adapter from the OpenAI SDK to the ChatClient protocol."""

    def __init__(self, *, api_key: str, base_url: str, timeout_seconds: float) -> None:
        """Store credentials; the SDK client is created per call."""
        self._api_key = api_key
        self._base_url = base_url
        self._timeout_seconds = timeout_seconds

    def chat(self, model: str, messages: list[dict[str, Any]]) -> str:
        """Call chat completions and return the final message content only."""
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        response = client.chat.completions.create(
            model=model, messages=messages, timeout=self._timeout_seconds
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ModelError("Le modèle n'a rien répondu. Réessayez dans un moment.")
        return content

    def chat_with_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> ChatAnswer:
        """Call chat completions and return text plus requested tool calls."""
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        extra: dict[str, Any] = {"tools": tools} if tools is not None else {}
        response = client.chat.completions.create(
            model=model, messages=messages, timeout=self._timeout_seconds, **extra
        )
        message = response.choices[0].message
        calls = [
            ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments)
            for call in (message.tool_calls or [])
        ]
        return ChatAnswer(content=message.content or "", tool_calls=calls)


def default_client(*, api_key: str, base_url: str, timeout_seconds: float) -> ChatClient:
    """Build the real model-backed chat client."""
    return OpenAIChatClient(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)


def build_cache_key(*, image_sha256: str, expected_code: str) -> str:
    """Derive a disk-cache key from the image, the prompt version and the code."""
    raw = f"{image_sha256}|{PROMPT_VERSION}|{expected_code}".encode()
    return sha256(raw).hexdigest()


def build_compare_cache_key(*, sale_constat: str, delivery_constat: str) -> str:
    """Derive a disk-cache key from both constats and the compare prompt version."""
    raw = f"{sale_constat}|{delivery_constat}|{COMPARE_PROMPT_VERSION}".encode()
    return sha256(raw).hexdigest()


def read_cache_json(cache_dir: Path, key: str) -> dict[str, Any] | None:
    """Return a raw cache payload, or None on any miss or corruption."""
    try:
        payload = json.loads((cache_dir / f"{key}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def write_cache_json(cache_dir: Path, key: str, payload: dict[str, Any]) -> None:
    """Store a cache payload; a cache failure never breaks the run."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")
    except OSError as error:
        logger.warning("Could not write model cache: %s", error)


def _read_cache(cache_file: Path) -> str | None:
    """Return the cached answer, or None on any miss or stale entry."""
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("prompt_version") != PROMPT_VERSION:
        return None
    content = payload.get("content")
    return content if isinstance(content, str) else None


def _write_cache(cache_file: Path, content: str) -> None:
    """Store a raw answer; a cache failure never breaks the constat."""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"prompt_version": PROMPT_VERSION, "content": content}
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as error:
        logger.warning("Could not write model cache: %s", error)


def validate_content(content: str, response_model: type[ResponseT]) -> ResponseT:
    """Extract the first {...} block and validate it against the model."""
    start, end = content.find("{"), content.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in the model answer")
    return response_model.model_validate_json(content[start:end])


def call_with_retries(
    call: Callable[[], _T],
    *,
    max_attempts: int,
    retry_wait_seconds: float,
) -> _T:
    """Run a model call, retrying transport failures with progressive waits."""
    last_error: Exception | None = None
    for attempt in range(max(1, max_attempts)):
        try:
            return call()
        except Exception as error:
            last_error = error
            logger.warning("Model call failed (attempt %d): %s", attempt + 1, error)
            if attempt + 1 < max_attempts:
                time.sleep(retry_wait_seconds * (attempt + 1))
    raise ModelError(
        "Le service d'analyse est injoignable. Vérifiez votre connexion puis réessayez."
    ) from last_error


def complete_json(
    messages: list[dict[str, Any]],
    response_model: type[ResponseT],
    *,
    client: ChatClient,
    model_name: str,
    cache_key: str,
    cache_dir: Path,
    max_attempts: int,
    retry_wait_seconds: float,
    json_retries: int,
    extra_check: Callable[[ResponseT], None] | None = None,
) -> ResponseT:
    """Query the model and return a validated response, using cache if possible.

    On invalid JSON, the error is sent back to the model for a new attempt.
    An optional extra_check runs after validation and feeds the same retry.
    """
    cache_file = cache_dir / f"{cache_key}.json"
    cached = _read_cache(cache_file)
    if cached is not None:
        try:
            return validate_content(cached, response_model)
        except ValueError:
            logger.warning("Ignoring invalid cache entry: %s", cache_file)
    current_messages = list(messages)
    last_error = "unknown error"
    for _ in range(max(0, json_retries) + 1):
        raw = call_with_retries(
            lambda messages=current_messages: client.chat(model_name, messages),
            max_attempts=max_attempts,
            retry_wait_seconds=retry_wait_seconds,
        )
        try:
            validated = validate_content(raw, response_model)
            if extra_check is not None:
                extra_check(validated)
        except ValueError as error:
            last_error = str(error)
            current_messages = [
                *current_messages,
                {"role": "assistant", "content": raw},
                build_json_retry_message(last_error),
            ]
            continue
        _write_cache(cache_file, raw)
        return validated
    logger.warning("Model answer rejected: %s", last_error)
    raise ModelError(
        "Le modèle n'a pas compris la photo. Reprenez-la de plus près, bien nette, puis réessayez."
    )
