"""Tests for the model gateway: retries, cache and JSON validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from temoin.model import ModelError, build_cache_key, complete_json
from temoin.schemas import ReportDraft
from tests.fake_client import FakeChatClient


def _complete(
    tmp_path: Path,
    client: FakeChatClient,
    *,
    cache_key: str = "key-1",
    json_retries: int = 1,
) -> ReportDraft:
    """Run complete_json against a fake client with test-friendly waits."""
    return complete_json(
        [{"role": "user", "content": "hello"}],
        ReportDraft,
        client=client,
        model_name="test-model",
        cache_key=cache_key,
        cache_dir=tmp_path / "cache",
        max_attempts=2,
        retry_wait_seconds=0.0,
        json_retries=json_retries,
    )


def test_complete_json_validates_answer(tmp_path: Path, draft_json: str) -> None:
    """A valid answer is validated and the model name is passed through."""
    client = FakeChatClient([draft_json])
    draft = _complete(tmp_path, client)
    assert isinstance(draft, ReportDraft)
    assert draft.challenge.read_code == "K7"
    assert client.calls[0]["model"] == "test-model"


def test_complete_json_strips_fences(tmp_path: Path, draft_json: str) -> None:
    """Markdown fences around the JSON are tolerated."""
    client = FakeChatClient([f"```json\n{draft_json}\n```"])
    assert _complete(tmp_path, client).description != ""


def test_complete_json_retries_invalid_json_with_error_feedback(
    tmp_path: Path, draft_json: str
) -> None:
    """After a bad answer, the error is sent back for one new attempt."""
    client = FakeChatClient(["not json at all", draft_json])
    draft = _complete(tmp_path, client)
    assert isinstance(draft, ReportDraft)
    assert len(client.calls) == 2
    retry_messages = client.calls[1]["messages"]
    assert len(retry_messages) == 3
    assert "STRICT JSON" in retry_messages[2]["content"]


def test_complete_json_gives_up_with_plain_error(tmp_path: Path) -> None:
    """Two bad answers raise a plain-language error."""
    client = FakeChatClient(["xxx", "yyy"])
    with pytest.raises(ModelError, match="pas compris"):
        _complete(tmp_path, client)


def test_complete_json_without_retry_calls_once(tmp_path: Path) -> None:
    """With json_retries=0, a bad answer fails immediately."""
    client = FakeChatClient(["xxx", "yyy"])
    with pytest.raises(ModelError):
        _complete(tmp_path, client, json_retries=0)
    assert len(client.calls) == 1


def test_complete_json_retries_transport_errors(tmp_path: Path, draft_json: str) -> None:
    """A connection failure is retried once before succeeding."""
    client = FakeChatClient([ConnectionError("boom"), draft_json])
    assert isinstance(_complete(tmp_path, client), ReportDraft)
    assert len(client.calls) == 2


def test_complete_json_gives_up_when_service_down(tmp_path: Path) -> None:
    """Two transport failures raise a plain-language error."""
    client = FakeChatClient([ConnectionError("a"), ConnectionError("b")])
    with pytest.raises(ModelError, match="injoignable"):
        _complete(tmp_path, client)


def test_complete_json_uses_disk_cache(tmp_path: Path, draft_json: str) -> None:
    """The second identical request is served from disk without a new call."""
    first = FakeChatClient([draft_json])
    _complete(tmp_path, first, cache_key="same")
    assert len(first.calls) == 1
    second = FakeChatClient([])
    draft = _complete(tmp_path, second, cache_key="same")
    assert isinstance(draft, ReportDraft)
    assert second.calls == []


def test_cache_key_changes_with_inputs() -> None:
    """Image, version and code all shape the cache key."""
    base = build_cache_key(image_sha256="aaa", expected_code="K7")
    assert base == build_cache_key(image_sha256="aaa", expected_code="K7")
    assert base != build_cache_key(image_sha256="bbb", expected_code="K7")
    assert base != build_cache_key(image_sha256="aaa", expected_code="X9")
