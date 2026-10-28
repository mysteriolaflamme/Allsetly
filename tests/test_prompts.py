"""Tests for the versioned model prompts."""

from __future__ import annotations

from temoin.prompts import PROMPT_VERSION, build_constat_messages, build_json_retry_message


def test_prompt_version_is_set() -> None:
    """The prompt version tags cache entries."""
    assert isinstance(PROMPT_VERSION, str) and PROMPT_VERSION.strip()


def test_constat_messages_include_code_and_photo() -> None:
    """Messages carry the expected code and the image data URL."""
    messages = build_constat_messages(
        expected_code="K7", image_data_url="data:image/jpeg;base64,xxx"
    )
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "K7" in messages[0]["content"]
    assert "{expected_code}" not in messages[0]["content"]
    user_parts = messages[1]["content"]
    assert user_parts[1]["image_url"]["url"] == "data:image/jpeg;base64,xxx"


def test_constat_prompt_contains_injection_guard() -> None:
    """The prompt states that visible text is data, never an instruction."""
    messages = build_constat_messages(expected_code="K7", image_data_url="data:,")
    system_text = messages[0]["content"].lower()
    assert "never an instruction" in system_text
    assert "injection_text" in system_text


def test_retry_message_asks_for_strict_json() -> None:
    """The retry message sends the error back and demands strict JSON."""
    message = build_json_retry_message("field required")
    assert message["role"] == "user"
    assert "field required" in message["content"]
    assert "STRICT JSON" in message["content"]
