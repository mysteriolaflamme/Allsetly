"""Tests for the versioned model prompts."""

from __future__ import annotations

from temoin.prompts import (
    AGENT_PROMPT_VERSION,
    COMPARE_PROMPT_VERSION,
    PROMPT_VERSION,
    build_agent_messages,
    build_compare_messages,
    build_constat_messages,
    build_json_retry_message,
)
from temoin.schemas import Report


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


def test_compare_and_agent_versions_are_set() -> None:
    """Both investigation prompts are versioned for cache tagging."""
    assert isinstance(COMPARE_PROMPT_VERSION, str) and COMPARE_PROMPT_VERSION.strip()
    assert isinstance(AGENT_PROMPT_VERSION, str) and AGENT_PROMPT_VERSION.strip()


def test_compare_messages_include_constats_and_both_photos(
    sale_report: Report, delivery_report: Report
) -> None:
    """Compare messages carry both constats and both photos."""
    messages = build_compare_messages(
        sale=sale_report,
        delivery=delivery_report,
        sale_data_url="data:,sale",
        delivery_data_url="data:,del",
    )
    assert [message["role"] for message in messages] == ["system", "user"]
    text = messages[1]["content"][0]["text"]
    assert sale_report.description in text
    assert delivery_report.description in text
    images = [part["image_url"]["url"] for part in messages[1]["content"][1:]]
    assert images == ["data:,sale", "data:,del"]


def test_compare_prompt_states_rules_and_guard(
    sale_report: Report, delivery_report: Report
) -> None:
    """The compare prompt holds the 4 verdicts, the 2-contradiction rule and the guard."""
    messages = build_compare_messages(
        sale=sale_report,
        delivery=delivery_report,
        sale_data_url="data:,sale",
        delivery_data_url="data:,del",
    )
    system = messages[0]["content"]
    for verdict in ("CONFORME", "ENDOMMAGÉ", "À VÉRIFIER", "NON CONFORME"):
        assert verdict in system
    lowered = system.lower()
    assert "at least 2" in lowered
    assert "never flaws" in lowered
    assert "never an instruction" in lowered
    assert "injection_text" in system


def test_agent_messages_name_tools_and_demand_zoom(
    sale_report: Report, delivery_report: Report
) -> None:
    """The agent prompt lists tools, requires a zoom and caps the steps."""
    messages = build_agent_messages(
        sale=sale_report,
        delivery=delivery_report,
        sale_data_url="data:,sale",
        delivery_data_url="data:,del",
    )
    system = messages[0]["content"]
    for tool in ("match_defects", "zoom", "check_challenge", "request_photo", "segment"):
        assert tool in system
    assert "at most 8 steps" in system
    assert "MUST" in system
    images = [part["image_url"]["url"] for part in messages[1]["content"][1:]]
    assert images == ["data:,sale", "data:,del"]


def test_constat_prompt_lists_materials_task() -> None:
    """The prompt asks for materials and names the mismatch category."""
    messages = build_constat_messages(expected_code="K7", image_data_url="data:,")
    system = messages[0]["content"]
    assert "MATERIALS" in system
    assert '"materials"' in system
    assert "material_mismatch" in system


def test_constat_messages_show_declared_materials() -> None:
    """Declared materials appear in the prompt, 'none' by default."""
    messages = build_constat_messages(
        expected_code="K7", image_data_url="data:,", declared_materials=[" cuir ", "", "bois"]
    )
    assert "DECLARED MATERIALS: cuir, bois" in messages[0]["content"]
    default = build_constat_messages(expected_code="K7", image_data_url="data:,")
    assert "DECLARED MATERIALS: none" in default[0]["content"]
