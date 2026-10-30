"""Investigator agent: tool-using loop over a sale/delivery pair, with trail."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any

from temoin.compare import check_match_references, compare_reports, finalize_comparison
from temoin.config import Settings, load_settings
from temoin.images import prepare_image
from temoin.model import (
    ChatAnswer,
    ChatClient,
    ToolCall,
    call_with_retries,
    default_client,
    read_cache_json,
    validate_content,
    write_cache_json,
)
from temoin.prompts import (
    AGENT_PROMPT_VERSION,
    COMPARE_PROMPT_VERSION,
    build_agent_messages,
    build_conclude_message,
    build_json_retry_message,
)
from temoin.report import create_report
from temoin.schemas import Comparison, ComparisonDraft, Report, TrailEntry
from temoin.tools import AGENT_TOOLS, ToolContext, ToolResult, execute_tool

logger = logging.getLogger(__name__)


def build_investigation_cache_key(*, sale_constat: str, delivery_constat: str) -> str:
    """Derive a cache key from both constats and the agent prompt versions."""
    raw = f"{sale_constat}|{delivery_constat}|{AGENT_PROMPT_VERSION}|{COMPARE_PROMPT_VERSION}"
    return sha256(raw.encode()).hexdigest()


def _shorten(text: str, limit: int = 160) -> str:
    """Truncate a text to one trail line."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _constat_entry(step: int, label: str, report: Report) -> TrailEntry:
    """Summarize one constat as a trail entry."""
    code = report.challenge.read_code or "none"
    return TrailEntry(
        step=step,
        action=f"constat_{label}",
        summary=f"{label} constat: {len(report.defects)} defects, code {code}.",
    )


def _assistant_tool_message(answer: ChatAnswer) -> dict[str, Any]:
    """Render an assistant answer with tool calls as a conversation message."""
    return {
        "role": "assistant",
        "content": answer.content or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in answer.tool_calls
        ],
    }


def _execute_logged(
    call: ToolCall,
    context: ToolContext,
    messages: list[dict[str, Any]],
    trail: list[TrailEntry],
    on_step: Callable[[TrailEntry], None] | None,
) -> ToolResult:
    """Execute one tool call, log it in the trail and append its result."""
    try:
        arguments = json.loads(call.arguments or "{}")
    except ValueError:
        arguments = None
    if not isinstance(arguments, dict):
        result = ToolResult(name=call.name, content="Tool error: arguments must be a JSON object.")
    else:
        result = execute_tool(call.name, arguments, context)
    messages.append({"role": "tool", "tool_call_id": call.id, "content": result.content})
    if result.image_data_url is not None:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": result.image_data_url},
                    }
                ],
            }
        )
    trail.append(
        TrailEntry(
            step=len(trail) + 1,
            action=f"tool:{call.name}",
            summary=_shorten(f"{call.name} {call.arguments} -> {result.content}"),
        )
    )
    if on_step is not None:
        on_step(trail[-1])
    return result


def _parse_draft(content: str, *, sale: Report, delivery: Report) -> ComparisonDraft:
    """Extract and validate a verdict draft, checking match references."""
    draft = validate_content(content, ComparisonDraft)
    check_match_references(draft, sale=sale, delivery=delivery)
    return draft


def _read_investigation_cache(
    settings: Settings, sale: Report, delivery: Report
) -> Comparison | None:
    """Replay a cached investigation, or None on any miss."""
    key = build_investigation_cache_key(
        sale_constat=sale.identity_json(),
        delivery_constat=delivery.identity_json(),
    )
    payload = read_cache_json(settings.cache_dir, key)
    if payload is None or payload.get("agent_version") != AGENT_PROMPT_VERSION:
        return None
    try:
        comparison = Comparison.model_validate_json(payload["comparison"])
    except (ValueError, KeyError, TypeError) as error:
        logger.warning("Ignoring invalid investigation cache: %s", error)
        return None
    return comparison


def _write_investigation_cache(
    settings: Settings, sale: Report, delivery: Report, comparison: Comparison
) -> None:
    """Store a finished investigation for identical reruns."""
    key = build_investigation_cache_key(
        sale_constat=sale.identity_json(),
        delivery_constat=delivery.identity_json(),
    )
    write_cache_json(
        settings.cache_dir,
        key,
        {"agent_version": AGENT_PROMPT_VERSION, "comparison": comparison.model_dump_json()},
    )


def run_investigation(
    sale_photo: str | Path,
    delivery_photo: str | Path,
    *,
    sale_code: str | None = None,
    delivery_code: str | None = None,
    settings: Settings | None = None,
    client: ChatClient | None = None,
    on_step: Callable[[TrailEntry], None] | None = None,
) -> Comparison:
    """Investigate a sale/delivery pair with tools and return the verdict."""
    active_settings = settings or load_settings()
    active_client = client or default_client(
        api_key=active_settings.baseten_api_key,
        base_url=active_settings.model_base_url,
        timeout_seconds=active_settings.model_timeout_seconds,
    )
    sale = create_report(
        sale_photo, sale_code or "", settings=active_settings, client=active_client
    )
    delivery = create_report(
        delivery_photo, delivery_code or "", settings=active_settings, client=active_client
    )
    cached = _read_investigation_cache(active_settings, sale, delivery)
    if cached is not None:
        return cached
    sale_image = prepare_image(
        sale_photo,
        max_pixels=active_settings.image_max_pixels,
        max_megabytes=active_settings.image_max_megabytes,
        jpeg_quality=active_settings.image_jpeg_quality,
    )
    delivery_image = prepare_image(
        delivery_photo,
        max_pixels=active_settings.image_max_pixels,
        max_megabytes=active_settings.image_max_megabytes,
        jpeg_quality=active_settings.image_jpeg_quality,
    )
    messages = build_agent_messages(
        sale=sale,
        delivery=delivery,
        sale_data_url=sale_image.data_url,
        delivery_data_url=delivery_image.data_url,
    )
    context = ToolContext(
        sale=sale,
        delivery=delivery,
        sale_photo=Path(sale_photo),
        delivery_photo=Path(delivery_photo),
        max_pixels=active_settings.image_max_pixels,
        jpeg_quality=active_settings.image_jpeg_quality,
    )
    trail = [_constat_entry(1, "sale", sale), _constat_entry(2, "delivery", delivery)]
    if on_step is not None:
        on_step(trail[0])
        on_step(trail[1])
    must_conclude = False
    photo_requested = False
    zoom_used = False
    draft: ComparisonDraft | None = None
    for _step in range(active_settings.agent_max_steps):
        answer = call_with_retries(
            lambda conclude=must_conclude: active_client.chat_with_tools(
                active_settings.model_name,
                messages,
                None if conclude else AGENT_TOOLS,
            ),
            max_attempts=active_settings.model_max_attempts,
            retry_wait_seconds=active_settings.model_retry_wait_seconds,
        )
        if answer.tool_calls and not must_conclude:
            messages.append(_assistant_tool_message(answer))
            for call in answer.tool_calls:
                _execute_logged(call, context, messages, trail, on_step)
                if call.name == "zoom":
                    zoom_used = True
                if call.name == "request_photo":
                    must_conclude = True
                    photo_requested = True
            if must_conclude:
                messages.append(build_conclude_message(photo_requested=photo_requested))
            elif not zoom_used:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Reminder: call zoom at least once on the most "
                            "doubtful defect before concluding."
                        ),
                    }
                )
            continue
        try:
            draft = _parse_draft(answer.content, sale=sale, delivery=delivery)
        except ValueError as error:
            messages.append({"role": "assistant", "content": answer.content})
            messages.append(build_json_retry_message(str(error)))
            continue
        break
    if draft is None:
        trail.append(
            TrailEntry(
                step=len(trail) + 1,
                action="fallback",
                summary="Agent did not conclude; single-shot verdict.",
            )
        )
        if on_step is not None:
            on_step(trail[-1])
        comparison = compare_reports(
            sale,
            delivery,
            sale_image=sale_image,
            delivery_image=delivery_image,
            settings=active_settings,
            client=active_client,
            trail=trail,
            on_step=on_step,
        )
    else:
        comparison = finalize_comparison(
            draft,
            sale=sale,
            delivery=delivery,
            trail=trail,
            prompt_version=AGENT_PROMPT_VERSION,
            disclaimer=active_settings.human_review_disclaimer,
            on_step=on_step,
        )
    _write_investigation_cache(active_settings, sale, delivery, comparison)
    return comparison
