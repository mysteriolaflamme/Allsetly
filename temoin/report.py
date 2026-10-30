"""Build the validated constat of a single photo."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from temoin.config import Settings, load_settings
from temoin.images import prepare_image
from temoin.model import ChatClient, build_cache_key, complete_json, default_client
from temoin.prompts import PROMPT_VERSION, build_constat_messages
from temoin.schemas import Report, ReportDraft


def create_report(
    photo: str | Path,
    expected_code: str,
    *,
    settings: Settings | None = None,
    client: ChatClient | None = None,
    declared_materials: list[str] | None = None,
) -> Report:
    """Analyse a photo and return its validated constat as a Report."""
    active_settings = settings or load_settings()
    prepared = prepare_image(
        photo,
        max_pixels=active_settings.image_max_pixels,
        max_megabytes=active_settings.image_max_megabytes,
        jpeg_quality=active_settings.image_jpeg_quality,
    )
    code = expected_code.strip().upper()
    messages = build_constat_messages(
        expected_code=code, image_data_url=prepared.data_url, declared_materials=declared_materials
    )
    active_client = client or default_client(
        api_key=active_settings.baseten_api_key,
        base_url=active_settings.model_base_url,
        timeout_seconds=active_settings.model_timeout_seconds,
    )
    draft = complete_json(
        messages,
        ReportDraft,
        client=active_client,
        model_name=active_settings.model_name,
        cache_key=build_cache_key(image_sha256=prepared.sha256, expected_code=code),
        cache_dir=active_settings.cache_dir,
        max_attempts=active_settings.model_max_attempts,
        retry_wait_seconds=active_settings.model_retry_wait_seconds,
        json_retries=active_settings.model_json_retries,
    )
    draft.challenge.expected_code = code
    return Report.from_draft(
        draft,
        image_sha256=prepared.sha256,
        created_at_utc=datetime.now(UTC),
        prompt_version=PROMPT_VERSION,
        declared_materials=declared_materials,
    )
