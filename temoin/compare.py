"""Comparison of a sale photo and a delivery photo: verdict under section 5 rules."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from temoin.config import Settings, load_settings
from temoin.images import PreparedImage
from temoin.model import (
    ChatClient,
    build_compare_cache_key,
    complete_json,
    default_client,
)
from temoin.prompts import COMPARE_PROMPT_VERSION, build_compare_messages
from temoin.schemas import (
    Comparison,
    ComparisonDraft,
    Report,
    TrailEntry,
    Verdict,
)


def is_fresh(report: Report) -> bool:
    """Return True when the challenge code on the photo is verified."""
    read = (report.challenge.read_code or "").strip().upper()
    expected = report.challenge.expected_code.strip().upper()
    return report.challenge.visible and bool(expected) and read == expected


def check_match_references(draft: ComparisonDraft, *, sale: Report, delivery: Report) -> None:
    """Raise ValueError when a match references an unknown defect id."""
    sale_ids = {defect.id for defect in sale.defects}
    delivery_ids = {defect.id for defect in delivery.defects}
    for match in draft.matches:
        if match.sale_defect_id is not None and match.sale_defect_id not in sale_ids:
            raise ValueError(f"Unknown sale defect id: {match.sale_defect_id}.")
        if match.delivery_defect_id is not None and match.delivery_defect_id not in delivery_ids:
            raise ValueError(f"Unknown delivery defect id: {match.delivery_defect_id}.")


def apply_verdict_guardrails(
    draft: ComparisonDraft, *, sale: Report, delivery: Report
) -> tuple[Verdict, str | None]:
    """Enforce the section 5 rules; return the final verdict and cap reason."""
    if not (is_fresh(sale) and is_fresh(delivery)):
        return "À VÉRIFIER", "preuve de fraîcheur manquante"
    if draft.verdict == "CONFORME":
        if not draft.same_object or draft.new_damage:
            return "À VÉRIFIER", "verdict incohérent avec l'analyse"
    elif draft.verdict == "ENDOMMAGÉ":
        if not draft.same_object or not draft.new_damage:
            return "À VÉRIFIER", "verdict incohérent avec l'analyse"
        if not any(match.status in ("new", "changed") for match in draft.matches):
            return "À VÉRIFIER", "dommage non localisé dans les correspondances"
    elif draft.verdict == "NON CONFORME":
        if draft.same_object:
            return "À VÉRIFIER", "verdict incohérent avec l'analyse"
        if len(draft.contradictions) < 2:
            return "À VÉRIFIER", "moins de 2 éléments contradictoires nets"
    return draft.verdict, None


def finalize_comparison(
    draft: ComparisonDraft,
    *,
    sale: Report,
    delivery: Report,
    trail: list[TrailEntry],
    prompt_version: str,
    disclaimer: str,
    on_step: Callable[[TrailEntry], None] | None = None,
) -> Comparison:
    """Apply guardrails and metadata to a validated draft, extending the trail."""
    verdict, capped_reason = apply_verdict_guardrails(draft, sale=sale, delivery=delivery)
    entries = list(trail)
    entries.append(
        TrailEntry(
            step=len(entries) + 1,
            action="verdict",
            summary=f"Model verdict: {draft.verdict} (confidence {draft.confidence}).",
        )
    )
    if capped_reason is not None:
        entries.append(
            TrailEntry(
                step=len(entries) + 1,
                action="guardrail",
                summary=f"Capped to {verdict}: {capped_reason}.",
            )
        )
    if on_step is not None:
        for entry in entries[len(trail) :]:
            on_step(entry)
    return Comparison.from_draft(
        draft,
        verdict=verdict,
        capped_reason=capped_reason,
        sale=sale,
        delivery=delivery,
        freshness_sale_ok=is_fresh(sale),
        freshness_delivery_ok=is_fresh(delivery),
        created_at_utc=datetime.now(UTC),
        prompt_version=prompt_version,
        disclaimer=disclaimer,
        trail=entries,
    )


def compare_reports(
    sale: Report,
    delivery: Report,
    *,
    sale_image: PreparedImage,
    delivery_image: PreparedImage,
    settings: Settings | None = None,
    client: ChatClient | None = None,
    trail: list[TrailEntry] | None = None,
    on_step: Callable[[TrailEntry], None] | None = None,
) -> Comparison:
    """Compare two validated constats in one structured call, then finalize."""
    active_settings = settings or load_settings()
    messages = build_compare_messages(
        sale=sale,
        delivery=delivery,
        sale_data_url=sale_image.data_url,
        delivery_data_url=delivery_image.data_url,
    )
    active_client = client or default_client(
        api_key=active_settings.baseten_api_key,
        base_url=active_settings.model_base_url,
        timeout_seconds=active_settings.model_timeout_seconds,
    )

    def _check_references(checked: ComparisonDraft) -> None:
        """Check match ids against both constats."""
        check_match_references(checked, sale=sale, delivery=delivery)

    draft = complete_json(
        messages,
        ComparisonDraft,
        client=active_client,
        model_name=active_settings.model_name,
        cache_key=build_compare_cache_key(
            sale_constat=sale.identity_json(),
            delivery_constat=delivery.identity_json(),
        ),
        cache_dir=active_settings.cache_dir,
        max_attempts=active_settings.model_max_attempts,
        retry_wait_seconds=active_settings.model_retry_wait_seconds,
        json_retries=active_settings.model_json_retries,
        extra_check=_check_references,
    )
    return finalize_comparison(
        draft,
        sale=sale,
        delivery=delivery,
        trail=list(trail) if trail else [],
        prompt_version=COMPARE_PROMPT_VERSION,
        disclaimer=active_settings.human_review_disclaimer,
        on_step=on_step,
    )
