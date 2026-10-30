"""Devil's advocate: force the model to attack its own verdict before sealing."""

from __future__ import annotations

import logging
from collections.abc import Callable
from hashlib import sha256

from temoin.config import Settings, load_settings
from temoin.model import ChatClient, ModelError, complete_json, default_client
from temoin.prompts import REDTEAM_PROMPT_VERSION, build_redteam_messages
from temoin.schemas import Comparison, DevilChallenge, Report, TrailEntry

logger = logging.getLogger(__name__)


def build_redteam_cache_key(comparison: Comparison) -> str:
    """Derive a cache key from the verdict and the redteam prompt version."""
    raw = f"{comparison.model_dump_json(exclude={'created_at_utc'})}|{REDTEAM_PROMPT_VERSION}"
    return sha256(raw.encode()).hexdigest()


def _downgrade_reason(challenge: DevilChallenge) -> str | None:
    """Return the French cap reason, or None when the verdict survives."""
    majors = sum(1 for item in challenge.objections if item.severity == "major")
    minors = sum(1 for item in challenge.objections if item.severity == "minor")
    if majors < 1 and minors < 2:
        return None
    bits = []
    if majors:
        bits.append(f"{majors} majeure" + ("s" if majors > 1 else ""))
    if minors:
        bits.append(f"{minors} mineure" + ("s" if minors > 1 else ""))
    return "avocat du diable : " + ", ".join(bits)


def challenge_comparison(
    comparison: Comparison,
    *,
    sale: Report,
    delivery: Report,
    settings: Settings | None = None,
    client: ChatClient | None = None,
    on_step: Callable[[TrailEntry], None] | None = None,
) -> Comparison:
    """Attack a verdict once; downgrade it to À VÉRIFIER on strong objections.

    The model proposes objections, the rule disposes: 1 major or 2 minor
    objections cap the verdict. A failing challenge never voids the verdict:
    the skip is logged in the trail and the original comparison is returned.
    """
    active_settings = settings or load_settings()
    active_client = client or default_client(
        api_key=active_settings.baseten_api_key,
        base_url=active_settings.model_base_url,
        timeout_seconds=active_settings.model_timeout_seconds,
    )
    entries = list(comparison.trail)
    try:
        challenge = complete_json(
            build_redteam_messages(sale=sale, delivery=delivery, comparison=comparison),
            DevilChallenge,
            client=active_client,
            model_name=active_settings.model_name,
            cache_key=build_redteam_cache_key(comparison),
            cache_dir=active_settings.cache_dir,
            max_attempts=active_settings.model_max_attempts,
            retry_wait_seconds=active_settings.model_retry_wait_seconds,
            json_retries=active_settings.model_json_retries,
        )
    except ModelError as error:
        skipped = TrailEntry(
            step=len(entries) + 1,
            action="redteam",
            summary=f"Devil's advocate skipped ({error})",
        )
        entries.append(skipped)
        if on_step is not None:
            on_step(skipped)
        return comparison.model_copy(update={"trail": entries})
    total = len(challenge.objections)
    majors = sum(1 for item in challenge.objections if item.severity == "major")
    stance = "stands" if challenge.stand_by_verdict else "challenged"
    entries.append(
        TrailEntry(
            step=len(entries) + 1,
            action="redteam",
            summary=f"Devil's advocate: {total} objection(s) ({majors} major), verdict {stance}.",
        )
    )
    reason = _downgrade_reason(challenge)
    if reason is not None and comparison.verdict != "À VÉRIFIER":
        entries.append(
            TrailEntry(
                step=len(entries) + 1,
                action="guardrail",
                summary=f"Capped to À VÉRIFIER: {reason}.",
            )
        )
        updated = comparison.model_copy(
            update={"verdict": "À VÉRIFIER", "capped_reason": reason, "trail": entries}
        )
    else:
        updated = comparison.model_copy(update={"trail": entries})
    if on_step is not None:
        for entry in entries[len(comparison.trail) :]:
            on_step(entry)
    logger.info("Devil's advocate: %d objection(s), verdict %s.", total, updated.verdict)
    return updated
