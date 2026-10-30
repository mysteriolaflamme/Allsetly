"""Tests for verdict guardrails and one-shot comparison (no network)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from temoin.compare import (
    apply_verdict_guardrails,
    check_match_references,
    compare_reports,
    finalize_comparison,
    is_fresh,
)
from temoin.config import Settings
from temoin.images import PreparedImage
from temoin.schemas import Comparison, ComparisonDraft, Report
from tests.fake_client import FakeChatClient


def _draft(payload: dict[str, Any], **changes: Any) -> ComparisonDraft:
    """Build a draft from the valid payload with changes applied."""
    return ComparisonDraft.model_validate({**payload, **changes})


def test_is_fresh_requires_visible_matching_code(sale_report: Report) -> None:
    """Freshness needs a visible paper with the exact expected code."""
    assert is_fresh(sale_report) is True
    hidden = sale_report.model_copy(deep=True)
    hidden.challenge.visible = False
    assert is_fresh(hidden) is False
    mismatch = sale_report.model_copy(deep=True)
    mismatch.challenge.read_code = "XX"
    assert is_fresh(mismatch) is False
    unknown = sale_report.model_copy(deep=True)
    unknown.challenge.expected_code = ""
    assert is_fresh(unknown) is False


def test_guardrails_cap_without_freshness(
    comparison_payload: dict[str, Any], sale_report: Report, delivery_report: Report
) -> None:
    """A missing code caps any verdict to À VÉRIFIER."""
    draft = _draft(comparison_payload)
    stale = sale_report.model_copy(deep=True)
    stale.challenge.visible = False
    verdict, reason = apply_verdict_guardrails(draft, sale=stale, delivery=delivery_report)
    assert verdict == "À VÉRIFIER"
    assert reason == "preuve de fraîcheur manquante"


def test_guardrails_keep_conforme_when_fresh(
    comparison_payload: dict[str, Any], sale_report: Report, delivery_report: Report
) -> None:
    """A consistent verdict passes through untouched when codes are verified."""
    draft = _draft(comparison_payload)
    result = apply_verdict_guardrails(draft, sale=sale_report, delivery=delivery_report)
    assert result == ("CONFORME", None)


def test_guardrails_require_two_contradictions(
    comparison_payload: dict[str, Any], sale_report: Report, delivery_report: Report
) -> None:
    """NON CONFORME needs at least 2 clear contradictions."""
    one = _draft(
        comparison_payload,
        verdict="NON CONFORME",
        same_object=False,
        contradictions=["Different color."],
    )
    verdict, reason = apply_verdict_guardrails(one, sale=sale_report, delivery=delivery_report)
    assert verdict == "À VÉRIFIER"
    assert reason == "moins de 2 éléments contradictoires nets"
    two = _draft(
        comparison_payload,
        verdict="NON CONFORME",
        same_object=False,
        contradictions=["Different color.", "Distinctive scratch absent."],
    )
    result = apply_verdict_guardrails(two, sale=sale_report, delivery=delivery_report)
    assert result == ("NON CONFORME", None)


def test_guardrails_reject_inconsistent_verdicts(
    comparison_payload: dict[str, Any], sale_report: Report, delivery_report: Report
) -> None:
    """A verdict contradicting its own analysis falls back to À VÉRIFIER."""
    bad_conforme = _draft(comparison_payload, same_object=False)
    result = apply_verdict_guardrails(bad_conforme, sale=sale_report, delivery=delivery_report)
    assert result == ("À VÉRIFIER", "verdict incohérent avec l'analyse")
    bad_damaged = _draft(
        comparison_payload, verdict="ENDOMMAGÉ", same_object=True, new_damage=False
    )
    result = apply_verdict_guardrails(bad_damaged, sale=sale_report, delivery=delivery_report)
    assert result == ("À VÉRIFIER", "verdict incohérent avec l'analyse")
    bad_accusation = _draft(
        comparison_payload,
        verdict="NON CONFORME",
        same_object=True,
        contradictions=["a", "b"],
    )
    result = apply_verdict_guardrails(bad_accusation, sale=sale_report, delivery=delivery_report)
    assert result == ("À VÉRIFIER", "verdict incohérent avec l'analyse")


def test_guardrails_require_located_damage(
    comparison_payload: dict[str, Any], sale_report: Report, delivery_report: Report
) -> None:
    """ENDOMMAGÉ must point to at least one new or changed defect."""
    unlocated = _draft(comparison_payload, verdict="ENDOMMAGÉ", same_object=True, new_damage=True)
    verdict, reason = apply_verdict_guardrails(
        unlocated, sale=sale_report, delivery=delivery_report
    )
    assert verdict == "À VÉRIFIER"
    assert reason == "dommage non localisé dans les correspondances"
    located = ComparisonDraft.model_validate(
        {
            **comparison_payload,
            "verdict": "ENDOMMAGÉ",
            "same_object": True,
            "new_damage": True,
            "matches": [
                {
                    "sale_defect_id": None,
                    "delivery_defect_id": "D1",
                    "status": "new",
                    "note": "New crack.",
                }
            ],
        }
    )
    result = apply_verdict_guardrails(located, sale=sale_report, delivery=delivery_report)
    assert result == ("ENDOMMAGÉ", None)


def test_check_match_references_accepts_valid_ids(
    comparison_payload: dict[str, Any], sale_report: Report, delivery_report: Report
) -> None:
    """Matches on known defect ids pass silently."""
    draft = _draft(comparison_payload)
    check_match_references(draft, sale=sale_report, delivery=delivery_report)


def test_check_match_references_rejects_unknown_ids(
    comparison_payload: dict[str, Any], sale_report: Report, delivery_report: Report
) -> None:
    """Matches on unknown defect ids raise a clear error."""
    bad_sale = ComparisonDraft.model_validate(
        {
            **comparison_payload,
            "matches": [
                {
                    "sale_defect_id": "D99",
                    "delivery_defect_id": "D1",
                    "status": "matched",
                    "note": "x",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="Unknown sale defect id"):
        check_match_references(bad_sale, sale=sale_report, delivery=delivery_report)
    bad_delivery = ComparisonDraft.model_validate(
        {
            **comparison_payload,
            "matches": [
                {
                    "sale_defect_id": "D1",
                    "delivery_defect_id": "D99",
                    "status": "matched",
                    "note": "x",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="Unknown delivery defect id"):
        check_match_references(bad_delivery, sale=sale_report, delivery=delivery_report)


def test_finalize_comparison_extends_trail(
    comparison_payload: dict[str, Any], sale_report: Report, delivery_report: Report
) -> None:
    """finalize attaches metadata and logs the verdict in the trail."""
    draft = _draft(comparison_payload)
    comparison = finalize_comparison(
        draft,
        sale=sale_report,
        delivery=delivery_report,
        trail=[],
        prompt_version="9",
        disclaimer="D.",
    )
    assert isinstance(comparison, Comparison)
    assert comparison.verdict == "CONFORME"
    assert comparison.model_verdict == "CONFORME"
    assert comparison.capped_reason is None
    assert comparison.freshness_sale_ok and comparison.freshness_delivery_ok
    assert comparison.sale_image_sha256 == sale_report.image_sha256
    assert comparison.prompt_version == "9"
    assert comparison.disclaimer == "D."
    assert [entry.action for entry in comparison.trail] == ["verdict"]


def test_finalize_comparison_logs_guardrail(
    comparison_payload: dict[str, Any], sale_report: Report, delivery_report: Report
) -> None:
    """A capped verdict keeps the model verdict and the reason in the trail."""
    draft = _draft(comparison_payload)
    stale = delivery_report.model_copy(deep=True)
    stale.challenge.visible = False
    comparison = finalize_comparison(
        draft,
        sale=sale_report,
        delivery=stale,
        trail=[],
        prompt_version="9",
        disclaimer="D.",
    )
    assert comparison.verdict == "À VÉRIFIER"
    assert comparison.model_verdict == "CONFORME"
    assert comparison.capped_reason == "preuve de fraîcheur manquante"
    assert [entry.action for entry in comparison.trail] == ["verdict", "guardrail"]


def test_compare_reports_returns_comparison(
    test_settings: Settings,
    sale_report: Report,
    delivery_report: Report,
    sale_image: PreparedImage,
    delivery_image: PreparedImage,
    comparison_json: str,
) -> None:
    """One structured call produces a validated comparison."""
    client = FakeChatClient([comparison_json])
    comparison = compare_reports(
        sale_report,
        delivery_report,
        sale_image=sale_image,
        delivery_image=delivery_image,
        settings=test_settings,
        client=client,
    )
    assert comparison.verdict == "CONFORME"
    assert comparison.prompt_version == "1"
    assert comparison.disclaimer == test_settings.human_review_disclaimer
    assert len(client.calls) == 1


def test_compare_reports_uses_cache(
    test_settings: Settings,
    sale_report: Report,
    delivery_report: Report,
    sale_image: PreparedImage,
    delivery_image: PreparedImage,
    comparison_json: str,
) -> None:
    """A second identical comparison replays from disk without a new call."""
    first = FakeChatClient([comparison_json])
    compare_reports(
        sale_report,
        delivery_report,
        sale_image=sale_image,
        delivery_image=delivery_image,
        settings=test_settings,
        client=first,
    )
    assert len(first.calls) == 1
    second = FakeChatClient([])
    out = compare_reports(
        sale_report,
        delivery_report,
        sale_image=sale_image,
        delivery_image=delivery_image,
        settings=test_settings,
        client=second,
    )
    assert out.verdict == "CONFORME"
    assert second.calls == []


def test_compare_reports_retries_dangling_reference(
    test_settings: Settings,
    comparison_payload: dict[str, Any],
    comparison_json: str,
    sale_report: Report,
    delivery_report: Report,
    sale_image: PreparedImage,
    delivery_image: PreparedImage,
) -> None:
    """A match on an unknown id is sent back for one new attempt."""
    bad = json.dumps(
        {
            **comparison_payload,
            "matches": [
                {
                    "sale_defect_id": "D99",
                    "delivery_defect_id": "D1",
                    "status": "matched",
                    "note": "x",
                }
            ],
        }
    )
    client = FakeChatClient([bad, comparison_json])
    out = compare_reports(
        sale_report,
        delivery_report,
        sale_image=sale_image,
        delivery_image=delivery_image,
        settings=test_settings,
        client=client,
    )
    assert out.verdict == "CONFORME"
    assert len(client.calls) == 2


def test_compare_reports_cache_ignores_timestamps(
    test_settings: Settings,
    sale_report: Report,
    delivery_report: Report,
    sale_image: PreparedImage,
    delivery_image: PreparedImage,
    comparison_json: str,
) -> None:
    """Rebuilt constats with new timestamps still hit the comparison cache."""
    first = FakeChatClient([comparison_json])
    compare_reports(
        sale_report,
        delivery_report,
        sale_image=sale_image,
        delivery_image=delivery_image,
        settings=test_settings,
        client=first,
    )
    later = datetime(2027, 1, 1, tzinfo=UTC)
    second = FakeChatClient([])
    out = compare_reports(
        sale_report.model_copy(update={"created_at_utc": later}),
        delivery_report.model_copy(update={"created_at_utc": later}),
        sale_image=sale_image,
        delivery_image=delivery_image,
        settings=test_settings,
        client=second,
    )
    assert out.verdict == "CONFORME"
    assert second.calls == []
