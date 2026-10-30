"""Tests for the pydantic report models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from temoin.schemas import Comparison, ComparisonDraft, Defect, DefectMatch, Report, ReportDraft


def _valid_defect() -> dict[str, Any]:
    """Return a valid defect payload."""
    return {
        "id": "D1",
        "type": "rayure",
        "description": "Rayure claire.",
        "zone": "devant",
        "severity": "minor",
        "bbox": [0, 0, 1000, 1000],
        "distinctive": True,
    }


def test_defect_accepts_optional_bbox() -> None:
    """bbox may be omitted."""
    payload = _valid_defect()
    payload["bbox"] = None
    assert Defect.model_validate(payload).bbox is None


def test_defect_rejects_bad_severity() -> None:
    """Severity must be minor, moderate or major."""
    payload = _valid_defect()
    payload["severity"] = "enorme"
    with pytest.raises(ValidationError):
        Defect.model_validate(payload)


def test_defect_rejects_out_of_range_bbox() -> None:
    """bbox coordinates must stay within 0-1000."""
    payload = _valid_defect()
    payload["bbox"] = [0, 0, 1001, 500]
    with pytest.raises(ValidationError):
        Defect.model_validate(payload)


def test_draft_rejects_unknown_checkpoint_reference(
    draft_payload: dict[str, Any],
) -> None:
    """Checkpoints must reference an existing defect id."""
    draft_payload["checkpoints"] = [{"defect_id": "D99", "instruction": "x"}]
    with pytest.raises(ValidationError):
        ReportDraft.model_validate(draft_payload)


def test_draft_rejects_duplicate_defect_ids(draft_payload: dict[str, Any]) -> None:
    """Defect ids must be unique."""
    draft_payload["defects"][1]["id"] = "D1"
    with pytest.raises(ValidationError):
        ReportDraft.model_validate(draft_payload)


def test_report_from_draft_attaches_metadata(draft_payload: dict[str, Any]) -> None:
    """from_draft keeps the answer and adds fingerprint, date and version."""
    draft = ReportDraft.model_validate(draft_payload)
    now = datetime.now(UTC)
    report = Report.from_draft(draft, image_sha256="abc123", created_at_utc=now, prompt_version="1")
    assert isinstance(report, Report)
    assert report.image_sha256 == "abc123"
    assert report.created_at_utc == now
    assert report.prompt_version == "1"
    assert [point.defect_id for point in report.checkpoints] == ["D1"]


def test_defect_match_statuses_need_consistent_ids() -> None:
    """matched/changed need both ids, missing/new need exactly one."""
    DefectMatch.model_validate(
        {
            "sale_defect_id": "D1",
            "delivery_defect_id": "D2",
            "status": "matched",
            "note": "x",
        }
    )
    DefectMatch.model_validate(
        {
            "sale_defect_id": "D1",
            "delivery_defect_id": None,
            "status": "missing",
            "note": "x",
        }
    )
    DefectMatch.model_validate(
        {"sale_defect_id": None, "delivery_defect_id": "D2", "status": "new", "note": "x"}
    )
    with pytest.raises(ValidationError):
        DefectMatch.model_validate(
            {
                "sale_defect_id": "D1",
                "delivery_defect_id": None,
                "status": "matched",
                "note": "x",
            }
        )
    with pytest.raises(ValidationError):
        DefectMatch.model_validate(
            {
                "sale_defect_id": "D1",
                "delivery_defect_id": "D2",
                "status": "missing",
                "note": "x",
            }
        )
    with pytest.raises(ValidationError):
        DefectMatch.model_validate(
            {"sale_defect_id": "D1", "delivery_defect_id": None, "status": "new", "note": "x"}
        )


def test_comparison_draft_rejects_bad_confidence(
    comparison_payload: dict[str, Any],
) -> None:
    """Confidence must stay within 0-100."""
    comparison_payload["confidence"] = 101
    with pytest.raises(ValidationError):
        ComparisonDraft.model_validate(comparison_payload)
    comparison_payload["confidence"] = -1
    with pytest.raises(ValidationError):
        ComparisonDraft.model_validate(comparison_payload)


def test_comparison_from_draft_attaches_metadata(
    comparison_payload: dict[str, Any], sale_report: Report, delivery_report: Report
) -> None:
    """from_draft keeps the answer and adds guardrail outcome and metadata."""
    draft = ComparisonDraft.model_validate(comparison_payload)
    now = datetime.now(UTC)
    comparison = Comparison.from_draft(
        draft,
        verdict="ENDOMMAGÉ",
        capped_reason=None,
        sale=sale_report,
        delivery=delivery_report,
        freshness_sale_ok=True,
        freshness_delivery_ok=False,
        created_at_utc=now,
        prompt_version="7",
        disclaimer="D.",
        trail=[],
    )
    assert comparison.verdict == "ENDOMMAGÉ"
    assert comparison.model_verdict == "CONFORME"
    assert comparison.sale_image_sha256 == sale_report.image_sha256
    assert comparison.freshness_delivery_ok is False
    assert comparison.prompt_version == "7"
    assert comparison.trail == []
