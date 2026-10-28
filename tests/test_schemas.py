"""Tests for the pydantic report models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from temoin.schemas import Defect, Report, ReportDraft


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
