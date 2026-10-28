"""Pydantic models for constats: defects, challenge check, alerts, checkpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

Severity = Literal["minor", "moderate", "major"]
BBoxCoord = Annotated[int, Field(ge=0, le=1000)]


class Defect(BaseModel):
    """One visible flaw that makes this item unique."""

    id: str = Field(description="Stable identifier, e.g. D1.")
    type: str = Field(description="Flaw kind: scratch, stain, wear, crack, dent...")
    description: str = Field(description="Factual description of the flaw.")
    zone: str = Field(description="Readable position of the flaw on the item.")
    severity: Severity
    bbox: list[BBoxCoord] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Optional [x1, y1, x2, y2] box, coordinates normalized 0-1000.",
    )
    distinctive: bool = Field(description="True when this flaw identifies this item.")


class ChallengeCheck(BaseModel):
    """Observation of the anti-reuse challenge code on the photo."""

    expected_code: str = Field(description="Challenge code the seller had to write down.")
    visible: bool = Field(description="True only when the paper with the code is visible.")
    read_code: str | None = Field(
        default=None, description="Code actually read on the paper, if any."
    )
    looks_printed_or_edited: bool = Field(
        default=False, description="True when the code looks printed or pasted."
    )


class Alert(BaseModel):
    """Something suspicious or worth signalling about the photo."""

    category: str = Field(
        description="Short category: injection_text, screen_photo, blurry, "
        "code_tampering, face_visible, other."
    )
    message: str = Field(description="Plain-language explanation of the alert.")


class Checkpoint(BaseModel):
    """One zone the buyer must photograph again at delivery time."""

    defect_id: str = Field(description="Identifier of the defect to re-photograph.")
    instruction: str = Field(description="Plain-language photo instruction for the buyer.")


class ReportDraft(BaseModel):
    """Raw structured answer of the model, before metadata is attached."""

    description: str = Field(description="Factual description of the item.")
    defects: list[Defect] = Field(default_factory=list)
    challenge: ChallengeCheck
    checkpoints: list[Checkpoint] = Field(
        default_factory=list, description="Up to 3 most distinctive defects."
    )
    alerts: list[Alert] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_references(self) -> ReportDraft:
        """Ensure checkpoints reference known defects and ids are unique."""
        ids = [defect.id for defect in self.defects]
        if len(set(ids)) != len(ids):
            raise ValueError("Defect ids must be unique.")
        unknown = [point.defect_id for point in self.checkpoints if point.defect_id not in ids]
        if unknown:
            raise ValueError(f"Unknown defect ids in checkpoints: {', '.join(unknown)}.")
        return self


class Report(BaseModel):
    """Validated constat of one photo, with its metadata."""

    description: str
    defects: list[Defect] = Field(default_factory=list)
    challenge: ChallengeCheck
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
    image_sha256: str
    created_at_utc: datetime
    prompt_version: str

    @classmethod
    def from_draft(
        cls,
        draft: ReportDraft,
        *,
        image_sha256: str,
        created_at_utc: datetime,
        prompt_version: str,
    ) -> Report:
        """Attach metadata to a validated model answer."""
        return cls(
            description=draft.description,
            defects=draft.defects,
            challenge=draft.challenge,
            checkpoints=draft.checkpoints,
            alerts=draft.alerts,
            image_sha256=image_sha256,
            created_at_utc=created_at_utc,
            prompt_version=prompt_version,
        )
