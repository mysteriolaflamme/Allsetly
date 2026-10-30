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

    def identity_json(self) -> str:
        """Serialize without volatile metadata, for stable cache keys."""
        return self.model_dump_json(exclude={"created_at_utc"})


Verdict = Literal["CONFORME", "ENDOMMAGÉ", "À VÉRIFIER", "NON CONFORME"]
MatchStatus = Literal["matched", "missing", "new", "changed"]


class DefectMatch(BaseModel):
    """How one flaw lines up between the sale photo and the delivery photo."""

    sale_defect_id: str | None = Field(default=None, description="Sale defect id, if any.")
    delivery_defect_id: str | None = Field(default=None, description="Delivery defect id, if any.")
    status: MatchStatus
    note: str = Field(description="Short explanation of this correspondence.")

    @model_validator(mode="after")
    def _check_ids(self) -> DefectMatch:
        """Ensure the status matches the provided identifiers."""
        sale, delivery = self.sale_defect_id, self.delivery_defect_id
        if self.status in ("matched", "changed") and not (sale and delivery):
            raise ValueError(f"Status '{self.status}' needs both defect ids.")
        if self.status == "missing" and not (sale and not delivery):
            raise ValueError("Status 'missing' needs only a sale defect id.")
        if self.status == "new" and not (delivery and not sale):
            raise ValueError("Status 'new' needs only a delivery defect id.")
        return self


class ComparisonDraft(BaseModel):
    """Raw structured comparison answer, before the deterministic guardrails."""

    verdict: Verdict
    confidence: Annotated[int, Field(ge=0, le=100, description="Confidence score 0-100.")]
    same_object: bool = Field(description="True when both photos show the same item.")
    new_damage: bool = Field(description="True when new damage appeared at delivery.")
    justification: str = Field(description="Short justification of the verdict.")
    matches: list[DefectMatch] = Field(default_factory=list)
    contradictions: list[str] = Field(
        default_factory=list, description="Clear elements proving different objects."
    )
    missing_proof: list[str] = Field(
        default_factory=list, description="Extra photos needed to decide."
    )
    alerts: list[Alert] = Field(default_factory=list)


class TrailEntry(BaseModel):
    """One logged step of the investigation trail."""

    step: int = Field(description="1-based step number in the investigation.")
    action: str = Field(description="Short action name, e.g. tool:zoom or verdict.")
    summary: str = Field(description="One-line human-readable summary of the step.")


class Comparison(BaseModel):
    """Validated comparison of a sale photo and a delivery photo."""

    verdict: Verdict
    model_verdict: Verdict = Field(description="Raw model verdict before guardrails.")
    capped_reason: str | None = Field(
        default=None, description="Guardrail reason when the verdict was capped."
    )
    confidence: Annotated[int, Field(ge=0, le=100)]
    same_object: bool
    new_damage: bool
    justification: str
    matches: list[DefectMatch] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    missing_proof: list[str] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
    freshness_sale_ok: bool
    freshness_delivery_ok: bool
    sale_image_sha256: str
    delivery_image_sha256: str
    created_at_utc: datetime
    prompt_version: str
    disclaimer: str
    trail: list[TrailEntry] = Field(default_factory=list)

    @classmethod
    def from_draft(
        cls,
        draft: ComparisonDraft,
        *,
        verdict: Verdict,
        capped_reason: str | None,
        sale: Report,
        delivery: Report,
        freshness_sale_ok: bool,
        freshness_delivery_ok: bool,
        created_at_utc: datetime,
        prompt_version: str,
        disclaimer: str,
        trail: list[TrailEntry],
    ) -> Comparison:
        """Attach guardrail outcome and metadata to a validated model answer."""
        return cls(
            verdict=verdict,
            model_verdict=draft.verdict,
            capped_reason=capped_reason,
            confidence=draft.confidence,
            same_object=draft.same_object,
            new_damage=draft.new_damage,
            justification=draft.justification,
            matches=draft.matches,
            contradictions=draft.contradictions,
            missing_proof=draft.missing_proof,
            alerts=draft.alerts,
            freshness_sale_ok=freshness_sale_ok,
            freshness_delivery_ok=freshness_delivery_ok,
            sale_image_sha256=sale.image_sha256,
            delivery_image_sha256=delivery.image_sha256,
            created_at_utc=created_at_utc,
            prompt_version=prompt_version,
            disclaimer=disclaimer,
            trail=list(trail),
        )


class IssuedCode(BaseModel):
    """A challenge code issued for one step of a transaction."""

    code: str = Field(description="The issued code, e.g. K7.")
    issued_at_utc: datetime = Field(description="UTC issue time of the code.")


class Certificate(BaseModel):
    """A sealed comparison: verdict, evidence, chain link and seal."""

    certificate_id: str = Field(description="Certificate identifier.")
    comparison: Comparison = Field(description="Sealed verdict with its trail.")
    sale_constat: Report = Field(description="Sealed sale constat.")
    delivery_constat: Report = Field(description="Sealed delivery constat.")
    previous_hash: str | None = Field(
        default=None, description="Seal of the previous certificate, if any."
    )
    sealed_at_utc: datetime
    content_sha256: str = Field(description="SHA-256 seal over the rest.")


class Transaction(BaseModel):
    """One sale case: codes, constats and sealed certificates."""

    id: str = Field(description="Transaction identifier, e.g. tx-1a2b3c4d.")
    created_at_utc: datetime
    sale_code: IssuedCode
    delivery_code: IssuedCode | None = None
    sale_constat: Report | None = None
    delivery_constat: Report | None = None
    certificates: list[Certificate] = Field(default_factory=list)


class FusedDefect(BaseModel):
    """One flaw confirmed across video frames, with its confirmation count."""

    id: str = Field(description="Stable identifier, e.g. F1.")
    type: str = Field(description="Flaw kind: scratch, stain, wear, crack, dent...")
    description: str = Field(description="Factual description of the flaw.")
    zone: str = Field(description="Readable position of the flaw on the item.")
    severity: Severity
    distinctive: bool = Field(description="True when this flaw identifies this item.")
    confirmations: int = Field(description="Frames where this flaw was matched.")
    frames: list[int] = Field(
        default_factory=list, description="0-based indices of matching frames."
    )
    bbox: list[BBoxCoord] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Box from the best occurrence, 0-1000.",
    )


class VideoChallenge(BaseModel):
    """Challenge-code observations across video frames."""

    expected_code: str = Field(description="Challenge code the seller had to show.")
    seen_frames: int = Field(description="Frames where the expected code was read.")
    total_frames: int = Field(description="Frames analyzed in the video.")
    read_code: str | None = Field(
        default=None, description="Code read on the clearest frame, if any."
    )
    verified: bool = Field(description="True when the code was read on >=1 frame.")


class FrameReport(BaseModel):
    """Constat of one video frame."""

    frame_index: int = Field(description="0-based index in the sampled frames.")
    timestamp_s: float = Field(description="Frame timestamp in seconds.")
    report: Report


class VideoReport(BaseModel):
    """Fused multi-view constat of a video."""

    description: str
    fused_defects: list[FusedDefect] = Field(default_factory=list)
    frames: list[FrameReport] = Field(default_factory=list)
    challenge: VideoChallenge
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    alerts: list[Alert] = Field(default_factory=list)
    video_sha256: str
    created_at_utc: datetime
    prompt_version: str
