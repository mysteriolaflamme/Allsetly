"""Témoin business API: let marketplaces and couriers embed the neutral witness.

Mounted at ``/api`` beside the Streamlit app (see ``asgi_app.py``), so one
deployment serves people and platforms. Interactive docs live at ``/api/docs``.

Flow for a partner platform:
1. ``POST /v1/cases`` — open a case, show the seller the sale challenge code.
2. ``POST /v1/cases/{id}/sale`` — upload the seller photo (code visible).
3. ``POST /v1/cases/{id}/delivery-code`` — show the buyer their own code.
4. ``POST /v1/cases/{id}/delivery`` — upload the buyer photo; the investigator
   agent compares the flaws and returns a sealed verdict.
5. ``POST /v1/verify`` — anyone can check a certificate was not altered.

Every call except health needs an ``X-API-Key`` header matching one of the
keys in ``TEMOIN_API_KEYS``; with no key configured the API stays closed.
"""

from __future__ import annotations

import hmac
from datetime import datetime, timedelta
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from temoin.compare import is_fresh
from temoin.config import ConfigError, Settings, load_settings
from temoin.images import ImageError
from temoin.model import ModelError
from temoin.schemas import Certificate, IssuedCode, Report, Transaction
from temoin.seal import verify_certificate
from temoin.service import investigate_delivery, record_sale
from temoin.storage import (
    StorageError,
    create_transaction,
    issue_delivery_code,
    load_transaction,
    save_transaction,
)

api = FastAPI(
    title="Témoin API",
    version="1.0.0",
    summary="Neutral visual witness for remote second-hand trade.",
    description=(
        "Prove that the delivered (or returned) item is the one that was sold. "
        "Témoin recognizes a used item by its visible flaws, checks one-time challenge "
        "codes against photo reuse, and seals every verdict in a verifiable chain. "
        "Reasoning runs on Meta Muse Spark Open. Témoin assists a human decision; "
        "it does not replace it."
    ),
    # Hosting proxies (e.g. Streamlit Cloud's /~/+ prefix) hide the real mount path,
    # so docs and "Try it out" use URLs relative to /api instead of absolute ones.
    docs_url=None,
    redoc_url=None,
    servers=[{"url": "."}],
    root_path_in_servers=False,
)


@api.get("/docs", include_in_schema=False)
def docs() -> HTMLResponse:
    """Interactive docs that work behind any path prefix."""
    return get_swagger_ui_html(openapi_url="openapi.json", title="Témoin API")


_PHOTO_TYPES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}


class CodeIssued(BaseModel):
    """A one-time challenge code to write on paper next to the item."""

    case_id: str
    code: str
    expires_at_utc: datetime
    instructions: str


class SaleResult(BaseModel):
    """The sealed sale constat: flaws, checkpoints and freshness."""

    case_id: str
    fresh: bool = Field(description="True when the sale challenge code was read on the photo.")
    constat: Report


class DeliveryResult(BaseModel):
    """The investigator agent's verdict, sealed in a certificate."""

    case_id: str
    verdict: str = Field(description="CONFORME, ENDOMMAGÉ, À VÉRIFIER or NON CONFORME.")
    confidence: int
    justification: str
    certificate: Certificate


class CaseStatus(BaseModel):
    """Where a case stands."""

    case_id: str
    created_at_utc: datetime
    sale_code: str
    delivery_code: str | None
    has_sale_constat: bool
    has_delivery_constat: bool
    verdicts: list[str]


class VerifyResult(BaseModel):
    """Integrity check of one certificate."""

    intact: bool
    verdict: str
    confidence: int
    chained: bool = Field(description="True when this certificate extends an earlier one.")


def get_settings() -> Settings:
    """Load runtime settings (overridden in tests)."""
    try:
        return load_settings()
    except ConfigError as error:
        raise HTTPException(503, str(error)) from error


def require_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> Settings:
    """Reject calls without a valid partner key; keep the API closed without keys."""
    if not settings.api_keys:
        raise HTTPException(503, "API fermée : aucune clé TEMOIN_API_KEYS n'est configurée.")
    if not x_api_key or not any(
        hmac.compare_digest(x_api_key.encode(), key.encode()) for key in settings.api_keys
    ):
        raise HTTPException(401, "Clé d'API absente ou invalide (en-tête X-API-Key).")
    return settings


Authorized = Annotated[Settings, Depends(require_key)]


def _load(case_id: str, settings: Settings) -> Transaction:
    try:
        return load_transaction(case_id, settings)
    except StorageError as error:
        raise HTTPException(404, str(error)) from error


async def _read_photo(photo: UploadFile, settings: Settings) -> bytes:
    if photo.content_type not in _PHOTO_TYPES:
        raise HTTPException(415, "Photo attendue : JPEG, PNG, WebP ou HEIC.")
    data = await photo.read()
    if len(data) > settings.image_max_megabytes * 1024 * 1024:
        raise HTTPException(413, f"Photo trop lourde (maximum {settings.image_max_megabytes} Mo).")
    return data


def _issued(case_id: str, issued: IssuedCode, settings: Settings, who: str) -> CodeIssued:
    return CodeIssued(
        case_id=case_id,
        code=issued.code,
        expires_at_utc=issued.issued_at_utc
        + timedelta(minutes=settings.challenge_validity_minutes),
        instructions=(
            f"Le {who} écrit {issued.code} sur un papier posé à côté de l'objet, "
            "visible sur la photo."
        ),
    )


def _model_step_failed(error: Exception) -> HTTPException:
    if isinstance(error, StorageError):
        return HTTPException(409, str(error))
    if isinstance(error, ImageError):
        return HTTPException(422, str(error))
    if isinstance(error, ConfigError):
        return HTTPException(503, str(error))
    return HTTPException(502, str(error))


@api.get("/v1/health", tags=["service"])
def health() -> dict[str, str]:
    """Liveness probe (no key needed)."""
    return {"status": "ok"}


@api.post("/v1/cases", tags=["cases"], status_code=201)
def open_case(settings: Authorized) -> CodeIssued:
    """Open a case and issue the seller's challenge code."""
    tx = create_transaction(settings)
    save_transaction(tx, settings)
    return _issued(tx.id, tx.sale_code, settings, "vendeur")


@api.post("/v1/cases/{case_id}/sale", tags=["cases"])
async def submit_sale_photo(
    case_id: str,
    settings: Authorized,
    photo: Annotated[UploadFile, File(description="Seller photo with the code visible.")],
    declared_materials: Annotated[str, Form(description="Comma-separated, optional.")] = "",
) -> SaleResult:
    """Build the sale constat: flaw map, checkpoints, freshness."""
    tx = _load(case_id, settings)
    data = await _read_photo(photo, settings)
    declared = [item.strip() for item in declared_materials.split(",") if item.strip()]
    try:
        tx, report = record_sale(tx, data, settings, declared_materials=declared)
    except (StorageError, ImageError, ConfigError, ModelError) as error:
        raise _model_step_failed(error) from error
    return SaleResult(case_id=tx.id, fresh=is_fresh(report), constat=report)


@api.post("/v1/cases/{case_id}/delivery-code", tags=["cases"], status_code=201)
def issue_buyer_code(case_id: str, settings: Authorized) -> CodeIssued:
    """Issue the buyer's challenge code (different from the seller's)."""
    tx = _load(case_id, settings)
    if tx.sale_constat is None:
        raise HTTPException(409, "Le constat de vente manque : envoyez d'abord la photo vendeur.")
    try:
        tx = issue_delivery_code(tx, settings)
    except StorageError as error:
        raise HTTPException(409, str(error)) from error
    save_transaction(tx, settings)
    return _issued(tx.id, tx.delivery_code, settings, "acheteur")


@api.post("/v1/cases/{case_id}/delivery", tags=["cases"])
async def submit_delivery_photo(
    case_id: str,
    settings: Authorized,
    photo: Annotated[UploadFile, File(description="Buyer photo with their code visible.")],
) -> DeliveryResult:
    """Run the investigator agent and return the sealed verdict (can take ~1 min)."""
    tx = _load(case_id, settings)
    data = await _read_photo(photo, settings)
    try:
        tx, certificate = investigate_delivery(tx, data, settings)
    except (StorageError, ImageError, ConfigError, ModelError) as error:
        raise _model_step_failed(error) from error
    comparison = certificate.comparison
    return DeliveryResult(
        case_id=tx.id,
        verdict=comparison.verdict,
        confidence=comparison.confidence,
        justification=comparison.justification,
        certificate=certificate,
    )


@api.get("/v1/cases/{case_id}", tags=["cases"])
def case_status(case_id: str, settings: Authorized) -> CaseStatus:
    """Summarize a case: codes issued, constats present, sealed verdicts."""
    tx = _load(case_id, settings)
    return CaseStatus(
        case_id=tx.id,
        created_at_utc=tx.created_at_utc,
        sale_code=tx.sale_code.code,
        delivery_code=tx.delivery_code.code if tx.delivery_code else None,
        has_sale_constat=tx.sale_constat is not None,
        has_delivery_constat=tx.delivery_constat is not None,
        verdicts=[certificate.comparison.verdict for certificate in tx.certificates],
    )


@api.post("/v1/verify", tags=["certificates"])
def verify(certificate: Certificate, settings: Authorized) -> VerifyResult:
    """Check that a certificate's content still matches its seal."""
    return VerifyResult(
        intact=verify_certificate(certificate),
        verdict=certificate.comparison.verdict,
        confidence=certificate.comparison.confidence,
        chained=certificate.previous_hash is not None,
    )
