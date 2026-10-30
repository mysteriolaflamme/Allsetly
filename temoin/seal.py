"""Sealing of comparisons: hash chain, JSON certificates and QR codes."""

from __future__ import annotations

import hmac
import io
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import qrcode

from temoin.schemas import Certificate, Comparison, Report


class CertificateError(Exception):
    """Raised when a certificate file cannot be used, with a plain message."""


def _canonical_json(certificate: Certificate) -> str:
    """Serialize a certificate without its seal for hashing."""
    return certificate.model_dump_json(exclude={"content_sha256"})


def seal_comparison(
    comparison: Comparison,
    *,
    certificate_id: str,
    previous_hash: str | None,
    sale: Report,
    delivery: Report,
) -> Certificate:
    """Seal a comparison with its evidence, chaining to the previous seal."""
    draft = Certificate(
        certificate_id=certificate_id,
        comparison=comparison,
        sale_constat=sale,
        delivery_constat=delivery,
        previous_hash=previous_hash,
        sealed_at_utc=datetime.now(UTC),
        content_sha256="",
    )
    digest = sha256(_canonical_json(draft).encode()).hexdigest()
    return draft.model_copy(update={"content_sha256": digest})


def verify_certificate(certificate: Certificate) -> bool:
    """Recompute the seal; True only when the certificate is intact."""
    expected = sha256(_canonical_json(certificate).encode()).hexdigest()
    actual = certificate.content_sha256
    if len(actual) != len(expected):
        return False
    try:
        return hmac.compare_digest(expected, actual)
    except TypeError:
        return False


def verify_chain(certificates: list[Certificate]) -> bool:
    """Verify every seal and every link of a chain, in order."""
    previous: str | None = None
    for certificate in certificates:
        if not verify_certificate(certificate):
            return False
        if certificate.previous_hash != previous:
            return False
        previous = certificate.content_sha256
    return True


def certificate_qr_payload(certificate: Certificate) -> str:
    """Build the compact QR payload: marker, id and seal."""
    return f"TEMOIN1:{certificate.certificate_id}:{certificate.content_sha256}"


def certificate_qr_png(certificate: Certificate) -> bytes:
    """Render the QR code of a certificate as PNG bytes."""
    qr = qrcode.QRCode(box_size=8, border=2)
    qr.add_data(certificate_qr_payload(certificate))
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def load_certificate(path: str | Path) -> Certificate:
    """Load a certificate file; French error when missing or invalid."""
    source = Path(path)
    if not source.is_file():
        raise CertificateError(f"Certificat introuvable : « {source} ».")
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as error:
        raise CertificateError(f"Certificat illisible : « {source} ».") from error
    try:
        return Certificate.model_validate_json(raw)
    except ValueError as error:
        raise CertificateError("Ce fichier n'est pas un certificat Témoin valide.") from error
