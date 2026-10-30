"""Tests for sealing, tamper detection, chains and QR codes (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from temoin.schemas import Certificate, Comparison, Report
from temoin.seal import (
    CertificateError,
    certificate_qr_payload,
    certificate_qr_png,
    load_certificate,
    seal_comparison,
    verify_certificate,
    verify_chain,
)


def _seal(
    comparison: Comparison,
    sale_report: Report,
    delivery_report: Report,
    *,
    cert_id: str = "c1",
    previous: str | None = None,
) -> Certificate:
    """Seal the fixture comparison."""
    return seal_comparison(
        comparison,
        certificate_id=cert_id,
        previous_hash=previous,
        sale=sale_report,
        delivery=delivery_report,
    )


def test_seal_then_verify_roundtrip(
    comparison: Comparison, sale_report: Report, delivery_report: Report
) -> None:
    """A fresh seal verifies as intact."""
    cert = _seal(comparison, sale_report, delivery_report)
    assert verify_certificate(cert) is True
    assert cert.previous_hash is None
    assert len(cert.content_sha256) == 64


def test_single_character_tamper_is_detected(
    tmp_path: Path, comparison: Comparison, sale_report: Report, delivery_report: Report
) -> None:
    """Flipping one JSON character breaks the seal while staying loadable."""
    cert = _seal(comparison, sale_report, delivery_report)
    path = tmp_path / "cert-1.json"
    path.write_text(cert.model_dump_json(indent=2), encoding="utf-8")
    assert verify_certificate(load_certificate(path)) is True
    payload = json.loads(path.read_text(encoding="utf-8"))
    original = payload["comparison"]["justification"]
    payload["comparison"]["justification"] = "X" + original[1:]
    tampered = tmp_path / "cert-1.tampered.json"
    tampered.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded = load_certificate(tampered)
    changed = sum(a != b for a, b in zip(loaded.comparison.justification, original, strict=True))
    assert changed == 1
    assert verify_certificate(loaded) is False


def test_tampered_seal_field_is_detected(
    comparison: Comparison, sale_report: Report, delivery_report: Report
) -> None:
    """Replacing the seal itself breaks verification."""
    cert = _seal(comparison, sale_report, delivery_report)
    forged = cert.model_copy(update={"content_sha256": "0" * 64})
    assert verify_certificate(forged) is False


def test_verify_chain_checks_links(
    comparison: Comparison, sale_report: Report, delivery_report: Report
) -> None:
    """A chained pair verifies; broken links and tampering fail."""
    first = _seal(comparison, sale_report, delivery_report, cert_id="c1")
    second = _seal(
        comparison, sale_report, delivery_report, cert_id="c2", previous=first.content_sha256
    )
    assert verify_chain([first, second]) is True
    assert verify_chain([first]) is True
    assert verify_chain([]) is True
    broken = second.model_copy(update={"previous_hash": "0" * 64})
    assert verify_chain([first, broken]) is False
    tampered = first.model_copy(update={"content_sha256": "0" * 64})
    assert verify_chain([tampered, second]) is False


def test_qr_payload_and_png(
    comparison: Comparison, sale_report: Report, delivery_report: Report
) -> None:
    """QR payload is compact; PNG output is deterministic."""
    cert = _seal(comparison, sale_report, delivery_report)
    payload = certificate_qr_payload(cert)
    assert payload == f"TEMOIN1:{cert.certificate_id}:{cert.content_sha256}"
    png = certificate_qr_png(cert)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert certificate_qr_png(cert) == png


def test_load_certificate_errors(tmp_path: Path) -> None:
    """Missing and invalid files give French errors."""
    with pytest.raises(CertificateError, match="introuvable"):
        load_certificate(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(CertificateError, match="pas un certificat"):
        load_certificate(bad)
    wrong = tmp_path / "wrong.json"
    wrong.write_text('{"hello": "world"}', encoding="utf-8")
    with pytest.raises(CertificateError, match="pas un certificat"):
        load_certificate(wrong)
