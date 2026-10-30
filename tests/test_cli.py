"""Tests for the command-line parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from temoin.__main__ import build_parser, main
from temoin.schemas import Comparison, Report
from temoin.seal import seal_comparison


def test_parser_constat_arguments() -> None:
    """python -m temoin constat photo.jpg --code K7 parses correctly."""
    args = build_parser().parse_args(["constat", "photo.jpg", "--code", "K7"])
    assert (args.command, args.photo, args.code) == ("constat", "photo.jpg", "K7")


def test_parser_compare_arguments() -> None:
    """compare with both photos and codes parses correctly."""
    args = build_parser().parse_args(
        [
            "compare",
            "vente.jpg",
            "livraison.jpg",
            "--sale-code",
            "K7",
            "--delivery-code",
            "Q9",
        ]
    )
    assert (args.command, args.sale, args.delivery) == ("compare", "vente.jpg", "livraison.jpg")
    assert (args.sale_code, args.delivery_code) == ("K7", "Q9")


def test_parser_compare_codes_are_optional() -> None:
    """compare works without codes (freshness then fails honestly)."""
    args = build_parser().parse_args(["compare", "vente.jpg", "livraison.jpg"])
    assert (args.sale_code, args.delivery_code) == (None, None)


def test_parser_verify_arguments() -> None:
    """verify with a certificate path parses correctly."""
    args = build_parser().parse_args(["verify", "cert-1.json"])
    assert (args.command, args.certificate) == ("verify", "cert-1.json")


def test_verify_command_reports_intact_and_tampered(
    tmp_path: Path,
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """verify exits 0 on intact certificates and 1 on tampered ones."""
    cert = seal_comparison(
        comparison,
        certificate_id="c1",
        previous_hash=None,
        sale=sale_report,
        delivery=delivery_report,
    )
    good = tmp_path / "cert-1.json"
    good.write_text(cert.model_dump_json(indent=2), encoding="utf-8")
    assert main(["verify", str(good)]) == 0
    out = capsys.readouterr().out
    assert "INTACT" in out
    assert "CONFORME" in out
    payload = json.loads(good.read_text(encoding="utf-8"))
    payload["comparison"]["confidence"] = 12
    bad = tmp_path / "cert-1.tampered.json"
    bad.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    assert main(["verify", str(bad)]) == 1
    assert "MODIFIÉ" in capsys.readouterr().out
    assert main(["verify", str(tmp_path / "nope.json")]) == 1
