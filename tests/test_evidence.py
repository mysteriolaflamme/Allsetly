"""Tests for the evidence board (offline, synthetic images)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from temoin.__main__ import main
from temoin.evidence import BoardRow, build_board, rows_from_reports
from temoin.schemas import Comparison, Defect, Report
from temoin.seal import seal_comparison


def _defect(id: str, bbox: list[int] | None) -> Defect:
    """Build one defect with an optional box."""
    return Defect(
        id=id,
        type="rayure",
        description=f"Rayure {id}.",
        zone="devant",
        severity="moderate",
        bbox=bbox,
        distinctive=True,
    )


def test_rows_align_matches_with_boxes(sale_report: Report, delivery_report: Report) -> None:
    """Each match row carries both boxes and readable captions."""
    sale = sale_report.model_copy(update={"defects": [_defect("D1", [100, 100, 300, 300])]})
    delivery = delivery_report.model_copy(update={"defects": [_defect("D1", [120, 120, 320, 320])]})
    comparison = Comparison.model_validate(
        {
            **_comparison_payload(),
            "matches": [
                {
                    "sale_defect_id": "D1",
                    "delivery_defect_id": "D1",
                    "status": "matched",
                    "note": "Même rayure.",
                }
            ],
        }
    )
    rows = rows_from_reports(sale, delivery, comparison)
    assert len(rows) == 1
    assert rows[0].sale_bbox == [100, 100, 300, 300]
    assert rows[0].delivery_bbox == [120, 120, 320, 320]
    assert rows[0].sale_missing is False
    assert "D1" in rows[0].sale_caption


def test_rows_flag_missing_side(sale_report: Report, delivery_report: Report) -> None:
    """A missing match flags its absent side for the placeholder cell."""
    sale = sale_report.model_copy(update={"defects": [_defect("D3", None)]})
    delivery = delivery_report.model_copy(update={"defects": []})
    comparison = Comparison.model_validate(
        {
            **_comparison_payload(),
            "matches": [
                {
                    "sale_defect_id": "D3",
                    "delivery_defect_id": None,
                    "status": "missing",
                    "note": "Rayure absente.",
                }
            ],
        }
    )
    rows = rows_from_reports(sale, delivery, comparison)
    assert rows[0].delivery_missing is True
    assert "non visible" in rows[0].delivery_caption


def test_board_geometry_and_status_frame() -> None:
    """One matched row: exact size and green status frames."""
    sale = Image.new("RGB", (400, 300), (200, 50, 50))
    delivery = Image.new("RGB", (400, 300), (50, 50, 200))
    rows = [
        BoardRow(
            label="D1 <-> D1",
            status="matched",
            note="Même rayure.",
            sale_caption="Vente · D1",
            delivery_caption="Livraison · D1",
            sale_bbox=[100, 100, 300, 300],
            delivery_bbox=[100, 100, 300, 300],
        )
    ]
    board = build_board(sale, delivery, rows)
    assert board.size == (884, 356)
    assert board.getpixel((18, 18)) == (46, 204, 113)
    assert board.getpixel((314, 18)) == (46, 204, 113)


def test_board_placeholder_and_clamp() -> None:
    """Missing sides are gray; wild boxes are clamped without crashing."""
    sale = Image.new("RGB", (400, 300), (200, 50, 50))
    delivery = Image.new("RGB", (400, 300), (50, 50, 200))
    rows = [
        BoardRow(
            label="D3 <-> ?",
            status="missing",
            note="Rayure absente.",
            sale_caption="Vente · D3",
            delivery_caption="Livraison · non visible",
            sale_bbox=[-50, -50, 2000, 2000],
            delivery_bbox=None,
            delivery_missing=True,
        )
    ]
    board = build_board(sale, delivery, rows)
    assert board.size == (884, 356)
    assert board.getpixel((452, 156)) == (238, 238, 238)
    assert board.getpixel((18, 18)) == (231, 76, 60)


def test_board_without_rows() -> None:
    """No match still renders an explicit board."""
    board = build_board(Image.new("RGB", (10, 10)), Image.new("RGB", (10, 10)), [])
    assert board.size == (640, 120)


def test_board_command_renders_certificate(
    tmp_path: Path,
    photo_path: Path,
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """board renders the sealed matches and prints the verdict."""
    certificate = seal_comparison(
        comparison,
        certificate_id="c1",
        previous_hash=None,
        sale=sale_report,
        delivery=delivery_report,
    )
    cert_path = tmp_path / "c1.json"
    cert_path.write_text(certificate.model_dump_json(), encoding="utf-8")
    out_path = tmp_path / "board.png"
    assert (
        main(
            [
                "board",
                str(cert_path),
                "--sale",
                str(photo_path),
                "--delivery",
                str(photo_path),
                "--output",
                str(out_path),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Planche : 2" in out
    assert "CONFORME" in out
    assert out_path.is_file()


def test_board_command_refuses_tampered_certificate(
    tmp_path: Path,
    photo_path: Path,
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A tampered certificate exits 1 with a plain-language error."""
    certificate = seal_comparison(
        comparison,
        certificate_id="c1",
        previous_hash=None,
        sale=sale_report,
        delivery=delivery_report,
    )
    cert_path = tmp_path / "c1.json"
    tampered = certificate.model_dump_json().replace('"CONFORME"', '"NON CONFORME"')
    cert_path.write_text(tampered, encoding="utf-8")
    assert (
        main(
            [
                "board",
                str(cert_path),
                "--sale",
                str(photo_path),
                "--delivery",
                str(photo_path),
            ]
        )
        == 1
    )
    assert "MODIFIÉ" in capsys.readouterr().err


def _comparison_payload() -> dict[str, object]:
    """Return a minimal comparison payload for validation."""
    return {
        "verdict": "CONFORME",
        "model_verdict": "CONFORME",
        "capped_reason": None,
        "confidence": 90,
        "same_object": True,
        "new_damage": False,
        "justification": "Mêmes défauts.",
        "contradictions": [],
        "missing_proof": [],
        "alerts": [],
        "freshness_sale_ok": True,
        "freshness_delivery_ok": True,
        "sale_image_sha256": "a" * 64,
        "delivery_image_sha256": "b" * 64,
        "created_at_utc": "2026-10-30T00:00:00Z",
        "prompt_version": "1",
        "disclaimer": "D.",
        "trail": [],
    }
