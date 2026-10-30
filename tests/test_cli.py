"""Tests for the command-line parser."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from temoin.__main__ import build_parser, main
from temoin.config import Settings
from temoin.forensics import ForensicsReport, QualityReport
from temoin.meta_api import Segmentation, SegmentMask, VoiceTranscript, VoiceTurn
from temoin.schemas import (
    Comparison,
    FrameReport,
    FusedDefect,
    Report,
    VideoChallenge,
    VideoReport,
)
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


def test_parser_segment_arguments() -> None:
    """segment with a photo and a concept parses correctly."""
    args = build_parser().parse_args(["segment", "photo.jpg", "--concept", "headphones"])
    assert (args.command, args.photo, args.concept, args.output) == (
        "segment",
        "photo.jpg",
        "headphones",
        None,
    )


def test_parser_transcribe_arguments() -> None:
    """transcribe with bias flags parses correctly."""
    args = build_parser().parse_args(
        ["transcribe", "note.wav", "--language-bias", "french", "--keywords", "casque"]
    )
    assert (args.command, args.audio) == ("transcribe", "note.wav")
    assert (args.language_bias, args.keywords) == (["french"], ["casque"])


def test_segment_command_saves_overlay(
    monkeypatch: pytest.MonkeyPatch,
    photo_path: Path,
    test_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """segment writes the overlay PNG next to the photo and prints a summary."""
    monkeypatch.setattr("temoin.__main__.load_settings", lambda *args: test_settings)
    width, height = 90, 80
    mask = SegmentMask(
        object_id="0",
        left=10.0,
        top=10.0,
        right=100.0,
        bottom=90.0,
        width=width,
        height=height,
        pixels=bytes([1]) * (width * height),
    )
    segmentation = Segmentation(
        concept="headphones", image_width=640, image_height=480, masks=(mask,)
    )
    monkeypatch.setattr("temoin.__main__.segment_image", lambda *a, **k: segmentation)
    assert main(["segment", str(photo_path), "--concept", "headphones"]) == 0
    assert (photo_path.parent / "photo-sam.png").is_file()
    out = capsys.readouterr().out
    assert "1 masque" in out
    assert "headphones" in out


def test_transcribe_command_prints_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    test_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """transcribe prints the turns and the duration."""
    monkeypatch.setattr("temoin.__main__.load_settings", lambda *args: test_settings)
    transcript = VoiceTranscript(
        text="bonjour",
        duration_seconds=2.5,
        turns=(VoiceTurn(start_seconds=0.5, text="bonjour", speaker="A"),),
    )
    monkeypatch.setattr("temoin.__main__.transcribe_audio", lambda *a, **k: transcript)
    audio = tmp_path / "note.wav"
    audio.write_bytes(b"fake")
    assert main(["transcribe", str(audio)]) == 0
    out = capsys.readouterr().out
    assert "[   0.50s] A: bonjour" in out
    assert "Durée : 2.5 s" in out


def test_parser_video_constat_arguments() -> None:
    """video-constat with a video, a code and max-frames parses correctly."""
    args = build_parser().parse_args(
        ["video-constat", "tour.mp4", "--code", "K7", "--max-frames", "4"]
    )
    assert (args.command, args.video, args.code, args.max_frames) == (
        "video-constat",
        "tour.mp4",
        "K7",
        4,
    )


def test_video_constat_command_prints_fusion(
    monkeypatch: pytest.MonkeyPatch,
    test_settings: Settings,
    sale_report: Report,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """video-constat prints views, fused flaws and code coverage."""
    monkeypatch.setattr("temoin.__main__.load_settings", lambda *args: test_settings)
    video = VideoReport(
        description="Casque beige.",
        fused_defects=[
            FusedDefect(
                id="F1",
                type="rayure",
                description="R.",
                zone="devant",
                severity="moderate",
                distinctive=True,
                confirmations=2,
                frames=[0, 1],
            )
        ],
        frames=[
            FrameReport(frame_index=0, timestamp_s=0.0, report=sale_report),
            FrameReport(frame_index=1, timestamp_s=1.0, report=sale_report),
        ],
        challenge=VideoChallenge(
            expected_code="K7", seen_frames=2, total_frames=2, read_code="K7", verified=True
        ),
        checkpoints=[],
        alerts=[],
        video_sha256="v" * 64,
        created_at_utc=datetime.now(UTC),
        prompt_version="1",
    )
    monkeypatch.setattr("temoin.__main__.create_video_report", lambda *a, **k: video)
    assert main(["video-constat", "tour.mp4", "--code", "K7"]) == 0
    out = capsys.readouterr().out
    assert "Vues analysées : 2" in out
    assert "F1" in out
    assert "2/2 vues" in out


def test_parser_constat_materials_are_optional() -> None:
    """constat accepts an optional declared-materials list."""
    args = build_parser().parse_args(["constat", "photo.jpg", "--code", "K7"])
    assert args.materials is None
    args = build_parser().parse_args(
        ["constat", "photo.jpg", "--code", "K7", "--materials", "cuir,bois"]
    )
    assert args.materials == "cuir,bois"


def test_parser_forensics_arguments() -> None:
    """forensics with a photo and an ELA map path parses correctly."""
    args = build_parser().parse_args(["forensics", "photo.jpg", "--ela-map", "ela.png"])
    assert (args.command, args.photo, args.ela_map) == ("forensics", "photo.jpg", "ela.png")


def test_forensics_command_prints_signals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    photo_path: Path,
    test_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """forensics prints signals, coaching and saves the ELA map on request."""
    monkeypatch.setattr("temoin.__main__.load_settings", lambda *args: test_settings)
    made = ForensicsReport(
        suspicious=True,
        signals=("Montage possible.",),
        quality=QualityReport(
            sharpness=3.0,
            width=100,
            height=100,
            brightness=50.0,
            ok=False,
            advice=("Stabilisez.",),
        ),
        ela_mean=1.0,
        ela_p99=2.0,
        noise_median=1.0,
        noise_suspicious_blocks=9,
        noise_total_blocks=64,
        software=None,
    )
    monkeypatch.setattr("temoin.__main__.analyze_photo", lambda *a, **k: made)
    assert main(["forensics", "photo.jpg"]) == 0
    out = capsys.readouterr().out
    assert "Photo suspecte" in out
    assert "Montage possible." in out
    assert "Stabilisez." in out
    ela_path = tmp_path / "ela.png"
    assert main(["forensics", str(photo_path), "--ela-map", str(ela_path)]) == 0
    assert ela_path.is_file()
    assert "Carte ELA" in capsys.readouterr().out


def test_parser_stereo_arguments() -> None:
    """stereo with two views, viewers and a box parses correctly."""
    args = build_parser().parse_args(
        [
            "stereo",
            "left.jpg",
            "right.jpg",
            "--wiggle",
            "relief.gif",
            "--cloud",
            "cloud.html",
            "--bbox",
            "400,400,600,600",
        ]
    )
    assert (args.command, args.left, args.right) == ("stereo", "left.jpg", "right.jpg")
    assert (args.wiggle, args.cloud, args.bbox) == ("relief.gif", "cloud.html", "400,400,600,600")
