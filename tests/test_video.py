"""Tests for multi-view video constats (offline, no network)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from PIL import Image

from temoin.config import Settings
from temoin.schemas import (
    Alert,
    ChallengeCheck,
    Checkpoint,
    Defect,
    FrameReport,
    Report,
    condition_grade_for,
)
from temoin.video import (
    VideoError,
    create_video_report,
    extract_frames,
    fuse_alerts,
    fuse_challenge,
    fuse_checkpoints,
    fuse_defects,
    is_fresh_video,
    save_frames,
)
from tests.fake_client import FakeChatClient


def _make_video(path: Path, colors: tuple[tuple[int, int, int], ...], fps: int = 10) -> Path:
    """Write a tiny mp4 with one solid-color frame per color."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for color in colors:
        writer.write(np.full((120, 160, 3), color, dtype=np.uint8))
    writer.release()
    return path


def _defect_dict(
    id: str,
    type: str,
    zone: str,
    severity: str = "moderate",
    distinctive: bool = True,
    bbox: list[int] | None = None,
) -> dict[str, Any]:
    """Build one defect payload."""
    return {
        "id": id,
        "type": type,
        "description": f"{type} {id}.",
        "zone": zone,
        "severity": severity,
        "bbox": bbox,
        "distinctive": distinctive,
    }


def _report(
    defects: list[dict[str, Any]],
    *,
    visible: bool = True,
    read: str | None = "K7",
    alerts: list[dict[str, str]] | None = None,
    checkpoints: list[dict[str, str]] | None = None,
) -> Report:
    """Build a validated frame Report."""
    validated = [Defect.model_validate(item) for item in defects]
    return Report(
        description="Casque beige.",
        defects=validated,
        materials=["plastique"],
        declared_materials=[],
        condition_grade=condition_grade_for(validated),
        challenge=ChallengeCheck(expected_code="K7", visible=visible, read_code=read),
        checkpoints=[Checkpoint.model_validate(item) for item in (checkpoints or [])],
        alerts=[Alert.model_validate(item) for item in (alerts or [])],
        image_sha256="a" * 64,
        created_at_utc=datetime.now(UTC),
        prompt_version="1",
    )


def _draft_json(
    defects: list[dict[str, Any]],
    *,
    visible: bool = True,
    read: str | None = "K7",
    alerts: list[dict[str, str]] | None = None,
    checkpoints: list[dict[str, str]] | None = None,
) -> str:
    """Build a valid model answer for one frame."""
    return json.dumps(
        {
            "description": "Casque beige.",
            "materials": ["plastique"],
            "defects": defects,
            "challenge": {
                "expected_code": "K7",
                "visible": visible,
                "read_code": read,
                "looks_printed_or_edited": False,
            },
            "checkpoints": checkpoints or [],
            "alerts": alerts or [],
        }
    )


def test_extract_frames_samples_uniformly(tmp_path: Path) -> None:
    """10 frames at 10 fps sampled to 3 give t=0.0, 0.4, 0.9 seconds."""
    video = _make_video(tmp_path / "tour.mp4", ((255, 0, 0),) * 10)
    frames = extract_frames(video, max_frames=3, max_pixels=1024, max_megabytes=200.0)
    assert [frame.index for frame in frames] == [0, 1, 2]
    assert [frame.timestamp_s for frame in frames] == [0.0, 0.4, 0.9]
    assert all(isinstance(frame.image, Image.Image) for frame in frames)
    assert frames[0].image.size == (160, 120)
    assert frames[0].image.mode == "RGB"


def test_extract_frames_keeps_short_videos_whole(tmp_path: Path) -> None:
    """A 2-frame video with max 6 returns both frames."""
    video = _make_video(tmp_path / "tour.mp4", ((0, 255, 0), (0, 0, 255)))
    frames = extract_frames(video, max_frames=6, max_pixels=1024, max_megabytes=200.0)
    assert len(frames) == 2


def test_extract_frames_rejects_bad_inputs(tmp_path: Path) -> None:
    """Missing, foreign, oversized or unreadable videos are plain errors."""
    with pytest.raises(VideoError, match="introuvable"):
        extract_frames(tmp_path / "nope.mp4", max_frames=2, max_pixels=64, max_megabytes=1.0)
    foreign = tmp_path / "tour.txt"
    foreign.write_bytes(b"nope")
    with pytest.raises(VideoError, match="non accepté"):
        extract_frames(foreign, max_frames=2, max_pixels=64, max_megabytes=1.0)
    video = _make_video(tmp_path / "tour.mp4", ((1, 2, 3),))
    with pytest.raises(VideoError, match="dépasse"):
        extract_frames(video, max_frames=2, max_pixels=64, max_megabytes=0.000001)
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is not a video" * 10)
    with pytest.raises(VideoError, match="vidéo"):
        extract_frames(broken, max_frames=2, max_pixels=64, max_megabytes=1.0)


def test_save_frames_writes_jpegs(tmp_path: Path) -> None:
    """Sampled frames are saved as frame-00.jpg and frame-01.jpg."""
    video = _make_video(tmp_path / "tour.mp4", ((9, 9, 9), (8, 8, 8)))
    frames = extract_frames(video, max_frames=2, max_pixels=64, max_megabytes=1.0)
    paths = save_frames(frames, tmp_path / "frames", jpeg_quality=80)
    assert [path.name for path in paths] == ["frame-00.jpg", "frame-01.jpg"]
    assert all(path.is_file() for path in paths)


def test_fuse_defects_groups_and_counts() -> None:
    """Same flaw on two frames fuses with 2 confirmations; others stay single."""
    first = _report(
        [
            _defect_dict("D1", "rayure", "devant", bbox=[100, 100, 200, 200]),
            _defect_dict("D2", "tache", "derrière", severity="minor", distinctive=False),
        ]
    )
    second = _report(
        [
            _defect_dict("D1", "rayure", "devant", bbox=[110, 110, 210, 210]),
            _defect_dict("D9", "fissure", "côté gauche", severity="major"),
        ]
    )
    frames = [
        FrameReport(frame_index=0, timestamp_s=0.0, report=first),
        FrameReport(frame_index=1, timestamp_s=1.0, report=second),
    ]
    fused = fuse_defects(frames)
    assert [(flaw.id, flaw.confirmations, flaw.frames) for flaw in fused] == [
        ("F1", 2, [0, 1]),
        ("F2", 1, [0]),
        ("F3", 1, [1]),
    ]
    assert fused[0].type == "rayure"
    assert fused[0].distinctive is True


def test_fuse_defects_vetoes_disjoint_boxes() -> None:
    """Same kind and zone but disjoint boxes stay two flaws."""
    first = _report([_defect_dict("D1", "rayure", "devant", bbox=[0, 0, 100, 100])])
    second = _report([_defect_dict("D1", "rayure", "devant", bbox=[900, 900, 1000, 1000])])
    frames = [
        FrameReport(frame_index=0, timestamp_s=0.0, report=first),
        FrameReport(frame_index=1, timestamp_s=1.0, report=second),
    ]
    fused = fuse_defects(frames)
    assert [(flaw.id, flaw.confirmations) for flaw in fused] == [("F1", 1), ("F2", 1)]


def test_fuse_defects_empty() -> None:
    """No frames fuse to no flaws."""
    assert fuse_defects([]) == []


def test_fuse_challenge_counts_frames() -> None:
    """Code seen on 2 of 3 frames verifies with the read code."""
    frames = [
        FrameReport(frame_index=index, timestamp_s=float(index), report=report)
        for index, report in enumerate(
            [
                _report([], read="K7"),
                _report([], visible=False, read=None),
                _report([], read="k7"),
            ]
        )
    ]
    challenge = fuse_challenge("K7", frames)
    assert (challenge.seen_frames, challenge.total_frames) == (2, 3)
    assert (challenge.read_code, challenge.verified) == ("K7", True)
    hidden = fuse_challenge("K7", frames[1:2])
    assert (hidden.seen_frames, hidden.verified, hidden.read_code) == (0, False, None)


def test_fuse_alerts_dedupes() -> None:
    """Repeated alerts merge once, order kept."""
    first = _report([], alerts=[{"category": "blurry", "message": "Floue."}])
    second = _report(
        [],
        alerts=[
            {"category": "blurry", "message": "Floue."},
            {"category": "screen_photo", "message": "Écran."},
        ],
    )
    frames = [
        FrameReport(frame_index=0, timestamp_s=0.0, report=first),
        FrameReport(frame_index=1, timestamp_s=1.0, report=second),
    ]
    assert [(alert.category, alert.message) for alert in fuse_alerts(frames)] == [
        ("blurry", "Floue."),
        ("screen_photo", "Écran."),
    ]


def test_fuse_checkpoints_remaps_ids() -> None:
    """Reference checkpoints move from D-ids to F-ids; unknown ids drop."""
    first = _report(
        [_defect_dict("D1", "rayure", "devant")],
        checkpoints=[
            {"defect_id": "D1", "instruction": "Gros plan."},
            {"defect_id": "DX", "instruction": "Perdu."},
        ],
    )
    frames = [FrameReport(frame_index=0, timestamp_s=0.0, report=first)]
    points = fuse_checkpoints(frames, fuse_defects(frames))
    assert [(point.defect_id, point.instruction) for point in points] == [("F1", "Gros plan.")]


def test_create_video_report_fuses_frames(tmp_path: Path, test_settings: Settings) -> None:
    """Three frame constats fuse: shared flaw confirmed 3x, code seen 2/3."""
    video = _make_video(tmp_path / "tour.mp4", ((255, 0, 0), (0, 255, 0), (0, 0, 255)))
    shared = _defect_dict("D1", "rayure", "devant", bbox=[100, 100, 200, 200])
    extra = _defect_dict("D2", "tache", "derrière", severity="minor", distinctive=False)
    fake = FakeChatClient(
        [
            _draft_json([shared]),
            _draft_json([shared], alerts=[{"category": "blurry", "message": "B."}]),
            _draft_json(
                [shared, extra],
                visible=False,
                read=None,
                checkpoints=[{"defect_id": "D1", "instruction": "Gros plan."}],
            ),
        ]
    )
    calls: list[tuple[int, int]] = []
    report = create_video_report(
        video,
        "K7",
        settings=test_settings,
        client=fake,
        frames_dir=tmp_path / "frames",
        on_frame=lambda done, total: calls.append((done, total)),
    )
    assert [frame.timestamp_s for frame in report.frames] == [0.0, 0.1, 0.2]
    assert [(flaw.id, flaw.confirmations, flaw.frames) for flaw in report.fused_defects] == [
        ("F1", 3, [0, 1, 2]),
        ("F2", 1, [2]),
    ]
    assert (report.challenge.seen_frames, report.challenge.total_frames) == (2, 3)
    assert report.challenge.verified is True
    assert is_fresh_video(report) is True
    assert [point.defect_id for point in report.checkpoints] == ["F1"]
    assert [alert.category for alert in report.alerts] == ["blurry"]
    assert len(report.video_sha256) == 64
    assert report.prompt_version == "2"
    assert calls == [(1, 3), (2, 3), (3, 3)]
    assert sorted(path.name for path in (tmp_path / "frames").iterdir()) == [
        "frame-00.jpg",
        "frame-01.jpg",
        "frame-02.jpg",
    ]


def test_create_video_report_rejects_zero_frames(tmp_path: Path, test_settings: Settings) -> None:
    """max_frames=0 is a plain-language error."""
    video = _make_video(tmp_path / "tour.mp4", ((1, 1, 1),))
    with pytest.raises(VideoError, match="au moins une vue"):
        create_video_report(video, "K7", settings=test_settings, max_frames=0)
