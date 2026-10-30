"""Tests for the Depth Matrix (offline, synthetic stereo pairs)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from temoin.__main__ import main
from temoin.config import Settings
from temoin.stereo import (
    DepthMatrix,
    StereoError,
    compute_depth,
    defect_relief,
    depth_from_video,
    load_pair,
    pick_stereo_pair,
    relief_cloud,
    relief_wiggle,
)
from temoin.video import VideoError


def _dots(seed: int = 11, size: int = 320) -> Image.Image:
    """Build a deterministic random-dot texture, ideal for stereo."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(
        rng.integers(0, 256, size=(size, size), dtype=np.uint8), mode="L"
    ).convert("RGB")


def _shifted_pair(shift: int = 16, bump: int = 24) -> tuple[Image.Image, Image.Image]:
    """Build a pair: global shift plus a raised central square."""
    base = np.asarray(_dots()).copy()
    right = np.zeros_like(base)
    right[:, :-shift] = base[:, shift:]
    lo, hi = 128, 192
    right[lo:hi, lo:hi] = base[lo:hi, lo + bump : hi + bump]
    return Image.fromarray(base), Image.fromarray(right)


def _make_stereo_video(path: Path, shifts: tuple[int, ...] = (0, 12, 24, 36)) -> Path:
    """Write a tiny avi sliding over a chunky random texture."""
    rng = np.random.default_rng(7)
    base = np.full((320, 320, 3), 128, dtype=np.uint8)
    for _ in range(1500):
        x, y = (int(value) for value in rng.integers(0, 308, size=2))
        size = int(rng.integers(4, 12))
        base[y : y + size, x : x + size] = int(rng.integers(0, 256))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 320))
    for shift in shifts:
        frame = np.zeros_like(base)
        frame[:, shift:] = base[:, : 320 - shift]
        writer.write(frame)
    writer.release()
    return path


def test_pick_stereo_pair_finds_overlapping_views() -> None:
    """The shifted pair matches; identical or blank views are rejected."""
    left, right = _shifted_pair()
    assert pick_stereo_pair([left, right]) == (0, 1)
    with pytest.raises(StereoError, match="trop semblables"):
        pick_stereo_pair([left, left])
    blank = Image.new("RGB", (160, 120))
    with pytest.raises(StereoError, match="trop semblables"):
        pick_stereo_pair([blank, blank])
    with pytest.raises(StereoError, match="deux vues"):
        pick_stereo_pair([_dots()])


def test_compute_depth_measures_known_shift(test_settings: Settings) -> None:
    """A 16px global shift gives high coverage and relief in 0..1."""
    left, right = _shifted_pair()
    matrix = compute_depth(left, right, settings=test_settings)
    assert matrix.coverage > 0.5
    assert (matrix.left_index, matrix.right_index) == (0, 1)
    flat = [value for row in matrix.values for value in row]
    assert all(value == -1.0 or 0.0 <= value <= 1.0 for value in flat)
    assert any(value >= 0.0 for value in flat)


def test_defect_relief_detects_raised_patch(test_settings: Settings) -> None:
    """The raised square reads closer than its ring (positive delta)."""
    left, right = _shifted_pair()
    matrix = compute_depth(left, right, settings=test_settings)
    stats = defect_relief(matrix, [400, 400, 600, 600])
    assert stats.delta > 0.05
    flat = defect_relief(matrix, [700, 100, 800, 200])
    assert abs(flat.delta) < 0.25


def test_compute_depth_rejects_featureless_views(test_settings: Settings) -> None:
    """Blank views give no matchable points."""
    blank = Image.new("RGB", (160, 120))
    with pytest.raises(StereoError, match="points communs"):
        compute_depth(blank, blank, settings=test_settings)


def test_relief_wiggle_returns_gif_bytes(test_settings: Settings) -> None:
    """The parallax render is a valid animated GIF."""
    left, right = _shifted_pair()
    matrix = compute_depth(left, right, settings=test_settings)
    data = relief_wiggle(left, matrix)
    assert data[:6] in (b"GIF89a", b"GIF87a")
    assert len(data) > 1000


def test_load_pair_rejects_missing_files(tmp_path: Path) -> None:
    """Missing view files are a plain-language error."""
    with pytest.raises(StereoError, match="lire"):
        load_pair(tmp_path / "a.jpg", tmp_path / "b.jpg")


def test_relief_cloud_builds_colored_points(test_settings: Settings) -> None:
    """The cloud holds measured points only, colored by the photo."""
    left, right = _shifted_pair()
    matrix = compute_depth(left, right, settings=test_settings)
    figure = relief_cloud(left, matrix, max_points=500)
    trace = figure.data[0]
    assert len(trace.x) > 100
    assert all(0.0 <= float(value) <= 1.0 for value in trace.z)
    assert len(trace.marker.color) == len(trace.x)


def test_relief_cloud_rejects_empty_matrix() -> None:
    """A matrix without measured relief is a plain-language error."""
    empty = DepthMatrix(
        width=4,
        height=4,
        values=tuple(tuple(-1.0 for _ in range(4)) for _ in range(4)),
        coverage=0.0,
        left_index=0,
        right_index=1,
    )
    with pytest.raises(StereoError, match="3D"):
        relief_cloud(Image.new("RGB", (4, 4)), empty)


def test_stereo_command_measures_and_saves_viewers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    test_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """stereo prints coverage, relief contrast and saves GIF + HTML (offline)."""
    left, right = _shifted_pair()
    left_path = tmp_path / "left.png"
    right_path = tmp_path / "right.png"
    left.save(left_path)
    right.save(right_path)
    monkeypatch.setattr("temoin.__main__.load_settings", lambda *args: test_settings)
    wiggle = tmp_path / "relief.gif"
    cloud = tmp_path / "cloud.html"
    assert (
        main(
            [
                "stereo",
                str(left_path),
                str(right_path),
                "--wiggle",
                str(wiggle),
                "--cloud",
                str(cloud),
                "--bbox",
                "400,400,600,600",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Relief mesuré" in out
    assert "Zone :" in out
    assert wiggle.is_file()
    assert cloud.is_file()


def test_depth_from_video_measures_relief(tmp_path: Path, test_settings: Settings) -> None:
    """A sliding-dot video gives two distinct views and solid coverage."""
    video = _make_stereo_video(tmp_path / "tour.avi")
    matrix, left = depth_from_video(video, settings=test_settings)
    assert matrix.coverage > 0.25
    assert matrix.left_index != matrix.right_index
    assert left.mode == "RGB"


def test_depth_from_video_rejects_blank_video(tmp_path: Path, test_settings: Settings) -> None:
    """A video without texture gives no overlapping pair."""
    path = tmp_path / "blank.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for _ in range(3):
        writer.write(np.full((120, 160, 3), 128, dtype=np.uint8))
    writer.release()
    with pytest.raises(StereoError, match="trop semblables"):
        depth_from_video(path, settings=test_settings)


def test_depth_from_video_rejects_missing_file(tmp_path: Path, test_settings: Settings) -> None:
    """A missing video keeps the video error contract."""
    with pytest.raises(VideoError, match="introuvable"):
        depth_from_video(tmp_path / "nope.mp4", settings=test_settings)


def test_video_depth_command_measures_from_video(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    test_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """video-depth prints used views, coverage and saves the GIF (offline)."""
    video = _make_stereo_video(tmp_path / "tour.avi")
    monkeypatch.setattr("temoin.__main__.load_settings", lambda *args: test_settings)
    wiggle = tmp_path / "relief.gif"
    assert main(["video-depth", str(video), "--wiggle", str(wiggle)]) == 0
    out = capsys.readouterr().out
    assert "Vues utilisées" in out
    assert "Relief mesuré" in out
    assert wiggle.is_file()
