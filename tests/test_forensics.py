"""Tests for photo forensics (offline, synthetic images)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from temoin.config import Settings
from temoin.forensics import (
    analyze_photo,
    ela_map,
    ela_stats,
    exif_notes,
    noise_stats,
    quality_report,
)
from temoin.images import ImageError


def _gradient(size: int = 256) -> Image.Image:
    """Build a deterministic smooth grayscale image."""
    ramp = np.tile(np.arange(size, dtype=np.uint8), (size, 1))
    return Image.fromarray(ramp, mode="L").convert("RGB")


def _checker(size: int = 256, square: int = 16) -> Image.Image:
    """Build a deterministic black-and-white checkerboard."""
    tiles = np.indices((size, size)).sum(axis=0) // square % 2 * 255
    return Image.fromarray(tiles.astype(np.uint8), mode="L").convert("RGB")


def _save_jpeg(image: Image.Image, path: Path, quality: int = 95) -> Path:
    """Save an image as JPEG and return its path."""
    image.save(path, format="JPEG", quality=quality)
    return path


def test_ela_flags_spliced_composite(tmp_path: Path) -> None:
    """A pasted noisy region raises the recompression error above a clean photo."""
    base = _checker()
    clean = _save_jpeg(base, tmp_path / "clean.jpg")
    rng = np.random.default_rng(3)
    patch = Image.fromarray(rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8))
    composite = base.copy()
    composite.paste(patch, (64, 64))
    tampered = _save_jpeg(composite, tmp_path / "tampered.jpg")
    with Image.open(clean) as left, Image.open(tampered) as right:
        left.load()
        right.load()
        clean_stats = ela_stats(left.convert("RGB"))
        tampered_stats = ela_stats(right.convert("RGB"))
    assert tampered_stats.mean > clean_stats.mean
    assert tampered_stats.p99 >= tampered_stats.mean >= 0.0


def test_ela_map_returns_amplified_grayscale() -> None:
    """The ELA map is a same-size grayscale image."""
    result = ela_map(_gradient())
    assert result.mode == "L"
    assert result.size == (256, 256)


def test_noise_stats_ignores_flat_images() -> None:
    """Uniform images have no suspicious blocks."""
    stats = noise_stats(Image.new("RGB", (128, 128), color=(50, 60, 70)))
    assert stats.suspicious_blocks == 0
    assert stats.total_blocks == 64


def test_noise_stats_flags_half_noisy_composite() -> None:
    """A half-noise composite deviates from its own median."""
    rng = np.random.default_rng(7)
    left = np.full((128, 64, 3), 120, dtype=np.uint8)
    right = rng.integers(0, 256, size=(128, 64, 3), dtype=np.uint8)
    stats = noise_stats(Image.fromarray(np.concatenate([left, right], axis=1)))
    assert stats.suspicious_blocks > 0


def test_exif_notes_reads_software_and_dates(tmp_path: Path) -> None:
    """Editor software and date mismatches are reported; missing EXIF is quiet."""
    plain = _gradient()
    assert exif_notes(plain) == exif_notes(plain)
    assert exif_notes(plain).editor_found is False
    edited = plain.copy()
    tags = edited.getexif()
    tags[271] = "Adobe Photoshop"
    tags[36867] = "2026:10:01 10:00:00"
    tags[306] = "2026:10:02 10:00:00"
    edited.save(tmp_path / "edited.jpg", format="JPEG", exif=tags)
    with Image.open(tmp_path / "edited.jpg") as handle:
        handle.load()
        notes = exif_notes(handle.convert("RGB"))
    assert notes.software == "Adobe Photoshop"
    assert notes.editor_found is True
    assert notes.datetime_mismatch is True
    assert notes.developed is False


def test_quality_report_orders_sharp_above_blurred() -> None:
    """Blur lowers sharpness; the gate coaches between the two scores."""
    sharp = _checker(512)
    blurred = sharp.filter(ImageFilter.GaussianBlur(5))
    sharp_score = quality_report(sharp, blur_min=0.0).sharpness
    blurred_score = quality_report(blurred, blur_min=0.0).sharpness
    assert sharp_score > blurred_score
    midpoint = (sharp_score + blurred_score) / 2
    assert quality_report(sharp, blur_min=midpoint).ok is True
    gated = quality_report(blurred, blur_min=midpoint)
    assert gated.ok is False
    assert any("floue" in tip for tip in gated.advice)


def test_quality_report_coaches_dark_and_tiny() -> None:
    """Dark and tiny photos get specific French advice."""
    dark = quality_report(Image.new("RGB", (800, 600), color=(10, 10, 10)), blur_min=0.0)
    assert any("sombre" in tip for tip in dark.advice)
    tiny = quality_report(_gradient(100), blur_min=0.0)
    assert any("petite" in tip for tip in tiny.advice)


def test_analyze_photo_combines_signals(tmp_path: Path, test_settings: Settings) -> None:
    """Missing files error; a zero threshold flags any photo suspicious."""
    with pytest.raises(ImageError, match="introuvable"):
        analyze_photo(tmp_path / "nope.jpg", settings=test_settings)
    rng = np.random.default_rng(7)
    left = np.full((128, 64, 3), 120, dtype=np.uint8)
    right = rng.integers(0, 256, size=(128, 64, 3), dtype=np.uint8)
    noisy = Image.fromarray(np.concatenate([left, right], axis=1))
    photo = _save_jpeg(noisy, tmp_path / "photo.jpg")
    strict = replace(test_settings, forensics_noise_ratio=0.0)
    report = analyze_photo(photo, settings=strict)
    assert report.suspicious is True
    assert any("Bruit" in signal for signal in report.signals)
    assert report.quality.width == 128
    calm = analyze_photo(_save_jpeg(_gradient(), tmp_path / "calm.jpg"), settings=test_settings)
    assert calm.suspicious is False
    assert calm.signals == ()
