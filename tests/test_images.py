"""Tests for photo validation, resizing, encoding and fingerprinting."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from temoin.images import ImageError, compute_sha256, draw_defects, prepare_image
from temoin.schemas import Defect


def _make_photo(path: Path, size: tuple[int, int] = (640, 480)) -> Path:
    """Create a JPEG photo of the given size."""
    Image.new("RGB", size, color=(120, 80, 40)).save(path, format="JPEG")
    return path


def test_prepare_image_shrinks_long_side_to_limit(tmp_path: Path) -> None:
    """A 2000 px wide photo is shrunk to 1024 px, keeping proportions."""
    photo = _make_photo(tmp_path / "big.jpg", size=(2000, 1000))
    prepared = prepare_image(photo, max_pixels=1024, max_megabytes=15.0, jpeg_quality=90)
    assert (prepared.width, prepared.height) == (1024, 512)
    assert prepared.data_url.startswith("data:image/jpeg;base64,")
    assert len(prepared.sha256) == 64


def test_prepare_image_keeps_small_photo(tmp_path: Path) -> None:
    """A small photo is never enlarged."""
    photo = _make_photo(tmp_path / "small.jpg", size=(400, 300))
    prepared = prepare_image(photo, max_pixels=1024, max_megabytes=15.0, jpeg_quality=90)
    assert (prepared.width, prepared.height) == (400, 300)


def test_compute_sha256_is_stable() -> None:
    """Same bytes give the same fingerprint, different bytes differ."""
    assert compute_sha256(b"abc") == compute_sha256(b"abc")
    assert compute_sha256(b"abc") != compute_sha256(b"abd")


def test_missing_photo_raises_plain_error(tmp_path: Path) -> None:
    """A missing file raises a plain-language error."""
    with pytest.raises(ImageError, match="introuvable"):
        prepare_image(tmp_path / "nope.jpg", max_pixels=1024, max_megabytes=15.0, jpeg_quality=90)


def test_oversized_photo_is_rejected(tmp_path: Path) -> None:
    """A photo above the size limit is rejected with a plain message."""
    photo = _make_photo(tmp_path / "photo.jpg")
    with pytest.raises(ImageError, match="Mo"):
        prepare_image(photo, max_pixels=1024, max_megabytes=0.0, jpeg_quality=90)


def test_unsupported_format_is_rejected(tmp_path: Path) -> None:
    """GIF files are refused with the list of accepted formats."""
    photo = tmp_path / "photo.gif"
    photo.write_bytes(b"GIF89a" + b"\x00" * 100)
    with pytest.raises(ImageError, match="JPEG, PNG"):
        prepare_image(photo, max_pixels=1024, max_megabytes=15.0, jpeg_quality=90)


def test_unreadable_image_is_rejected(tmp_path: Path) -> None:
    """A .jpg file that is not an image raises a plain-language error."""
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"this is not an image" * 10)
    with pytest.raises(ImageError, match="Impossible de lire"):
        prepare_image(photo, max_pixels=1024, max_megabytes=15.0, jpeg_quality=90)


def test_heic_without_plugin_suggests_conversion(tmp_path: Path) -> None:
    """HEIC without pillow-heif explains how to convert the photo."""
    if importlib.util.find_spec("pillow_heif") is not None:
        pytest.skip("pillow-heif is installed")
    photo = tmp_path / "photo.heic"
    photo.write_bytes(b"fake-heic-bytes")
    with pytest.raises(ImageError, match="HEIC"):
        prepare_image(photo, max_pixels=1024, max_megabytes=15.0, jpeg_quality=90)


def test_draw_defects_frames_only_boxes(tmp_path: Path) -> None:
    """Boxes are drawn for defects with bboxes; the file is untouched."""
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (200, 100), color=(255, 255, 255)).save(photo, format="JPEG")
    before = photo.read_bytes()
    boxed = Defect.model_validate(
        {
            "id": "D1",
            "type": "rayure",
            "description": "x",
            "zone": "y",
            "severity": "major",
            "bbox": [0, 0, 1000, 1000],
            "distinctive": True,
        }
    )
    plain = Defect.model_validate(
        {
            "id": "D2",
            "type": "tache",
            "description": "x",
            "zone": "y",
            "severity": "minor",
            "bbox": None,
            "distinctive": False,
        }
    )
    framed = draw_defects(photo, [boxed, plain])
    assert framed.size == (200, 100)
    assert framed.mode == "RGB"
    assert photo.read_bytes() == before
    assert framed.getpixel((0, 0)) != (255, 255, 255)
