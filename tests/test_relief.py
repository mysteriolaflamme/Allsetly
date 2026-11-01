"""Tests for raking-light relief (offline, synthetic Lambertian scenes)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from temoin.__main__ import main
from temoin.config import Settings
from temoin.images import ImageError
from temoin.meta_api import MetaAPIError, Segmentation, SegmentMask
from temoin.model import ModelError
from temoin.relief import (
    FLAT_ALERT,
    ReliefError,
    colorize_relief,
    compute_relief,
    describe_relief,
    estimate_lights_hayakawa,
    frankot_chellappa,
    light_directions,
    load_stack,
    normals_image,
    object_mask,
    photometric_stereo,
    run_relief,
    threshold_mask,
)
from temoin.schemas import Comparison, ReliefDescription, ReliefReport, Report
from temoin.seal import seal_comparison, verify_certificate
from tests.fake_client import FakeChatClient


def _sphere(size: int = 96, radius: int = 36, albedo: float = 0.9):
    """Render a shaded sphere stack under the 4 calibrated lights."""
    lights = light_directions(4)
    ys, xs = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2
    dx, dy = xs - center, ys - center
    disk = dx**2 + dy**2 <= radius**2
    dz = np.sqrt(np.maximum(radius**2 - dx**2 - dy**2, 0))
    normals = np.zeros((size, size, 3))
    normals[disk] = np.stack([dx[disk], dy[disk], dz[disk]], axis=1) / radius
    grays = np.stack([np.clip(normals @ light, 0, None) * albedo for light in lights])
    images = [
        Image.fromarray((frame * 255).round().astype(np.uint8)).convert("RGB") for frame in grays
    ]
    return images, grays, disk, dz


def _flat_sheet(size: int = 96, albedo: float = 0.7):
    """Render a flat albedo sheet under the 4 calibrated lights."""
    level = albedo * light_directions(4)[0, 2]
    grays = np.full((4, size, size), level)
    images = [
        Image.fromarray((frame * 255).round().astype(np.uint8)).convert("RGB") for frame in grays
    ]
    return images, grays, np.ones((size, size), bool)


def _save_stack(images: list[Image.Image], folder: Path) -> list[str]:
    """Save a synthetic stack as PNG files, return their paths."""
    paths = []
    for number, image in enumerate(images):
        target = folder / f"rake-{number}.png"
        image.save(target)
        paths.append(str(target))
    return paths


def test_light_directions_follow_clock_order() -> None:
    """12h points up, 3h right, 6h down, 9h left, all unit at 45 degrees."""
    lights = light_directions(4)
    assert lights.shape == (4, 3)
    assert np.allclose(np.linalg.norm(lights, axis=1), 1.0)
    top, right, bottom, left = lights
    assert top[1] < 0 < bottom[1] and left[0] < 0 < right[0]
    assert bool((lights[:, 2] > 0).all())
    assert light_directions(8).shape == (8, 3)
    with pytest.raises(ReliefError, match="positions de lampe"):
        light_directions(2)


def test_calibrated_sphere_recovers_shape(test_settings: Settings) -> None:
    """A rendered sphere gives back a sphere: high correlation, not flat."""
    images, _, disk, dz = _sphere()
    maps = compute_relief(
        images, lights=None, mask=disk, mask_source="threshold", settings=test_settings
    )
    assert maps.coverage > 0.7
    assert not maps.flat
    assert maps.compute_seconds >= 0
    picked = maps.valid & disk
    recovered = maps.height[picked].astype(float)
    assert np.corrcoef(recovered, dz[picked])[0, 1] > 0.9


def test_flat_sheet_triggers_flat_alert(test_settings: Settings) -> None:
    """A flat surface gives uniform normals, no relief and the alert."""
    images, _, sheet = _flat_sheet()
    maps = compute_relief(
        images, lights=None, mask=sheet, mask_source="threshold", settings=test_settings
    )
    assert maps.coverage == 1.0
    assert maps.height_std < test_settings.relief_flat_height_std
    assert maps.normal_spread < test_settings.relief_flat_normal_spread
    assert maps.flat


def test_uncalibrated_sphere_finds_lights(test_settings: Settings) -> None:
    """Hayakawa SVD recovers the 4 lamp directions and the sphere shape."""
    images, grays, disk, dz = _sphere()
    estimated = estimate_lights_hayakawa(grays, disk)
    assert estimated.shape == (4, 3)
    dots = (estimated * light_directions(4)).sum(axis=1)
    assert bool((dots > 0.99).all())
    maps = compute_relief(
        images, lights=estimated, mask=disk, mask_source="threshold", settings=test_settings
    )
    assert not maps.flat
    picked = maps.valid & disk
    assert np.corrcoef(maps.height[picked].astype(float), dz[picked])[0, 1] > 0.9


def test_frankot_chellappa_integrates_gaussian_bump() -> None:
    """Analytic bump gradients integrate back to the bump (correlation)."""
    size = 64
    ys, xs = np.mgrid[0:size, 0:size]
    bump = np.exp(-((xs - 31.5) ** 2 + (ys - 31.5) ** 2) / (2 * 12.0**2))
    slope_y, slope_x = np.gradient(bump)
    normals = np.stack([-slope_x, -slope_y, np.ones_like(bump)], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    recovered = frankot_chellappa(normals, np.ones((size, size), bool))
    assert np.corrcoef(recovered.ravel(), bump.ravel())[0, 1] > 0.99


def test_photometric_stereo_rejects_shape_mismatch(test_settings: Settings) -> None:
    """Lights that do not match the photo count are a plain-language error."""
    _, grays, disk, _ = _sphere()
    with pytest.raises(ReliefError, match="ne correspondent pas"):
        photometric_stereo(grays, light_directions(3), disk)


def test_hayakawa_rejects_identical_photos() -> None:
    """Four copies of one photo carry no shading variety to factorize."""
    _, grays, disk, _ = _sphere()
    with pytest.raises(ReliefError, match="trop semblables"):
        estimate_lights_hayakawa(np.stack([grays[0]] * 4), disk)


def test_threshold_mask_keeps_central_object() -> None:
    """Otsu split keeps the centered square, drops the dark background."""
    frame = np.zeros((64, 64), dtype=np.uint8)
    frame[16:48, 16:48] = 200
    mask = threshold_mask(Image.fromarray(frame).convert("RGB"))
    assert bool(mask[32, 32])
    assert not bool(mask[0, 0])
    assert 0.1 < mask.mean() < 0.5


def _sam_segmentation(masks: tuple[SegmentMask, ...]) -> Segmentation:
    """Build a canned SAM answer on a 64x64 sent image."""
    return Segmentation(concept="object", image_width=64, image_height=64, masks=masks)


def test_object_mask_prefers_sam_then_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """SAM union wins; API failure or empty answer degrades to threshold."""
    small = Image.new("RGB", (32, 32), color=(200, 200, 200))
    full = SegmentMask(
        object_id="0",
        left=8.0,
        top=8.0,
        right=56.0,
        bottom=56.0,
        width=12,
        height=12,
        pixels=bytes([1]) * 144,
    )
    monkeypatch.setattr(
        "temoin.relief.segment_image", lambda *args, **kwargs: _sam_segmentation((full,))
    )
    mask, source = object_mask("data:image/jpeg;base64,x", small, "object")
    assert source == "sam"
    assert bool(mask.any())

    def _offline(*args, **kwargs):
        """Fail as an unreachable Meta API would."""
        raise MetaAPIError("offline")

    monkeypatch.setattr("temoin.relief.segment_image", _offline)
    _, fallback_source = object_mask("data:image/jpeg;base64,x", small, "object")
    assert fallback_source == "threshold"
    monkeypatch.setattr(
        "temoin.relief.segment_image", lambda *args, **kwargs: _sam_segmentation(())
    )
    _, empty_source = object_mask("data:image/jpeg;base64,x", small, "object")
    assert empty_source == "threshold"


def _relief_answer(defect_id: str) -> str:
    """Build a scripted relief reading linking one defect."""
    return json.dumps(
        {
            "summary": "Dôme central lisse.",
            "features": [
                {
                    "id": "R1",
                    "kind": "bump",
                    "description": "Dôme central.",
                    "zone": "centre",
                    "defect_id": defect_id,
                }
            ],
            "physical_presence": "real",
            "note": "",
            "citations": [defect_id],
        }
    )


def test_describe_relief_links_known_defects(sale_report: Report, test_settings: Settings) -> None:
    """A valid reading with a real defect id validates and caches."""
    fake = FakeChatClient([_relief_answer("D1")])
    description = describe_relief(
        relief_data_url="data:image/jpeg;base64,r",
        normals_data_url="data:image/jpeg;base64,n",
        defects=sale_report.defects,
        flat=False,
        photo_sha256s=["a" * 64],
        uncalibrated=False,
        settings=test_settings,
        client=fake,
    )
    assert description.summary == "Dôme central lisse."
    assert description.features[0].defect_id == "D1"
    sent = fake.calls[0]["messages"][1]["content"]
    assert sum(1 for part in sent if part.get("type") == "image_url") == 2


def test_describe_relief_rejects_unknown_defect(
    sale_report: Report, test_settings: Settings
) -> None:
    """An invented defect id is sent back, then fails honestly."""
    fake = FakeChatClient([_relief_answer("DX"), _relief_answer("DX")])
    with pytest.raises(ModelError, match="pas compris"):
        describe_relief(
            relief_data_url="data:image/jpeg;base64,r",
            normals_data_url="data:image/jpeg;base64,n",
            defects=sale_report.defects,
            flat=False,
            photo_sha256s=["a" * 64],
            uncalibrated=False,
            settings=test_settings,
            client=fake,
        )


def test_load_stack_rejects_bad_counts(
    tmp_path: Path, photo_path: Path, test_settings: Settings
) -> None:
    """3 or 9 photos are refused; missing files keep the image contract."""
    with pytest.raises(ReliefError, match="4 à 8"):
        load_stack([str(photo_path)] * 3, settings=test_settings)
    with pytest.raises(ReliefError, match="4 à 8"):
        load_stack([str(photo_path)] * 9, settings=test_settings)
    with pytest.raises(ImageError, match="introuvable"):
        load_stack([str(tmp_path / "nope.jpg")] * 4, settings=test_settings)


def test_views_paint_relief_and_normals(test_settings: Settings) -> None:
    """Unknown pixels stay dark; dome top differs from its rim."""
    images, _, disk, _ = _sphere()
    maps = compute_relief(
        images, lights=None, mask=disk, mask_source="threshold", settings=test_settings
    )
    relief_view = colorize_relief(maps.height)
    normals_view = normals_image(maps.normals, maps.valid)
    assert relief_view.size == images[0].size and relief_view.mode == "RGB"
    assert normals_view.size == images[0].size
    assert relief_view.getpixel((0, 0)) == (40, 40, 40)
    assert normals_view.getpixel((0, 0)) == (0, 0, 0)
    assert relief_view.getpixel((48, 48)) != relief_view.getpixel((48, 80))


def test_run_relief_end_to_end_offline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sale_report: Report,
    test_settings: Settings,
) -> None:
    """Sphere PNGs give a non-flat sealed-ready report with a reading."""
    images, _, _, _ = _sphere()
    paths = _save_stack(images, tmp_path)

    def _offline(*args, **kwargs):
        """Fail as an unreachable Meta API would."""
        raise MetaAPIError("offline")

    monkeypatch.setattr("temoin.relief.segment_image", _offline)
    fake = FakeChatClient([_relief_answer("D1")])
    analysis = run_relief(paths, settings=test_settings, defects=sale_report.defects, client=fake)
    report = analysis.report
    assert report.method == "raking-calibrated"
    assert report.mask_source == "threshold"
    assert len(report.photo_sha256s) == 4
    assert not report.flat
    assert report.alert is None
    assert report.compute_seconds >= 0
    assert report.description is not None
    assert report.description.features[0].defect_id == "D1"
    assert analysis.matrix.method == "raking"
    assert analysis.relief_image.size == analysis.first.size


def test_run_relief_flags_flat_sheet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sale_report: Report,
    test_settings: Settings,
) -> None:
    """Flat PNGs give the sealed flat alert with a suspect reading."""
    images, _, _ = _flat_sheet()
    paths = _save_stack(images, tmp_path)

    def _offline(*args, **kwargs):
        """Fail as an unreachable Meta API would."""
        raise MetaAPIError("offline")

    monkeypatch.setattr("temoin.relief.segment_image", _offline)
    payload = json.loads(_relief_answer("D1"))
    payload["physical_presence"] = "flat_suspect"
    fake = FakeChatClient([json.dumps(payload)])
    analysis = run_relief(paths, settings=test_settings, defects=sale_report.defects, client=fake)
    assert analysis.report.flat
    assert analysis.report.alert == FLAT_ALERT
    assert analysis.report.description is not None
    assert analysis.report.description.physical_presence == "flat_suspect"


def _canned_description() -> ReliefDescription:
    """Build a fixed model reading for CLI tests."""
    return ReliefDescription.model_validate(json.loads(_relief_answer("D1")))


def test_relief_command_measures_and_saves(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    test_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """relief prints CPU time and saves maps, report, GIF (offline)."""
    images, _, _, _ = _sphere()
    paths = _save_stack(images, tmp_path)
    monkeypatch.setattr("temoin.__main__.load_settings", lambda *args: test_settings)

    def _offline(*args, **kwargs):
        """Fail as an unreachable Meta API would."""
        raise MetaAPIError("offline")

    monkeypatch.setattr("temoin.relief.segment_image", _offline)
    monkeypatch.setattr("temoin.relief.describe_relief", lambda **kwargs: _canned_description())
    relief_png, normals_png = tmp_path / "relief.png", tmp_path / "normals.png"
    report_json, wiggle = tmp_path / "report.json", tmp_path / "relief.gif"
    assert (
        main(
            [
                "relief",
                *paths,
                "--relief",
                str(relief_png),
                "--normals",
                str(normals_png),
                "--json",
                str(report_json),
                "--wiggle",
                str(wiggle),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Relief mesuré" in out
    assert "Calcul CPU" in out
    assert "seuillage simple" in out
    assert "RELIEF PLAT" not in out
    assert relief_png.is_file() and normals_png.is_file()
    assert wiggle.is_file()
    saved = ReliefReport.model_validate_json(report_json.read_text(encoding="utf-8"))
    assert saved.method == "raking-calibrated" and not saved.flat


def test_relief_command_flags_flat_sheet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    test_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """relief on a flat sheet prints the sealed flat alert (offline)."""
    images, _, _ = _flat_sheet()
    paths = _save_stack(images, tmp_path)
    monkeypatch.setattr("temoin.__main__.load_settings", lambda *args: test_settings)

    def _offline(*args, **kwargs):
        """Fail as an unreachable Meta API would."""
        raise MetaAPIError("offline")

    monkeypatch.setattr("temoin.relief.segment_image", _offline)
    monkeypatch.setattr("temoin.relief.describe_relief", lambda **kwargs: _canned_description())
    assert main(["relief", *paths]) == 0
    assert FLAT_ALERT in capsys.readouterr().out


def test_relief_command_rejects_missing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    test_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Missing raking-light photos keep the image error contract."""
    monkeypatch.setattr("temoin.__main__.load_settings", lambda *args: test_settings)
    missing = [str(tmp_path / f"nope-{index}.jpg") for index in range(4)]
    assert main(["relief", *missing]) == 1
    assert "introuvable" in capsys.readouterr().err


def test_relief_report_sealed_in_constat(
    sale_report: Report, delivery_report: Report, comparison: Comparison
) -> None:
    """A relief inside a constat is covered by the seal; tampering breaks it."""
    relief = ReliefReport(
        method="raking-calibrated",
        photo_sha256s=["a" * 64] * 4,
        mask_source="threshold",
        coverage=0.85,
        height_std=0.06,
        normal_spread=0.27,
        flat=False,
        alert=None,
        description=None,
        compute_seconds=0.07,
        created_at_utc=datetime.now(UTC),
        prompt_version="1",
    )
    enriched = sale_report.model_copy(update={"relief": relief})
    certificate = seal_comparison(
        comparison,
        certificate_id="c9",
        previous_hash=None,
        sale=enriched,
        delivery=delivery_report,
    )
    assert verify_certificate(certificate)
    tampered = certificate.model_copy(deep=True)
    assert tampered.sale_constat.relief is not None
    tampered.sale_constat.relief.flat = True
    assert not verify_certificate(tampered)
