"""Photo forensics: retouch signals and quality gate (CPU only, no model)."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from temoin.config import Settings, load_settings
from temoin.images import ImageError

logger = logging.getLogger(__name__)

_EDITOR_SOFTWARE = ("photoshop", "gimp", "paint.net", "photopea", "pixlr")
_DEVELOP_SOFTWARE = ("lightroom", "snapseed", "darktable", "rawtherapee", "capture one")
_EXIF_SOFTWARE_TAG = 271
_EXIF_DATETIME_TAG = 306
_EXIF_ORIGINAL_TAG = 36867
_BRIGHTNESS_LO = 25.0
_BRIGHTNESS_HI = 230.0
_MIN_SIDE = 480


@dataclass(frozen=True)
class ElaStats:
    """Error-level statistics: high values suggest heterogeneous edits."""

    mean: float
    p99: float


@dataclass(frozen=True)
class NoiseStats:
    """Per-block noise stats: natural photos vary, so this never decides alone."""

    median: float
    suspicious_blocks: int
    total_blocks: int


@dataclass(frozen=True)
class ExifNotes:
    """Metadata notes: editor software, development app, date coherence."""

    software: str | None
    editor_found: bool
    developed: bool
    datetime_mismatch: bool


@dataclass(frozen=True)
class QualityReport:
    """Capture quality with coaching advice when something is fixable."""

    sharpness: float
    width: int
    height: int
    brightness: float
    ok: bool
    advice: tuple[str, ...]


@dataclass(frozen=True)
class ForensicsReport:
    """Combined retouch signals and quality gate for one photo."""

    suspicious: bool
    signals: tuple[str, ...]
    quality: QualityReport
    ela_mean: float
    ela_p99: float
    noise_median: float
    noise_suspicious_blocks: int
    noise_total_blocks: int
    software: str | None


def _ela_error(image: Image.Image, quality: int) -> np.ndarray:
    """Return the per-pixel recompression error (0-255 grayscale)."""
    rgb = image.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as recomp:
        recomp.load()
        diff = np.asarray(recomp.convert("RGB")).astype(np.int16)
        diff -= np.asarray(rgb).astype(np.int16)
    return np.abs(diff).mean(axis=2)


def ela_stats(image: Image.Image, *, quality: int = 90) -> ElaStats:
    """Summarize error-level analysis: mean and 99th percentile."""
    error = _ela_error(image, quality)
    return ElaStats(mean=float(error.mean()), p99=float(np.percentile(error, 99)))


def ela_map(image: Image.Image, *, quality: int = 90, scale: float = 12.0) -> Image.Image:
    """Return the amplified error-level map (bright areas look edited)."""
    amplified = np.clip(_ela_error(image, quality) * scale, 0, 255).astype(np.uint8)
    return Image.fromarray(amplified, mode="L")


def noise_stats(image: Image.Image, *, grid: int = 8) -> NoiseStats:
    """Count blocks whose noise deviates widely from the median block."""
    gray = np.asarray(image.convert("L")).astype(np.float64)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    height, width = laplacian.shape
    rows = max(height // grid, 1)
    cols = max(width // grid, 1)
    variances = [
        float(laplacian[y * rows : (y + 1) * rows, x * cols : (x + 1) * cols].var())
        for y in range(grid)
        for x in range(grid)
    ]
    median = float(np.median(variances)) if variances else 0.0
    if median < 1e-3:
        return NoiseStats(median=median, suspicious_blocks=0, total_blocks=len(variances))
    floor = max(median, 1e-6)
    suspicious = sum(1 for value in variances if value > 4 * floor or value < floor / 4)
    return NoiseStats(median=median, suspicious_blocks=suspicious, total_blocks=len(variances))


def exif_notes(image: Image.Image) -> ExifNotes:
    """Read software and date tags, tolerating missing metadata."""
    try:
        tags = image.getexif()
    except Exception:
        return ExifNotes(
            software=None, editor_found=False, developed=False, datetime_mismatch=False
        )
    raw_software = tags.get(_EXIF_SOFTWARE_TAG)
    name = str(raw_software).strip() if raw_software else ""
    lowered = name.lower()
    taken = tags.get(_EXIF_ORIGINAL_TAG)
    modified = tags.get(_EXIF_DATETIME_TAG)
    mismatch = bool(taken and modified and str(taken) != str(modified))
    return ExifNotes(
        software=name or None,
        editor_found=any(tag in lowered for tag in _EDITOR_SOFTWARE),
        developed=any(tag in lowered for tag in _DEVELOP_SOFTWARE),
        datetime_mismatch=mismatch,
    )


def quality_report(image: Image.Image, *, blur_min: float) -> QualityReport:
    """Score sharpness, brightness and resolution with French coaching advice."""
    gray = np.asarray(image.convert("L"))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    width, height = image.size
    advice: list[str] = []
    if sharpness < blur_min:
        advice.append(
            "Photo floue : stabilisez le téléphone et touchez l'objet pour la mise au point."
        )
    if brightness < _BRIGHTNESS_LO:
        advice.append("Photo trop sombre : ajoutez de la lumière.")
    if brightness > _BRIGHTNESS_HI:
        advice.append("Photo surexposée : évitez le contre-jour.")
    if min(width, height) < _MIN_SIDE:
        advice.append("Photo trop petite : rapprochez-vous de l'objet.")
    return QualityReport(
        sharpness=sharpness,
        width=width,
        height=height,
        brightness=brightness,
        ok=not advice,
        advice=tuple(advice),
    )


def analyze_photo(photo: str | Path, *, settings: Settings | None = None) -> ForensicsReport:
    """Combine retouch signals and the quality gate for one photo."""
    active = settings or load_settings()
    path = Path(photo)
    if not path.is_file():
        raise ImageError(f"Photo introuvable : « {path} ». Vérifiez le nom du fichier.")
    try:
        with Image.open(path) as handle:
            handle.load()
            image = handle.convert("RGB")
    except (OSError, ValueError) as error:
        raise ImageError(
            "Impossible de lire cette photo. Essayez un fichier JPG ou PNG."
        ) from error
    ela = ela_stats(image, quality=active.forensics_ela_quality)
    noise = noise_stats(image)
    notes = exif_notes(image)
    quality = quality_report(image, blur_min=active.forensics_blur_min)
    signals: list[str] = []
    noise_ratio = noise.suspicious_blocks / max(noise.total_blocks, 1)
    noise_hit = noise_ratio > active.forensics_noise_ratio
    if noise_hit:
        signals.append(
            f"Bruit hétérogène ({noise.suspicious_blocks}/{noise.total_blocks} zones) : "
            "montage possible."
        )
    if notes.editor_found:
        signals.append(f"Logiciel de retouche détecté : {notes.software}.")
    if notes.datetime_mismatch:
        signals.append("Dates de prise de vue incohérentes.")
    if notes.developed:
        signals.append(f"Photo développée ({notes.software}) : améliorations possibles.")
    suspicious = noise_hit or notes.editor_found
    logger.info(
        "Forensics: suspicious=%s signals=%d quality_ok=%s.", suspicious, len(signals), quality.ok
    )
    return ForensicsReport(
        suspicious=suspicious,
        signals=tuple(signals),
        quality=quality,
        ela_mean=ela.mean,
        ela_p99=ela.p99,
        noise_median=noise.median,
        noise_suspicious_blocks=noise.suspicious_blocks,
        noise_total_blocks=noise.total_blocks,
        software=notes.software,
    )
