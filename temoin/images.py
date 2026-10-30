"""Photo loading, validation, resizing, encoding and fingerprinting."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from temoin.schemas import Defect

SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif")


class ImageError(Exception):
    """Raised when a photo cannot be used, with a plain-language message."""


@dataclass(frozen=True)
class PreparedImage:
    """Photo ready to send to the model."""

    data_url: str
    sha256: str
    width: int
    height: int


def compute_sha256(data: bytes) -> str:
    """Return the hex SHA-256 fingerprint of raw bytes."""
    return sha256(data).hexdigest()


def _ensure_heic_support() -> None:
    """Register the HEIC plugin, or raise a plain-language error."""
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError as error:
        raise ImageError(
            "Cette photo est au format HEIC et sa lecture n'est pas installée. "
            "Convertissez-la en JPEG sur votre téléphone."
        ) from error


def prepare_image(
    path: str | Path,
    *,
    max_pixels: int,
    max_megabytes: float,
    jpeg_quality: int,
) -> PreparedImage:
    """Validate a photo, straighten it, shrink it and encode it for the model."""
    source = Path(path)
    if not source.is_file():
        raise ImageError(f"Photo introuvable : « {source} ». Vérifiez le nom du fichier.")
    raw = source.read_bytes()
    if len(raw) > int(max_megabytes * 1024 * 1024):
        raise ImageError(
            f"La photo dépasse {max_megabytes:.0f} Mo. Choisissez une photo plus légère."
        )
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ImageError("Format de photo non accepté. Utilisez une photo JPEG, PNG, WebP ou HEIC.")
    if suffix in (".heic", ".heif"):
        _ensure_heic_support()
    try:
        with Image.open(io.BytesIO(raw)) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ImageError(
            "Impossible de lire cette photo. Essayez une autre photo JPEG, PNG ou WebP."
        ) from error
    image.thumbnail((max_pixels, max_pixels), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    width, height = image.size
    return PreparedImage(
        data_url=f"data:image/jpeg;base64,{encoded}",
        sha256=compute_sha256(raw),
        width=width,
        height=height,
    )


_SEVERITY_COLORS = {"minor": "yellow", "moderate": "orange", "major": "red"}


def draw_defects(photo: str | Path, defects: list[Defect]) -> Image.Image:
    """Draw defect boxes with ids on a copy of the photo."""
    with Image.open(photo) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    pen = max(2, width // 300)
    for defect in defects:
        if defect.bbox is None:
            continue
        x1, y1, x2, y2 = (
            round(value * dim / 1000)
            for value, dim in zip(defect.bbox, (width, height, width, height), strict=True)
        )
        color = _SEVERITY_COLORS.get(defect.severity, "red")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=pen)
        draw.text((x1 + pen, y1 + pen), defect.id, fill=color)
    return image


def decode_data_url(data_url: str) -> Image.Image:
    """Decode a data:image/...;base64,... URL back to a PIL image."""
    try:
        header, encoded = data_url.split(",", 1)
        if not header.startswith("data:image/") or ";base64" not in header:
            raise ValueError("not an image data URL")
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as handle:
            return handle.convert("RGB")
    except (ValueError, OSError) as error:
        raise ImageError("Cette image est illisible.") from error
