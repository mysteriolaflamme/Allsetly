"""Monocular AI depth: dense indicative relief from a single view (CPU).

MiDaS-small ONNX runs on CPU through the OpenCV DNN module (no new
dependency, no GPU). The relief is estimated, never measured: callers must
label it as indicative (DepthMatrix.method == "mono").
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from temoin.config import Settings, load_settings
from temoin.stereo import DepthMatrix, normalize_relief

logger = logging.getLogger(__name__)

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)
_PAD_GRAY = 114


class MonoDepthError(Exception):
    """Raised when AI relief cannot be produced, with a plain-language message."""


def _sha256_ok(path: Path, expected: str) -> bool:
    """Check a file fingerprint in streaming blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected.lower()


def model_path(settings: Settings | None = None) -> Path:
    """Return the local MiDaS weights, downloading them once when missing."""
    active = settings or load_settings()
    target = Path(active.data_dir) / active.monodepth_model_file
    if target.is_file() and _sha256_ok(target, active.monodepth_model_sha256):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading monocular depth model (%s).", active.monodepth_model_url)
    try:
        urllib.request.urlretrieve(active.monodepth_model_url, target)
    except OSError as error:
        raise MonoDepthError(
            "Modèle IA introuvable et téléchargement impossible : "
            "vérifiez la connexion, puis réessayez."
        ) from error
    if not _sha256_ok(target, active.monodepth_model_sha256):
        target.unlink(missing_ok=True)
        raise MonoDepthError("Le modèle IA téléchargé est corrompu : réessayez.")
    return target


def _letterbox(image: Image.Image, size: int) -> tuple[np.ndarray, tuple[float, ...]]:
    """Fit an image on a square canvas, return canvas + content box fractions."""
    width, height = image.size
    scale = size / max(width, height)
    fitted = image.resize(
        (max(round(width * scale), 1), max(round(height * scale), 1)), Image.BILINEAR
    )
    canvas = Image.new("RGB", (size, size), (_PAD_GRAY, _PAD_GRAY, _PAD_GRAY))
    left = (size - fitted.width) // 2
    top = (size - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    box = (left / size, top / size, fitted.width / size, fitted.height / size)
    return np.asarray(canvas), box


def estimate_depth(
    image: Image.Image, *, settings: Settings | None = None, net: cv2.dnn.Net | None = None
) -> DepthMatrix:
    """Estimate dense relative relief (0 far .. 1 near) from a single view."""
    active = settings or load_settings()
    network = net if net is not None else cv2.dnn.readNetFromONNX(str(model_path(active)))
    width, height = image.size
    canvas, box = _letterbox(image.convert("RGB"), active.monodepth_input_size)
    rgb = canvas.astype(np.float32) / 255.0
    rgb -= np.array(_MEAN, dtype=np.float32)
    rgb /= np.array(_STD, dtype=np.float32)
    blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None])
    try:
        network.setInput(blob)
        raw = network.forward()
    except Exception as error:
        raise MonoDepthError("Estimation IA impossible sur cette image.") from error
    depth = np.asarray(raw, dtype=np.float32).reshape(raw.shape[-2:])
    out_height, out_width = depth.shape
    left, top, width_frac, height_frac = box
    content = depth[
        round(top * out_height) : round((top + height_frac) * out_height),
        round(left * out_width) : round((left + width_frac) * out_width),
    ]
    target_width = min(active.stereo_max_width, width)
    target_height = max(round(height * target_width / width), 1)
    resized = cv2.resize(content, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    relief = normalize_relief(resized, np.ones_like(resized, dtype=bool))
    rows = tuple(tuple(round(float(value), 3) for value in row) for row in relief)
    logger.info("Mono depth: %dx%d indicative relief.", target_width, target_height)
    return DepthMatrix(
        width=target_width,
        height=target_height,
        values=rows,
        coverage=1.0,
        left_index=-1,
        right_index=-1,
        method="mono",
    )
