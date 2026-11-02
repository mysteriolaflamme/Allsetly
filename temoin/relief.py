"""Raking-light relief: CPU photometric stereo from 4-8 lamp positions.

The phone stays still while a single lamp moves around the item (12h, 3h,
6h, 9h first, about 45 degrees high). Shading changes reveal the surface:
least-squares photometric stereo recovers normals + albedo, then a
Frankot-Chellappa FFT integration turns the normals into a relief map.

Pure NumPy on CPU: no GPU, no neural network. The object mask comes from
SAM 3.1 with a plain threshold fallback. A flat relief with uniform
normals means the "object" is probably a screen photo or a print.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]

from PIL import Image

from temoin.config import Settings, load_settings
from temoin.images import decode_data_url, prepare_image
from temoin.meta_api import MetaAPIError, Segmentation, segment_image
from temoin.model import ChatClient, complete_json, default_client
from temoin.prompts import RELIEF_PROMPT_VERSION, build_relief_messages
from temoin.schemas import Defect, ReliefDescription, ReliefReport
from temoin.stereo import DepthMatrix

logger = logging.getLogger(__name__)

CLOCK_POSITIONS = ("12h", "3h", "6h", "9h", "1h30", "4h30", "7h30", "10h30")
"""Photo order: cardinals first, then diagonals when more than 4 photos."""

_AZIMUTH_DEG = {
    "12h": -90.0,
    "3h": 0.0,
    "6h": 90.0,
    "9h": 180.0,
    "1h30": -45.0,
    "4h30": 45.0,
    "7h30": 135.0,
    "10h30": -135.0,
}
"""Lamp azimuth in degrees (x right, y down: 3h = 0, 6h = +90)."""

FLAT_ALERT = "RELIEF PLAT — possible photo d'écran ou impression"

_UNKNOWN_GRAY = 40
_SHADOW_LEVEL = 0.02
_MIN_ACTIVE_LIGHTS = 3
_MIN_COVERAGE = 0.02


class ReliefError(Exception):
    """Raised when relief cannot be measured, with a plain-language message."""


def _require_numpy() -> None:
    """Raise a plain-language error when NumPy is not installed."""
    if np is None:
        raise ReliefError("Le relief demande NumPy, qui n'est pas installé ici.")


@dataclass(frozen=True)
class ReliefMaps:
    """Measured relief arrays with their quality metrics (CPU, NumPy)."""

    normals: np.ndarray
    albedo: np.ndarray
    height: np.ndarray
    mask: np.ndarray
    valid: np.ndarray
    mask_source: Literal["sam", "threshold"]
    coverage: float
    height_std: float
    normal_spread: float
    flat: bool
    compute_seconds: float


@dataclass(frozen=True)
class ReliefAnalysis:
    """Full relief result: maps, viewers and the seal-ready report."""

    first: Image.Image
    matrix: DepthMatrix
    relief_image: Image.Image
    normals_image: Image.Image
    report: ReliefReport


def light_directions(count: int, *, elevation_deg: float = 45.0) -> np.ndarray:
    """Return count unit light vectors (x right, y down, z toward viewer)."""
    _require_numpy()
    if count < _MIN_ACTIVE_LIGHTS or count > len(CLOCK_POSITIONS):
        raise ReliefError(f"Il faut {3} à {len(CLOCK_POSITIONS)} positions de lampe.")
    elevation = math.radians(elevation_deg)
    vertical = math.sin(elevation)
    horizontal = math.cos(elevation)
    rows = []
    for position in CLOCK_POSITIONS[:count]:
        azimuth = math.radians(_AZIMUTH_DEG[position])
        rows.append(
            [
                horizontal * math.cos(azimuth),
                horizontal * math.sin(azimuth),
                vertical,
            ]
        )
    return np.array(rows, dtype=np.float64)


def load_stack(
    paths: list[str | Path], *, settings: Settings | None = None
) -> tuple[list[Image.Image], list[str], str]:
    """Load 4-8 raking-light photos as same-size RGB images, SHAs + first URL."""
    active = settings or load_settings()
    count = len(paths)
    if count < active.relief_min_photos or count > active.relief_max_photos:
        raise ReliefError(
            f"Déposez {active.relief_min_photos} à {active.relief_max_photos} photos "
            "du même objet (lampe à 12 h, 3 h, 6 h puis 9 h)."
        )
    prepared = [
        prepare_image(
            path,
            max_pixels=active.image_max_pixels,
            max_megabytes=active.image_max_megabytes,
            jpeg_quality=active.image_jpeg_quality,
        )
        for path in paths
    ]
    images = [decode_data_url(item.data_url) for item in prepared]
    width = min(image.width for image in images)
    height = min(image.height for image in images)
    target_width = min(active.relief_max_width, width)
    target_height = max(round(height * target_width / width), 1)
    resized = [image.resize((target_width, target_height), Image.BILINEAR) for image in images]
    return resized, [item.sha256 for item in prepared], prepared[0].data_url


def _sam_union_mask(segmentation: Segmentation, size: tuple[int, int]) -> np.ndarray | None:
    """Merge SAM masks into a boolean array, or None when nothing was found."""
    if not segmentation.masks:
        return None
    full = np.zeros((segmentation.image_height, segmentation.image_width), dtype=bool)
    for mask in segmentation.masks:
        left = min(max(round(mask.left), 0), full.shape[1])
        top = min(max(round(mask.top), 0), full.shape[0])
        right = min(max(round(mask.right), 0), full.shape[1])
        bottom = min(max(round(mask.bottom), 0), full.shape[0])
        if right <= left or bottom <= top:
            continue
        raster = np.frombuffer(mask.pixels, dtype=np.uint8).reshape(mask.height, mask.width)
        cell = Image.fromarray(raster * 255).resize((right - left, bottom - top), Image.NEAREST)
        full[top:bottom, left:right] |= np.asarray(cell) > 0
    if not full.any():
        return None
    return np.asarray(Image.fromarray(full).resize(size, Image.NEAREST), dtype=bool)


def otsu_threshold(gray: np.ndarray) -> float:
    """Split a 0..1 grayscale image with Otsu's method (pure NumPy)."""
    _require_numpy()
    hist = np.histogram(gray, bins=256, range=(0.0, 1.0))[0].astype(np.float64)
    centers = (np.arange(256) + 0.5) / 256.0
    weight_back = np.cumsum(hist)
    weight_fore = gray.size - weight_back
    cum_mean = np.cumsum(hist * centers)
    mean_back = cum_mean / np.maximum(weight_back, 1.0)
    mean_fore = (cum_mean[-1] - cum_mean) / np.maximum(weight_fore, 1.0)
    between = weight_back * weight_fore * (mean_back - mean_fore) ** 2
    between[(weight_back == 0) | (weight_fore == 0)] = -1.0
    return float(centers[int(np.argmax(between))])


def threshold_mask(image: Image.Image) -> np.ndarray:
    """Split object from background: Otsu, the background touches the frame."""
    _require_numpy()
    gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    level = otsu_threshold(gray)
    dark, bright = gray < level, gray >= level
    if not dark.any():
        return bright
    if not bright.any():
        return dark

    def _touches_frame(side: np.ndarray) -> bool:
        """True when one side reaches the image frame."""
        return bool(side[0, :].any() or side[-1, :].any() or side[:, 0].any() or side[:, -1].any())

    if _touches_frame(dark) != _touches_frame(bright):
        return bright if _touches_frame(dark) else dark
    height, width = gray.shape
    center = np.array([(height - 1) / 2.0, (width - 1) / 2.0])

    def _off_center(side: np.ndarray) -> float:
        """Mean distance of one side to the image center."""
        return float(np.linalg.norm(np.argwhere(side).mean(axis=0) - center))

    return dark if _off_center(dark) <= _off_center(bright) else bright


def object_mask(
    first_data_url: str,
    first_small: Image.Image,
    concept: str,
    *,
    settings: Settings | None = None,
) -> tuple[np.ndarray, Literal["sam", "threshold"]]:
    """Mask the object with SAM 3.1, falling back to a plain threshold."""
    _require_numpy()
    active = settings or load_settings()
    try:
        segmentation = segment_image(first_data_url, concept.strip() or "object", settings=active)
    except MetaAPIError as error:
        logger.warning("SAM unavailable for relief mask, threshold fallback: %s", error)
        return threshold_mask(first_small), "threshold"
    merged = _sam_union_mask(segmentation, first_small.size)
    if merged is None:
        logger.warning("SAM found no object, threshold fallback.")
        return threshold_mask(first_small), "threshold"
    return merged, "sam"


def _to_grays(images: list[Image.Image]) -> np.ndarray:
    """Stack RGB images as 0..1 grayscale (n, H, W)."""
    return np.stack([np.asarray(image.convert("L"), dtype=np.float64) / 255.0 for image in images])


def photometric_stereo(
    grays: np.ndarray, lights: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares normals + albedo with shadow-aware weights (vectorized)."""
    _require_numpy()
    count, height, width = grays.shape
    if lights.shape != (count, _MIN_ACTIVE_LIGHTS):
        raise ReliefError("Les photos et les positions de lampe ne correspondent pas.")
    flat = grays.reshape(count, -1)
    weights = (flat > _SHADOW_LEVEL).astype(np.float64)
    usable = mask.reshape(-1) & (weights.sum(axis=0) >= _MIN_ACTIVE_LIGHTS)
    normals = np.zeros((height * width, 3))
    normals[:, 2] = 1.0
    albedo = np.zeros(height * width)
    valid = np.zeros(height * width, dtype=bool)
    if not usable.any():
        raise ReliefError(
            "Trop d'ombres sur l'objet : rapprochez la lampe et évitez le contre-jour."
        )
    bits = 2 ** np.arange(count)
    codes = (weights * bits[:, None]).sum(axis=0).astype(int)
    # One exact least-squares fit per active-light pattern (shadow-aware).
    for code in np.unique(codes[usable]):
        members = usable & (codes == code)
        active = np.array([bool(code & bit) for bit in bits])
        rows = lights[active]
        if rows.shape[0] < _MIN_ACTIVE_LIGHTS:
            continue
        solved = (np.linalg.pinv(rows) @ flat[active][:, members]).T
        lengths = np.linalg.norm(solved, axis=1)
        lit = lengths > 1e-6
        unit = np.zeros_like(solved)
        unit[lit] = solved[lit] / lengths[lit, None]
        albedo[members] = lengths
        normals[members] = unit
        valid[members] = lit & (unit[:, 2] > 0)
    return (
        normals.reshape(height, width, 3),
        albedo.reshape(height, width),
        valid.reshape(height, width),
    )


def estimate_lights_hayakawa(grays: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Estimate light directions from shading alone (Hayakawa SVD + GBR fix).

    Returns (n, 3) unit vectors, up to a bas-relief ambiguity: fine for an
    imprint, not for metrology. Raises ReliefError when rank < 3.
    """
    _require_numpy()
    count = grays.shape[0]
    rows = grays.reshape(count, -1).T[mask.reshape(-1)]
    # Shadows break the rank-3 model: keep pixels lit by every lamp.
    rows = rows[rows.min(axis=1) > _SHADOW_LEVEL]
    if rows.shape[0] < 10:
        raise ReliefError("Pas assez de pixels éclairés pour estimer les lumières.")
    factors, singular, loadings = np.linalg.svd(rows, full_matrices=False)
    if singular[2] < 1e-9 * singular[0]:
        raise ReliefError("Les photos sont trop semblables : changez vraiment la lampe de côté.")
    root = np.sqrt(singular[:3])
    pseudo_shapes = factors[:, :3] * root
    pseudo_lights = (loadings[:3, :] * root[:, None]).T
    # Unit-norm constraint b G b' = 1: least squares on the 6 Gram unknowns.
    axis = pseudo_shapes
    design = np.stack(
        [
            axis[:, 0] ** 2,
            axis[:, 1] ** 2,
            axis[:, 2] ** 2,
            2 * axis[:, 0] * axis[:, 1],
            2 * axis[:, 0] * axis[:, 2],
            2 * axis[:, 1] * axis[:, 2],
        ],
        axis=1,
    )
    coeffs = np.linalg.lstsq(design, np.ones(rows.shape[0]), rcond=None)[0]
    gram = np.array(
        [
            [coeffs[0], coeffs[3], coeffs[4]],
            [coeffs[3], coeffs[1], coeffs[5]],
            [coeffs[4], coeffs[5], coeffs[2]],
        ]
    )
    try:
        root_gram = np.linalg.cholesky(gram)
    except np.linalg.LinAlgError as error:
        raise ReliefError(
            "Lumières inestimables sur ces photos : utilisez le mode calibré "
            "(lampe à 12 h, 3 h, 6 h puis 9 h)."
        ) from error
    lights = pseudo_lights @ np.linalg.inv(root_gram).T
    shapes = pseudo_shapes @ root_gram
    # The unit-norm fix leaves an orthogonal ambiguity: face the camera
    # (mean normal to +z), then undo the in-plane spin with the known
    # shooting order (exact lamp angles still come from the data).
    mean_shape = shapes.mean(axis=0)
    scale = float(np.linalg.norm(mean_shape))
    if scale < 1e-9:
        raise ReliefError(
            "Lumières inestimables sur ces photos : utilisez le mode calibré "
            "(lampe à 12 h, 3 h, 6 h puis 9 h)."
        )
    face = _rotation_to_view(mean_shape / scale)
    lights = lights @ face.T
    # A mirrored lamp ring explains shading equally well: the known photo
    # order (clockwise 12h, 3h, 6h, 9h) restores the true chirality.
    azimuths = np.arctan2(lights[:, 1], lights[:, 0])
    steps = np.diff(np.unwrap(azimuths))
    if steps.size and float(steps.mean()) < 0:
        lights[:, 0] = -lights[:, 0]
        azimuths = np.arctan2(lights[:, 1], lights[:, 0])
    expected = np.radians([_AZIMUTH_DEG[name] for name in CLOCK_POSITIONS[:count]])
    gaps = expected - azimuths
    sine_sum, cosine_sum = float(np.sin(gaps).sum()), float(np.cos(gaps).sum())
    offset = math.atan2(sine_sum, cosine_sum) if sine_sum or cosine_sum else 0.0
    cosine, sine = math.cos(offset), math.sin(offset)
    spin = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    lights = lights @ spin.T
    if lights[:, 2].mean() < 0:
        lights[:, 2] = -lights[:, 2]
    norms = np.linalg.norm(lights, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return lights / norms


def _rotation_to_view(direction: np.ndarray) -> np.ndarray:
    """Rotation taking a unit direction onto +z (Rodrigues' formula)."""
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(direction, target)
    sine = float(np.linalg.norm(axis))
    cosine = float(direction @ target)
    if sine < 1e-9:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    skew = (
        np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        / sine
    )
    return np.eye(3) + skew * sine + (skew @ skew) * (1.0 - cosine)


def frankot_chellappa(normals: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Integrate normals into a height map with the Frankot-Chellappa FFT."""
    _require_numpy()
    height, width = valid.shape
    slope_x = np.zeros((height, width))
    slope_y = np.zeros((height, width))
    depth = np.clip(normals[:, :, 2], 0.1, None)
    slope_x[valid] = -normals[:, :, 0][valid] / depth[valid]
    slope_y[valid] = -normals[:, :, 1][valid] / depth[valid]
    freq_x = 2 * math.pi * np.fft.fftfreq(width)
    freq_y = 2 * math.pi * np.fft.fftfreq(height)
    grid_x, grid_y = np.meshgrid(freq_x, freq_y)
    denom = grid_x**2 + grid_y**2
    denom[0, 0] = 1.0
    spectrum = (-1j * grid_x * np.fft.fft2(slope_x) - 1j * grid_y * np.fft.fft2(slope_y)) / denom
    spectrum[0, 0] = 0.0
    return np.real(np.fft.ifft2(spectrum)).astype(np.float32)


def normalize_height(height: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Scale valid heights to 0 far .. 1 near, saturating outliers (-1 out)."""
    _require_numpy()
    relief = np.full(height.shape, -1.0, dtype=np.float32)
    if not valid.any():
        return relief
    measured = height[valid]
    floor = float(np.percentile(measured, 1))
    ceiling = float(np.percentile(measured, 99))
    span = max(ceiling - floor, 1e-9)
    relief[valid] = np.clip((height[valid] - floor) / span, 0.0, 1.0)
    return relief


def normal_spread(normals: np.ndarray, valid: np.ndarray) -> float:
    """Mean deviation of valid normals from their mean direction (0 uniform)."""
    _require_numpy()
    picked = normals[valid]
    mean = picked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-9:
        return 1.0
    dots = np.clip(picked @ (mean / norm), -1.0, 1.0)
    return float((1.0 - dots).mean())


def compute_relief(
    images: list[Image.Image],
    *,
    lights: np.ndarray | None,
    mask: np.ndarray,
    mask_source: Literal["sam", "threshold"],
    settings: Settings | None = None,
) -> ReliefMaps:
    """Run the full CPU pipeline: stereo, integration, metrics (timed)."""
    _require_numpy()
    active = settings or load_settings()
    start = time.perf_counter()
    grays = _to_grays(images)
    active_lights = (
        lights
        if lights is not None
        else light_directions(len(images), elevation_deg=active.relief_light_elevation_deg)
    )
    normals, albedo, valid = photometric_stereo(grays, active_lights, mask)
    raw_height = frankot_chellappa(normals, valid)
    height = normalize_height(raw_height, valid)
    masked = int(mask.sum())
    coverage = float(valid.sum() / masked) if masked else 0.0
    if coverage < _MIN_COVERAGE:
        raise ReliefError("Relief illisible : l'objet est trop sombre ou le masque est vide.")
    height_std = float(raw_height[valid].std() / images[0].width)
    spread = normal_spread(normals, valid)
    flat = bool(
        height_std < active.relief_flat_height_std and spread < active.relief_flat_normal_spread
    )
    elapsed = time.perf_counter() - start
    logger.info(
        "Raking relief: coverage %.0f%%, height_std %.4f, spread %.4f, flat=%s, %.2fs CPU.",
        100 * coverage,
        height_std,
        spread,
        flat,
        elapsed,
    )
    return ReliefMaps(
        normals=normals,
        albedo=albedo,
        height=height,
        mask=mask,
        valid=valid,
        mask_source=mask_source,
        coverage=coverage,
        height_std=height_std,
        normal_spread=spread,
        flat=flat,
        compute_seconds=elapsed,
    )


_RELIEF_STOPS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.0, (38, 70, 160)),
    (0.35, (40, 150, 200)),
    (0.5, (120, 200, 150)),
    (0.65, (240, 220, 120)),
    (1.0, (200, 50, 40)),
)
"""Blue (hollows) to red (bumps) gradient stops for the relief map."""


def colorize_relief(height: np.ndarray) -> Image.Image:
    """Paint relief blue (hollows) to red (bumps), dark gray when unknown."""
    _require_numpy()
    rgb = np.full((*height.shape, 3), _UNKNOWN_GRAY, dtype=np.uint8)
    known = height >= 0
    if known.any():
        values = np.clip(height[known], 0.0, 1.0)
        positions = [stop for stop, _ in _RELIEF_STOPS]
        for channel in range(3):
            levels = [color[channel] for _, color in _RELIEF_STOPS]
            rgb[:, :, channel][known] = (
                np.interp(values, positions, levels).round().astype(np.uint8)
            )
    return Image.fromarray(rgb)


def normals_image(normals: np.ndarray, valid: np.ndarray) -> Image.Image:
    """Encode normals as RGB (x red, y green, z blue), black when unknown."""
    _require_numpy()
    rgb = np.zeros((*valid.shape, 3), dtype=np.uint8)
    rgb[valid] = (np.clip(normals[valid] * 0.5 + 0.5, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def encode_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a JPEG data URL for the model."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_relief_cache_key(
    *, photo_sha256s: list[str], uncalibrated: bool, defect_ids: list[str]
) -> str:
    """Derive a cache key from the photos, the mode and the relief version."""
    raw = (
        f"{'|'.join(photo_sha256s)}|{int(uncalibrated)}"
        f"|{','.join(defect_ids)}|{RELIEF_PROMPT_VERSION}"
    ).encode()
    return sha256(raw).hexdigest()


def _check_relief_citations(answer: ReliefDescription, defect_ids: set[str]) -> None:
    """Validate linked defect ids; ValueError feeds the model retry."""
    for feature in answer.features:
        if feature.defect_id is not None and feature.defect_id not in defect_ids:
            raise ValueError(f"unknown defect id {feature.defect_id!r}: use known ids or null")
    for citation in answer.citations:
        if citation not in defect_ids:
            raise ValueError(f"unknown citation {citation!r}: use known defect ids")


def describe_relief(
    *,
    relief_data_url: str,
    normals_data_url: str,
    defects: list[Defect],
    flat: bool,
    photo_sha256s: list[str],
    uncalibrated: bool,
    settings: Settings | None = None,
    client: ChatClient | None = None,
) -> ReliefDescription:
    """Ask Muse Spark Open to read hollows/bumps and link constat defects."""
    active = settings or load_settings()
    known = {defect.id for defect in defects}
    summaries = [{"id": defect.id, "type": defect.type, "zone": defect.zone} for defect in defects]
    active_client = client or default_client(
        api_key=active.baseten_api_key,
        base_url=active.model_base_url,
        timeout_seconds=active.model_timeout_seconds,
    )

    def _check(answer: ReliefDescription) -> None:
        """Check citations against the constat defects."""
        _check_relief_citations(answer, known)

    return complete_json(
        build_relief_messages(
            relief_data_url=relief_data_url,
            normals_data_url=normals_data_url,
            defects=summaries,
            flat=flat,
        ),
        ReliefDescription,
        client=active_client,
        model_name=active.model_name,
        cache_key=build_relief_cache_key(
            photo_sha256s=photo_sha256s,
            uncalibrated=uncalibrated,
            defect_ids=sorted(known),
        ),
        cache_dir=active.cache_dir,
        max_attempts=active.model_max_attempts,
        retry_wait_seconds=active.model_retry_wait_seconds,
        json_retries=active.model_json_retries,
        extra_check=_check,
    )


def run_relief(
    paths: list[str | Path],
    *,
    settings: Settings | None = None,
    concept: str = "object",
    uncalibrated: bool = False,
    defects: list[Defect] | None = None,
    client: ChatClient | None = None,
) -> ReliefAnalysis:
    """Measure, visualize and describe raking-light relief (CPU + 1 call)."""
    _require_numpy()
    active = settings or load_settings()
    images, shas, first_data_url = load_stack(paths, settings=active)
    mask, source = object_mask(first_data_url, images[0], concept, settings=active)
    lights = estimate_lights_hayakawa(_to_grays(images), mask) if uncalibrated else None
    maps = compute_relief(images, lights=lights, mask=mask, mask_source=source, settings=active)
    relief_view = colorize_relief(maps.height)
    normals_view = normals_image(maps.normals, maps.valid)
    description = describe_relief(
        relief_data_url=encode_data_url(relief_view),
        normals_data_url=encode_data_url(normals_view),
        defects=list(defects or []),
        flat=maps.flat,
        photo_sha256s=shas,
        uncalibrated=uncalibrated,
        settings=active,
        client=client,
    )
    rows = tuple(tuple(round(float(value), 3) for value in row) for row in maps.height.tolist())
    report = ReliefReport(
        method="raking-uncalibrated" if uncalibrated else "raking-calibrated",
        photo_sha256s=shas,
        mask_source=source,
        coverage=round(maps.coverage, 3),
        height_std=round(maps.height_std, 4),
        normal_spread=round(maps.normal_spread, 4),
        flat=maps.flat,
        alert=FLAT_ALERT if maps.flat else None,
        description=description,
        compute_seconds=round(maps.compute_seconds, 2),
        created_at_utc=datetime.now(UTC),
        prompt_version=RELIEF_PROMPT_VERSION,
    )
    matrix = DepthMatrix(
        width=maps.height.shape[1],
        height=maps.height.shape[0],
        values=rows,
        coverage=round(maps.coverage, 3),
        left_index=0,
        right_index=len(images) - 1,
        method="raking",
    )
    return ReliefAnalysis(
        first=images[0],
        matrix=matrix,
        relief_image=relief_view,
        normals_image=normals_view,
        report=report,
    )
