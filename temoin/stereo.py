"""Depth Matrix: CPU-only relief sensing from two views (classical stereo).

No neural network, no GPU: two video frames with a small baseline give a
relative depth matrix through calibrated classical geometry (ORB matches,
essential matrix, rectification, semi-global matching). Unmeasurable pixels
are marked -1 rather than invented.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from temoin.config import Settings, load_settings

logger = logging.getLogger(__name__)


class StereoError(Exception):
    """Raised when relief cannot be measured, with a plain-language message."""


@dataclass(frozen=True)
class DepthMatrix:
    """Relative relief (0 far .. 1 near, -1 unknown) with its provenance."""

    width: int
    height: int
    values: tuple[tuple[float, ...], ...]
    coverage: float
    left_index: int
    right_index: int


@dataclass(frozen=True)
class ReliefStats:
    """Relief contrast between a box and its surrounding ring."""

    inside: float
    ring: float
    delta: float


def _orb_matches(
    left_gray: np.ndarray, right_gray: np.ndarray, *, max_features: int
) -> tuple[np.ndarray, np.ndarray, list]:
    """Match ORB keypoints with a Lowe ratio test."""
    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints_left, desc_left = orb.detectAndCompute(left_gray, None)
    keypoints_right, desc_right = orb.detectAndCompute(right_gray, None)
    if desc_left is None or desc_right is None or len(desc_left) < 8 or len(desc_right) < 8:
        return np.empty((0, 2)), np.empty((0, 2)), []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_left, desc_right, k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    points_left = np.float32([keypoints_left[item.queryIdx].pt for item in good])
    points_right = np.float32([keypoints_right[item.trainIdx].pt for item in good])
    return points_left, points_right, good


def pick_stereo_pair(images: list[Image.Image], *, min_matches: int = 30) -> tuple[int, int]:
    """Pick the overlapping pair with the most matches and real parallax."""
    if len(images) < 2:
        raise StereoError("Il faut au moins deux vues pour mesurer le relief.")
    grays = [np.asarray(image.convert("L")) for image in images]
    best: tuple[int, int] | None = None
    best_count = 0
    for left in range(len(grays)):
        for right in range(left + 1, len(grays)):
            points_left, points_right, good = _orb_matches(
                grays[left], grays[right], max_features=2000
            )
            if len(good) < min_matches:
                continue
            motion = float(np.median(np.linalg.norm(points_right - points_left, axis=1)))
            if motion < 2.0 or len(good) <= best_count:
                continue
            best, best_count = (left, right), len(good)
    if best is None:
        raise StereoError("Les vues sont trop semblables ou trop floues pour mesurer le relief.")
    return best


def _rectify(
    left_gray: np.ndarray, right_gray: np.ndarray, points_left: np.ndarray, points_right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rectify a pair with an approximated focal length (f = 1.2 x width)."""
    height, width = left_gray.shape
    focal = 1.2 * max(width, height)
    camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    essential, mask = cv2.findEssentialMat(
        points_left, points_right, camera, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if essential is None or mask is None or int(mask.ravel().sum()) < 15:
        raise StereoError("Le mouvement entre les vues ne permet pas de mesurer le relief.")
    _, rotation, translation, _ = cv2.recoverPose(
        essential, points_left, points_right, camera, mask=mask
    )
    rect_left, rect_right, proj_left, proj_right, _, _, _ = cv2.stereoRectify(
        camera,
        None,
        camera,
        None,
        (width, height),
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    map_left = cv2.initUndistortRectifyMap(
        camera, None, rect_left, proj_left, (width, height), cv2.CV_32FC1
    )
    map_right = cv2.initUndistortRectifyMap(
        camera, None, rect_right, proj_right, (width, height), cv2.CV_32FC1
    )
    return (
        cv2.remap(left_gray, map_left[0], map_left[1], cv2.INTER_LINEAR),
        cv2.remap(right_gray, map_right[0], map_right[1], cv2.INTER_LINEAR),
    )


def compute_depth(
    left: Image.Image,
    right: Image.Image,
    *,
    settings: Settings | None = None,
    left_index: int = 0,
    right_index: int = 1,
) -> DepthMatrix:
    """Compute the relative relief matrix of a stereo pair (0 far .. 1 near)."""
    active = settings or load_settings()
    left_gray = np.asarray(left.convert("L"))
    right_gray = np.asarray(right.convert("L"))
    points_left, points_right, good = _orb_matches(
        left_gray, right_gray, max_features=active.stereo_orb_features
    )
    if len(good) < active.stereo_min_matches:
        raise StereoError("Pas assez de points communs entre les deux vues.")
    rect_left, rect_right = _rectify(left_gray, right_gray, points_left, points_right)
    block = active.stereo_block_size
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=active.stereo_num_disparities,
        blockSize=block,
        P1=8 * block * block,
        P2=32 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=active.stereo_uniqueness,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = matcher.compute(rect_left, rect_right).astype(np.float32) / 16.0
    valid = disparity > 0
    coverage = float(valid.mean())
    if coverage < 0.02:
        raise StereoError("Aucun relief mesurable : surface trop lisse ou vues trop proches.")
    floor, ceiling = float(disparity[valid].min()), float(disparity[valid].max())
    span = max(ceiling - floor, 1e-6)
    relief = np.full(disparity.shape, -1.0, dtype=np.float32)
    relief[valid] = (disparity[valid] - floor) / span
    height, width = relief.shape
    target_width = min(active.stereo_max_width, width)
    target_height = max(round(height * target_width / width), 1)
    small_values = cv2.resize(relief, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    small_mask = cv2.resize(
        valid.astype(np.float32), (target_width, target_height), interpolation=cv2.INTER_NEAREST
    )
    small_values = np.where(small_mask < 0.5, -1.0, np.clip(small_values, 0.0, 1.0))
    rows = tuple(tuple(round(float(value), 3) for value in row) for row in small_values)
    logger.info(
        "Depth matrix: coverage %.0f%%, pair (%d, %d).", 100 * coverage, left_index, right_index
    )
    return DepthMatrix(
        width=target_width,
        height=target_height,
        values=rows,
        coverage=round(coverage, 3),
        left_index=left_index,
        right_index=right_index,
    )


def defect_relief(matrix: DepthMatrix, bbox: list[int]) -> ReliefStats:
    """Contrast mean relief inside a 0-1000 box vs its surrounding ring."""
    x0 = min(max(round(bbox[0] / 1000 * matrix.width), 0), matrix.width)
    y0 = min(max(round(bbox[1] / 1000 * matrix.height), 0), matrix.height)
    x1 = min(max(round(bbox[2] / 1000 * matrix.width), 0), matrix.width)
    y1 = min(max(round(bbox[3] / 1000 * matrix.height), 0), matrix.height)
    if x1 <= x0 or y1 <= y0:
        raise StereoError("Relief illisible sur cette zone du défaut.")
    values = np.array(matrix.values, dtype=np.float32)
    inside = values[y0:y1, x0:x1]
    pad_x = max(round((x1 - x0) * 0.4), 1)
    pad_y = max(round((y1 - y0) * 0.4), 1)
    ring_mask = np.zeros_like(values, dtype=bool)
    ring_mask[
        max(y0 - pad_y, 0) : min(y1 + pad_y, matrix.height),
        max(x0 - pad_x, 0) : min(x1 + pad_x, matrix.width),
    ] = True
    ring_mask[y0:y1, x0:x1] = False
    inside_valid = inside[inside >= 0]
    ring_valid = values[ring_mask]
    ring_valid = ring_valid[ring_valid >= 0]
    if inside_valid.size == 0 or ring_valid.size == 0:
        raise StereoError("Relief illisible sur cette zone du défaut.")
    mean_inside = float(inside_valid.mean())
    mean_ring = float(ring_valid.mean())
    return ReliefStats(inside=mean_inside, ring=mean_ring, delta=mean_inside - mean_ring)


def relief_wiggle(
    image: Image.Image, matrix: DepthMatrix, *, amplitude: float = 12.0, frames: int = 4
) -> bytes:
    """Render an animated parallax GIF from a relief matrix."""
    base = np.asarray(image.convert("RGB").resize((matrix.width, matrix.height)))
    relief = np.array(matrix.values, dtype=np.float32)
    relief[relief < 0] = 0.0
    height, width = relief.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    shots: list[Image.Image] = []
    for step in range(frames):
        phase = 2 * np.pi * step / frames
        shift = np.sin(phase) * amplitude * relief
        warped = cv2.remap(
            base,
            grid_x + shift.astype(np.float32),
            grid_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        shots.append(Image.fromarray(warped))
    buffer = io.BytesIO()
    shots[0].save(
        buffer, format="GIF", save_all=True, append_images=shots[1:], duration=140, loop=0
    )
    return buffer.getvalue()


def relief_cloud(image: Image.Image, matrix: DepthMatrix, *, max_points: int = 6000) -> go.Figure:
    """Build an interactive 3D point cloud from measured relief only."""
    relief = np.array(matrix.values, dtype=np.float32)
    known = np.argwhere(relief >= 0)
    if known.size == 0:
        raise StereoError("Aucun relief mesuré à afficher en 3D.")
    stride = max(int(np.sqrt(len(known) / max_points)) + 1, 1)
    picked = known[::stride]
    colors = np.asarray(image.convert("RGB").resize((matrix.width, matrix.height)))
    figure = go.Figure(
        data=[
            go.Scatter3d(
                x=picked[:, 1].tolist(),
                y=(matrix.height - picked[:, 0]).tolist(),
                z=relief[picked[:, 0], picked[:, 1]].tolist(),
                mode="markers",
                marker={
                    "size": 2,
                    "color": [
                        f"rgb({red},{green},{blue})"
                        for red, green, blue in colors[picked[:, 0], picked[:, 1]].tolist()
                    ],
                },
            )
        ]
    )
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "aspectmode": "data",
        },
    )
    return figure


def load_pair(left: str | Path, right: str | Path) -> tuple[Image.Image, Image.Image]:
    """Load two view files as RGB images."""
    try:
        with Image.open(left) as first, Image.open(right) as second:
            first.load()
            second.load()
            return first.convert("RGB"), second.convert("RGB")
    except (OSError, ValueError) as error:
        raise StereoError("Impossible de lire ces deux vues.") from error
