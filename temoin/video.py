"""Multi-view video constats: extract frames, analyse each, fuse defects."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
from PIL import Image

from temoin.config import Settings, load_settings
from temoin.images import compute_sha256
from temoin.model import ChatClient
from temoin.prompts import PROMPT_VERSION
from temoin.report import create_report
from temoin.schemas import (
    Alert,
    Checkpoint,
    Defect,
    FrameReport,
    FusedDefect,
    VideoChallenge,
    VideoReport,
)
from temoin.tools import pair_score

logger = logging.getLogger(__name__)

SUPPORTED_VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v", ".3gp")
MATCH_THRESHOLD = 3
_MAX_SEQUENTIAL_FRAMES = 1500
_SEVERITY_RANK = {"minor": 0, "moderate": 1, "major": 2}


class VideoError(Exception):
    """Raised when a video cannot be used, with a plain-language message."""


@dataclass(frozen=True)
class VideoFrame:
    """One sampled frame with its timestamp."""

    index: int
    timestamp_s: float
    image: Image.Image


def _bbox_iou(left: list[int], right: list[int]) -> float:
    """Return the intersection-over-union of two 0-1000 boxes."""
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    area = (x2 - x1) * (y2 - y1)
    union = (left[2] - left[0]) * (left[3] - left[1])
    union += (right[2] - right[0]) * (right[3] - right[1]) - area
    return area / union if union > 0 else 0.0


def _uniform_positions(total: int, count: int) -> list[int]:
    """Spread count frame positions evenly over total frames."""
    if total <= count:
        return list(range(total))
    if count == 1:
        return [total // 2]
    return [round(index * (total - 1) / (count - 1)) for index in range(count)]


def extract_frames(
    path: str | Path, *, max_frames: int, max_pixels: int, max_megabytes: float
) -> list[VideoFrame]:
    """Validate a video and sample evenly spaced frames as RGB images."""
    source = Path(path)
    if not source.is_file():
        raise VideoError(f"Vidéo introuvable : « {source} ». Vérifiez le nom du fichier.")
    raw = source.read_bytes()
    if len(raw) > int(max_megabytes * 1024 * 1024):
        raise VideoError(
            f"La vidéo dépasse {max_megabytes:.0f} Mo. Choisissez une vidéo plus courte."
        )
    if source.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise VideoError("Format de vidéo non accepté. Utilisez une vidéo MP4, MOV, WebM ou AVI.")
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise VideoError("Impossible de lire cette vidéo. Essayez une autre vidéo.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        rate = fps if fps and fps > 0 else 25.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        images: list[tuple[int, int, Image.Image]] = []
        if total >= max_frames:
            for order, position in enumerate(_uniform_positions(total, max_frames)):
                capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                images.append((order, position, _to_pil(frame, max_pixels)))
        else:
            collected: list[Image.Image] = []
            while len(collected) < _MAX_SEQUENTIAL_FRAMES:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                collected.append(_to_pil(frame, max_pixels))
            for order, position in enumerate(_uniform_positions(len(collected), max_frames)):
                images.append((order, position, collected[position]))
    finally:
        capture.release()
    if not images:
        raise VideoError("Aucune image lisible dans cette vidéo.")
    return [
        VideoFrame(index=order, timestamp_s=round(position / rate, 2), image=image)
        for order, position, image in images
    ]


def _to_pil(frame: object, max_pixels: int) -> Image.Image:
    """Convert an OpenCV BGR frame to a resized RGB image."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore[arg-type]
    image = Image.fromarray(rgb)
    image.thumbnail((max_pixels, max_pixels), Image.LANCZOS)
    return image


def frames_dir_for(settings: Settings, video_sha256: str) -> Path:
    """Return the frames directory for one video fingerprint."""
    return settings.data_dir / "videos" / video_sha256[:12]


def save_frames(
    frames: list[VideoFrame], directory: str | Path, *, jpeg_quality: int
) -> list[Path]:
    """Save sampled frames as JPEGs and return their paths."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for frame in frames:
        path = target / f"frame-{frame.index:02d}.jpg"
        frame.image.save(path, format="JPEG", quality=jpeg_quality)
        paths.append(path)
    return paths


def fuse_defects(frame_reports: list[FrameReport]) -> list[FusedDefect]:
    """Group flaw occurrences across frames into fused defects with counts."""
    if not frame_reports:
        return []
    ordered = sorted(frame_reports, key=lambda item: (-len(item.report.defects), item.frame_index))
    clusters: list[list[tuple[int, Defect]]] = []
    for item in ordered:
        for defect in item.report.defects:
            best: list[tuple[int, Defect]] | None = None
            best_score = MATCH_THRESHOLD - 1
            for cluster in clusters:
                if any(index == item.frame_index for index, _ in cluster):
                    continue
                score = pair_score(cluster[0][1], defect)
                first, second = cluster[0][1].bbox, defect.bbox
                if first is not None and second is not None and _bbox_iou(first, second) == 0.0:
                    continue
                if score > best_score:
                    best, best_score = cluster, score
            if best is None or best_score < MATCH_THRESHOLD:
                clusters.append([(item.frame_index, defect)])
            else:
                best.append((item.frame_index, defect))
    fused: list[FusedDefect] = []
    for number, cluster in enumerate(clusters, start=1):
        representative = cluster[0][1]
        severity = representative.severity
        for _, occurrence in cluster:
            if _SEVERITY_RANK[occurrence.severity] > _SEVERITY_RANK[severity]:
                severity = occurrence.severity
        bbox = representative.bbox
        if bbox is None:
            for _, occurrence in cluster:
                if occurrence.bbox is not None:
                    bbox = occurrence.bbox
                    break
        fused.append(
            FusedDefect(
                id=f"F{number}",
                type=representative.type,
                description=representative.description,
                zone=representative.zone,
                severity=severity,
                distinctive=any(occurrence.distinctive for _, occurrence in cluster),
                confirmations=len(cluster),
                frames=sorted(index for index, _ in cluster),
                bbox=bbox,
            )
        )
    return fused


def fuse_challenge(expected_code: str, frame_reports: list[FrameReport]) -> VideoChallenge:
    """Summarize challenge-code observations across frames."""
    code = expected_code.strip().upper()
    seen = 0
    read: str | None = None
    for item in frame_reports:
        check = item.report.challenge
        observed = (check.read_code or "").strip().upper()
        if check.visible and observed and observed == code:
            seen += 1
            if read is None:
                read = observed
    return VideoChallenge(
        expected_code=code,
        seen_frames=seen,
        total_frames=len(frame_reports),
        read_code=read,
        verified=seen >= 1,
    )


def fuse_alerts(frame_reports: list[FrameReport]) -> list[Alert]:
    """Union frame alerts, keeping first-seen order."""
    merged: list[Alert] = []
    seen: set[tuple[str, str]] = set()
    for item in frame_reports:
        for alert in item.report.alerts:
            key = (alert.category, alert.message)
            if key not in seen:
                seen.add(key)
                merged.append(alert)
    return merged


def fuse_checkpoints(
    frame_reports: list[FrameReport], fused: list[FusedDefect]
) -> list[Checkpoint]:
    """Remap the reference frame checkpoints to fused defect ids."""
    if not frame_reports:
        return []
    reference = max(frame_reports, key=lambda item: (len(item.report.defects), -item.frame_index))
    owners: dict[str, str] = {}
    for flaw in fused:
        for index in flaw.frames:
            for item in frame_reports:
                if item.frame_index != index:
                    continue
                for defect in item.report.defects:
                    owners.setdefault(f"{index}:{defect.id}", flaw.id)
    remapped: list[Checkpoint] = []
    for point in reference.report.checkpoints:
        owner = owners.get(f"{reference.frame_index}:{point.defect_id}")
        if owner is not None:
            remapped.append(Checkpoint(defect_id=owner, instruction=point.instruction))
    return remapped


def is_fresh_video(report: VideoReport) -> bool:
    """Return True when the challenge code was read on at least one frame."""
    return report.challenge.verified


def create_video_report(
    video: str | Path,
    expected_code: str,
    *,
    settings: Settings | None = None,
    client: ChatClient | None = None,
    max_frames: int | None = None,
    frames_dir: str | Path | None = None,
    on_frame: Callable[[int, int], None] | None = None,
) -> VideoReport:
    """Analyse sampled video frames and return their fused constat."""
    active_settings = settings or load_settings()
    wanted = active_settings.video_max_frames if max_frames is None else max_frames
    if wanted < 1:
        raise VideoError("Analysez au moins une vue de la vidéo.")
    source = Path(video)
    video_sha = compute_sha256(source.read_bytes()) if source.is_file() else ""
    frames = extract_frames(
        source,
        max_frames=wanted,
        max_pixels=active_settings.image_max_pixels,
        max_megabytes=active_settings.video_max_megabytes,
    )
    if frames_dir:
        target = Path(frames_dir)
    else:
        target = frames_dir_for(active_settings, video_sha)
    paths = save_frames(frames, target, jpeg_quality=active_settings.image_jpeg_quality)
    code = expected_code.strip().upper()
    frame_reports: list[FrameReport] = []
    for position, (frame, path) in enumerate(zip(frames, paths, strict=True)):
        if on_frame is not None:
            on_frame(position + 1, len(frames))
        report = create_report(path, code, settings=active_settings, client=client)
        frame_reports.append(
            FrameReport(frame_index=frame.index, timestamp_s=frame.timestamp_s, report=report)
        )
    fused = fuse_defects(frame_reports)
    reference = max(frame_reports, key=lambda item: len(item.report.defects))
    logger.info("Video constat: %d frames, %d fused defects.", len(frame_reports), len(fused))
    return VideoReport(
        description=reference.report.description,
        fused_defects=fused,
        frames=frame_reports,
        challenge=fuse_challenge(code, frame_reports),
        checkpoints=fuse_checkpoints(frame_reports, fused),
        alerts=fuse_alerts(frame_reports),
        video_sha256=video_sha,
        created_at_utc=datetime.now(UTC),
        prompt_version=PROMPT_VERSION,
    )
