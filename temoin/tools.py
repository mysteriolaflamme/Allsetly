"""Local tools of the investigator agent: zoom, match, challenge check, photo ask."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from temoin.config import Settings
from temoin.images import decode_data_url, prepare_image
from temoin.meta_api import MetaAPIError, draw_segmentation, segment_image
from temoin.schemas import Defect, Report


class ToolError(Exception):
    """Raised when a tool call has invalid arguments."""


@dataclass(frozen=True)
class ToolResult:
    """Outcome of one tool execution."""

    name: str
    content: str
    image_data_url: str | None = None


@dataclass(frozen=True)
class ToolContext:
    """Everything tools need: constats, photo paths and output size."""

    sale: Report
    delivery: Report
    sale_photo: Path
    delivery_photo: Path
    max_pixels: int
    jpeg_quality: int
    settings: Settings | None = None


@dataclass(frozen=True)
class ZoomView:
    """Enlarged crop of one photo region."""

    data_url: str
    width: int
    height: int


AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "match_defects",
            "description": (
                "Show the alignment table of flaws between the sale and "
                "delivery constats, with similarity scores."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zoom",
            "description": (
                "Re-examine a photo region closely: crop the bbox area from "
                "the full-resolution photo and enlarge it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "photo": {"type": "string", "enum": ["sale", "delivery"]},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "[x1, y1, x2, y2] in 0-1000 coordinates.",
                    },
                },
                "required": ["photo", "bbox"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_challenge",
            "description": "Re-check the challenge code recorded for one photo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "photo": {"type": "string", "enum": ["sale", "delivery"]},
                    "code": {"type": "string", "description": "Expected code."},
                },
                "required": ["photo", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_photo",
            "description": (
                "Ask the user for a missing proof photo. Ends the "
                "investigation: conclude À VÉRIFIER afterwards."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zone": {"type": "string"},
                    "instruction": {"type": "string"},
                },
                "required": ["zone", "instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "segment",
            "description": (
                "Outline one named object in a photo with SAM masks: "
                "returns boxes and an overlay image with exact contours."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "photo": {"type": "string", "enum": ["sale", "delivery"]},
                    "concept": {
                        "type": "string",
                        "description": "Short noun phrase, e.g. headphones.",
                    },
                },
                "required": ["photo", "concept"],
            },
        },
    },
]


def _photo_label(value: Any) -> str:
    """Validate a photo argument, returning 'sale' or 'delivery'."""
    if value not in ("sale", "delivery"):
        raise ToolError("Invalid photo: expected 'sale' or 'delivery'.")
    return str(value)


def zoom_photo(
    photo_path: str | Path, bbox: list[Any], *, max_pixels: int, jpeg_quality: int
) -> ZoomView:
    """Crop a bbox region (0-1000) from the full-resolution photo and enlarge it."""
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ToolError("Invalid bbox: expected [x1, y1, x2, y2] integers.")
    if not all(isinstance(value, int) for value in bbox):
        raise ToolError("Invalid bbox: expected [x1, y1, x2, y2] integers.")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise ToolError("Invalid bbox: need 0 <= x1 < x2 <= 1000, same for y.")
    with Image.open(photo_path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
    width, height = image.size
    box = (
        round(x1 * width / 1000),
        round(y1 * height / 1000),
        round(x2 * width / 1000),
        round(y2 * height / 1000),
    )
    crop = image.crop(box)
    scale = max_pixels / max(crop.size)
    enlarged = crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
        Image.LANCZOS,
    )
    buffer = io.BytesIO()
    enlarged.save(buffer, format="JPEG", quality=jpeg_quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    out_width, out_height = enlarged.size
    return ZoomView(
        data_url=f"data:image/jpeg;base64,{encoded}",
        width=out_width,
        height=out_height,
    )


def _zone_words(zone: str) -> set[str]:
    """Split a zone into significant lowercase words."""
    return {word.strip(".,;:").lower() for word in zone.split() if len(word) > 2}


def pair_score(sale: Defect, delivery: Defect) -> int:
    """Score how likely two flaws are the same one (higher is likelier)."""
    score = 0
    if sale.type.strip().lower() == delivery.type.strip().lower():
        score += 2
    score += len(_zone_words(sale.zone) & _zone_words(delivery.zone))
    if sale.distinctive and delivery.distinctive:
        score += 1
    if sale.severity == delivery.severity:
        score += 1
    return score


def match_defects_table(sale: Report, delivery: Report) -> str:
    """Build a text table of flaw alignment candidates with scores."""
    if not sale.defects and not delivery.defects:
        return "No flaws recorded in either constat."
    lines = ["Flaw alignment candidates (higher score is likelier):"]
    for flaw in sale.defects:
        ranked = sorted(
            ((pair_score(flaw, other), other) for other in delivery.defects),
            key=lambda item: item[0],
            reverse=True,
        )[:3]
        if not ranked:
            lines.append(f"- {flaw.id} ({flaw.type}, {flaw.zone}): no delivery flaws.")
            continue
        for score, other in ranked:
            lines.append(
                f"- {flaw.id} ({flaw.type}, {flaw.zone}) "
                f"<-> {other.id} ({other.type}, {other.zone}): score {score}."
            )
    if delivery.defects:
        listed = ", ".join(f"{flaw.id} ({flaw.type})" for flaw in delivery.defects)
        lines.append(f"Delivery flaws to account for: {listed}.")
    return "\n".join(lines)


def check_challenge(report: Report, photo_label: str, code: str) -> str:
    """Re-check the recorded code observation of one photo against a code."""
    read = (report.challenge.read_code or "").strip().upper()
    expected = code.strip().upper()
    verified = report.challenge.visible and bool(expected) and read == expected
    return (
        f"photo={photo_label} expected={expected or '?'} "
        f"visible={str(report.challenge.visible).lower()} "
        f"read={read or 'none'} verified={str(verified).lower()}"
    )


def execute_tool(name: str, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    """Run one tool by name; tool failures become result content, never raise."""
    try:
        return _dispatch_tool(name, arguments, context)
    except ToolError as error:
        return ToolResult(name=name, content=f"Tool error: {error}")
    except Exception as error:
        # Report tool crashes to the agent instead of breaking the loop.
        return ToolResult(name=name, content=f"Tool failed: {error}")


def _dispatch_tool(name: str, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    """Route one tool call to its implementation."""
    if name == "match_defects":
        return ToolResult(name=name, content=match_defects_table(context.sale, context.delivery))
    if name == "zoom":
        photo = _photo_label(arguments.get("photo"))
        bbox = arguments.get("bbox")
        if not isinstance(bbox, list):
            raise ToolError("Invalid bbox: expected [x1, y1, x2, y2] integers.")
        path = context.sale_photo if photo == "sale" else context.delivery_photo
        view = zoom_photo(
            path, bbox, max_pixels=context.max_pixels, jpeg_quality=context.jpeg_quality
        )
        return ToolResult(
            name=name,
            content=(
                f"Zoomed view of the {photo} photo, region {bbox}: "
                f"{view.width}x{view.height} pixels. The cropped image follows."
            ),
            image_data_url=view.data_url,
        )
    if name == "check_challenge":
        photo = _photo_label(arguments.get("photo"))
        code = arguments.get("code")
        if not isinstance(code, str):
            raise ToolError("Invalid code: expected a string.")
        report = context.sale if photo == "sale" else context.delivery
        return ToolResult(name=name, content=check_challenge(report, photo, code))
    if name == "request_photo":
        zone = arguments.get("zone")
        instruction = arguments.get("instruction")
        if not isinstance(zone, str) or not zone.strip():
            raise ToolError("Invalid zone: expected a non-empty string.")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ToolError("Invalid instruction: expected a non-empty string.")
        return ToolResult(
            name=name,
            content=(
                f"Recorded photo request for zone '{zone.strip()}': "
                f"{instruction.strip()} No new photo can arrive during this run: "
                "conclude with À VÉRIFIER and list this request in missing_proof."
            ),
        )
    if name == "segment":
        photo = _photo_label(arguments.get("photo"))
        concept = arguments.get("concept")
        if not isinstance(concept, str) or not concept.strip():
            raise ToolError("Invalid concept: expected a non-empty string.")
        if context.settings is None:
            return ToolResult(
                name=name,
                content=(
                    "Segmentation unavailable (Meta API key not configured). "
                    "Rely on the constat bboxes."
                ),
            )
        path = context.sale_photo if photo == "sale" else context.delivery_photo
        prepared = prepare_image(
            path,
            max_pixels=context.max_pixels,
            max_megabytes=context.settings.image_max_megabytes,
            jpeg_quality=context.jpeg_quality,
        )
        try:
            segmentation = segment_image(
                prepared.data_url, concept.strip(), settings=context.settings
            )
        except MetaAPIError as error:
            return ToolResult(
                name=name,
                content=f"Segmentation unavailable ({error}). Rely on the constat bboxes.",
            )
        overlay = draw_segmentation(decode_data_url(prepared.data_url), segmentation)
        buffer = io.BytesIO()
        overlay.save(buffer, format="JPEG", quality=context.jpeg_quality)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        boxes = (
            "; ".join(
                f"{mask.object_id}:[{round(mask.left)},{round(mask.top)},"
                f"{round(mask.right)},{round(mask.bottom)}]"
                for mask in segmentation.masks
            )
            or "none"
        )
        return ToolResult(
            name=name,
            content=(
                f"Segmentation of the {photo} photo for '{concept.strip()}': "
                f"{len(segmentation.masks)} mask(s). "
                f"Boxes (sent-image pixels): {boxes}. The overlay image follows."
            ),
            image_data_url=f"data:image/jpeg;base64,{encoded}",
        )
    known = ", ".join(tool["function"]["name"] for tool in AGENT_TOOLS)
    return ToolResult(name=name, content=f"Unknown tool: {name}. Available: {known}.")
