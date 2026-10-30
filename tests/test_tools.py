"""Tests for the investigator agent tools (no network)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from temoin.config import Settings
from temoin.meta_api import MetaAPIError, Segmentation, SegmentMask
from temoin.schemas import Report
from temoin.tools import (
    AGENT_TOOLS,
    ToolContext,
    ToolError,
    check_challenge,
    execute_tool,
    match_defects_table,
    pair_score,
    zoom_photo,
)


@pytest.fixture
def context(
    tmp_path: Path, sale_report: Report, delivery_report: Report, test_settings: Settings
) -> ToolContext:
    """Return a tool context with two real photo files."""
    sale_photo = tmp_path / "vente.jpg"
    delivery_photo = tmp_path / "livraison.jpg"
    Image.new("RGB", (1000, 800), color=(90, 60, 30)).save(sale_photo, format="JPEG")
    Image.new("RGB", (1000, 800), color=(30, 60, 90)).save(delivery_photo, format="JPEG")
    return ToolContext(
        sale=sale_report,
        delivery=delivery_report,
        sale_photo=sale_photo,
        delivery_photo=delivery_photo,
        max_pixels=1024,
        jpeg_quality=90,
        settings=test_settings,
    )


def test_agent_tools_are_the_known_tools() -> None:
    """The agent exposes match, zoom, challenge check, photo ask and segment."""
    names = {tool["function"]["name"] for tool in AGENT_TOOLS}
    assert names == {"match_defects", "zoom", "check_challenge", "request_photo", "segment"}


def test_zoom_photo_enlarges_region(tmp_path: Path) -> None:
    """A 500x500 region of a 1000x1000 photo becomes 1024x1024."""
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (1000, 1000), color=(10, 20, 30)).save(photo, format="JPEG")
    view = zoom_photo(photo, [0, 0, 500, 500], max_pixels=1024, jpeg_quality=90)
    assert (view.width, view.height) == (1024, 1024)
    assert view.data_url.startswith("data:image/jpeg;base64,")


def test_zoom_photo_rejects_bad_boxes(tmp_path: Path) -> None:
    """Malformed or out-of-range boxes raise ToolError."""
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (100, 100)).save(photo, format="JPEG")
    for bad in ([0, 0, 500], [0, 0, 500, 1001], [300, 300, 100, 100], ["a", 0, 1, 2]):
        with pytest.raises(ToolError, match="bbox"):
            zoom_photo(photo, bad, max_pixels=1024, jpeg_quality=90)


def test_pair_score_prefers_same_type_and_zone(
    sale_report: Report, delivery_report: Report
) -> None:
    """Identical flaws score higher than different ones."""
    flaw = sale_report.defects[0]
    same = pair_score(flaw, delivery_report.defects[0])
    other = delivery_report.defects[1].model_copy(update={"type": "tache", "zone": "derrière"})
    assert same > pair_score(flaw, other)


def test_match_defects_table_lists_candidates(sale_report: Report, delivery_report: Report) -> None:
    """The table shows ranked candidates for each sale flaw."""
    table = match_defects_table(sale_report, delivery_report)
    assert "D1" in table and "score" in table
    assert "Delivery flaws to account for" in table


def test_match_defects_table_empty(sale_report: Report, delivery_report: Report) -> None:
    """Two empty constats give a plain sentence."""
    empty_sale = sale_report.model_copy(update={"defects": []})
    empty_delivery = delivery_report.model_copy(update={"defects": []})
    assert match_defects_table(empty_sale, empty_delivery) == (
        "No flaws recorded in either constat."
    )


def test_check_challenge_verifies_code(sale_report: Report) -> None:
    """The recorded observation is compared to the expected code."""
    assert "verified=true" in check_challenge(sale_report, "sale", "K7")
    assert "verified=true" in check_challenge(sale_report, "sale", " k7 ")
    assert "verified=false" in check_challenge(sale_report, "sale", "XX")


def test_execute_tool_routes_all_tools(context: ToolContext) -> None:
    """Each known tool runs through execute_tool."""
    assert "candidates" in execute_tool("match_defects", {}, context).content
    zoomed = execute_tool("zoom", {"photo": "sale", "bbox": [0, 0, 500, 400]}, context)
    assert zoomed.image_data_url is not None
    assert zoomed.image_data_url.startswith("data:image/jpeg;base64,")
    checked = execute_tool("check_challenge", {"photo": "sale", "code": "K7"}, context)
    assert "verified=true" in checked.content
    asked = execute_tool("request_photo", {"zone": "devant", "instruction": "Gros plan."}, context)
    assert "devant" in asked.content and asked.image_data_url is None


def test_execute_tool_reports_errors_as_content(context: ToolContext) -> None:
    """Bad arguments and unknown tools never raise."""
    bad = execute_tool("zoom", {"photo": "sale", "bbox": [0, 0]}, context)
    assert bad.content.startswith("Tool error:")
    unknown = execute_tool("fly", {}, context)
    assert "Unknown tool" in unknown.content


def _fake_segmentation() -> Segmentation:
    """Return one synthetic headphone mask on a 1000x800 photo."""
    width, height = 90, 80
    mask = SegmentMask(
        object_id="0",
        left=10.0,
        top=10.0,
        right=100.0,
        bottom=90.0,
        width=width,
        height=height,
        pixels=bytes([1]) * (width * height),
    )
    return Segmentation(concept="headphones", image_width=1000, image_height=800, masks=(mask,))


def test_segment_tool_outlines_object(
    monkeypatch: pytest.MonkeyPatch, context: ToolContext
) -> None:
    """The segment tool returns boxes plus an overlay image."""
    seen: dict[str, Any] = {}

    def fake_segment(
        data_url: str, concept: str, *, settings: Settings | None = None
    ) -> Segmentation:
        seen["concept"] = concept
        return _fake_segmentation()

    monkeypatch.setattr("temoin.tools.segment_image", fake_segment)
    result = execute_tool("segment", {"photo": "sale", "concept": "headphones"}, context)
    assert seen["concept"] == "headphones"
    assert "1 mask(s)" in result.content
    assert "0:[10,10,100,90]" in result.content
    assert result.image_data_url is not None
    assert result.image_data_url.startswith("data:image/jpeg;base64,")


def test_segment_tool_degrades_without_settings(context: ToolContext) -> None:
    """Without settings the tool guides back to the constat boxes."""
    bare = replace(context, settings=None)
    result = execute_tool("segment", {"photo": "delivery", "concept": "x"}, bare)
    assert "unavailable" in result.content
    assert "constat bboxes" in result.content
    assert result.image_data_url is None


def test_segment_tool_degrades_on_meta_error(
    monkeypatch: pytest.MonkeyPatch, context: ToolContext
) -> None:
    """A Meta failure is reported as content, never raised."""

    def failing(data_url: str, concept: str, *, settings: Settings | None = None) -> Segmentation:
        raise MetaAPIError("nope")

    monkeypatch.setattr("temoin.tools.segment_image", failing)
    result = execute_tool("segment", {"photo": "sale", "concept": "x"}, context)
    assert "unavailable" in result.content
    assert "nope" in result.content
    assert result.image_data_url is None


def test_segment_tool_rejects_bad_arguments(context: ToolContext) -> None:
    """Bad photo or concept values become tool errors."""
    bad_photo = execute_tool("segment", {"photo": "sideways", "concept": "x"}, context)
    assert bad_photo.content.startswith("Tool error:")
    missing = execute_tool("segment", {"photo": "sale"}, context)
    assert missing.content.startswith("Tool error:")
    blank = execute_tool("segment", {"photo": "sale", "concept": "  "}, context)
    assert blank.content.startswith("Tool error:")
