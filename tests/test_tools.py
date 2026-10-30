"""Tests for the investigator agent tools (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

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
def context(tmp_path: Path, sale_report: Report, delivery_report: Report) -> ToolContext:
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
    )


def test_agent_tools_are_the_step_four_tools() -> None:
    """Step 4 exposes exactly zoom, match, challenge check and photo ask."""
    names = {tool["function"]["name"] for tool in AGENT_TOOLS}
    assert names == {"match_defects", "zoom", "check_challenge", "request_photo"}


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
