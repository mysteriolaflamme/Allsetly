"""Shared fixtures: sample photo, valid model answer and test settings."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from temoin.config import Settings
from temoin.images import PreparedImage
from temoin.schemas import (
    ChallengeCheck,
    Comparison,
    ComparisonDraft,
    Defect,
    Report,
    TrailEntry,
)


@pytest.fixture
def photo_path(tmp_path: Path) -> Path:
    """Create a small valid JPEG photo."""
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (640, 480), color=(120, 80, 40)).save(path, format="JPEG")
    return path


@pytest.fixture
def draft_payload() -> dict[str, Any]:
    """Return a valid model answer payload."""
    return {
        "description": "Une sacoche en cuir marron avec bandoulière.",
        "defects": [
            {
                "id": "D1",
                "type": "rayure",
                "description": "Rayure claire de 3 cm sur le devant.",
                "zone": "devant, en bas à droite",
                "severity": "moderate",
                "bbox": [600, 700, 800, 800],
                "distinctive": True,
            },
            {
                "id": "D2",
                "type": "usure",
                "description": "Coin usé, cuir éclairci.",
                "zone": "coin inférieur gauche",
                "severity": "minor",
                "bbox": None,
                "distinctive": False,
            },
        ],
        "challenge": {
            "expected_code": "K7",
            "visible": True,
            "read_code": "K7",
            "looks_printed_or_edited": False,
        },
        "checkpoints": [
            {
                "defect_id": "D1",
                "instruction": "Photographiez la rayure du devant en gros plan.",
            }
        ],
        "alerts": [],
    }


@pytest.fixture
def draft_json(draft_payload: dict[str, Any]) -> str:
    """Return the valid model answer as a JSON string."""
    return json.dumps(draft_payload)


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Return settings pointing at an isolated cache directory."""
    return Settings(baseten_api_key="test-key", cache_dir=tmp_path / "cache")


@pytest.fixture
def empty_env_file(tmp_path: Path) -> str:
    """Return the path of an empty dotenv file."""
    path = tmp_path / "empty.env"
    path.write_text("", encoding="utf-8")
    return str(path)


def _defect_dict(id: str, distinctive: bool = True) -> dict[str, Any]:
    """Build one defect payload for comparison tests."""
    return {
        "id": id,
        "type": "rayure",
        "description": f"Rayure {id}.",
        "zone": "devant",
        "severity": "moderate",
        "bbox": None,
        "distinctive": distinctive,
    }


def _make_report(
    *,
    expected_code: str,
    read_code: str | None,
    visible: bool,
    defect_ids: list[str],
) -> Report:
    """Build a validated Report for comparison tests."""
    return Report(
        description=f"Objet de test pour le code {expected_code}.",
        defects=[
            Defect.model_validate(_defect_dict(id, distinctive=index == 0))
            for index, id in enumerate(defect_ids)
        ],
        challenge=ChallengeCheck(expected_code=expected_code, visible=visible, read_code=read_code),
        checkpoints=[],
        alerts=[],
        image_sha256=f"sha256-{expected_code}",
        created_at_utc=datetime.now(UTC),
        prompt_version="1",
    )


@pytest.fixture
def sale_report() -> Report:
    """Return a fresh sale constat with two defects."""
    return _make_report(expected_code="K7", read_code="K7", visible=True, defect_ids=["D1", "D2"])


@pytest.fixture
def delivery_report() -> Report:
    """Return a fresh delivery constat with two defects."""
    return _make_report(expected_code="Q9", read_code="Q9", visible=True, defect_ids=["D1", "D2"])


@pytest.fixture
def sale_image() -> PreparedImage:
    """Return a tiny fake prepared sale image."""
    return PreparedImage(data_url="data:,sale", sha256="a" * 64, width=100, height=100)


@pytest.fixture
def delivery_image() -> PreparedImage:
    """Return a tiny fake prepared delivery image."""
    return PreparedImage(data_url="data:,delivery", sha256="b" * 64, width=100, height=100)


@pytest.fixture
def comparison_payload() -> dict[str, Any]:
    """Return a valid comparison answer payload."""
    return {
        "verdict": "CONFORME",
        "confidence": 90,
        "same_object": True,
        "new_damage": False,
        "justification": "Mêmes défauts distinctifs aux mêmes endroits.",
        "matches": [
            {
                "sale_defect_id": "D1",
                "delivery_defect_id": "D1",
                "status": "matched",
                "note": "Même rayure.",
            },
            {
                "sale_defect_id": "D2",
                "delivery_defect_id": "D2",
                "status": "matched",
                "note": "Même usure.",
            },
        ],
        "contradictions": [],
        "missing_proof": [],
        "alerts": [],
    }


@pytest.fixture
def comparison_json(comparison_payload: dict[str, Any]) -> str:
    """Return the valid comparison answer as a JSON string."""
    return json.dumps(comparison_payload)


@pytest.fixture
def comparison(
    comparison_payload: dict[str, Any], sale_report: Report, delivery_report: Report
) -> Comparison:
    """Return a finalized comparison with a one-step trail."""
    draft = ComparisonDraft.model_validate(comparison_payload)
    return Comparison.from_draft(
        draft,
        verdict="CONFORME",
        capped_reason=None,
        sale=sale_report,
        delivery=delivery_report,
        freshness_sale_ok=True,
        freshness_delivery_ok=True,
        created_at_utc=datetime.now(UTC),
        prompt_version="1",
        disclaimer="D.",
        trail=[TrailEntry(step=1, action="verdict", summary="Model verdict: CONFORME.")],
    )
