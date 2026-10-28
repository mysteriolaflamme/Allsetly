"""Shared fixtures: sample photo, valid model answer and test settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from temoin.config import Settings


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
