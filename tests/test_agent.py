"""Tests for the investigator agent loop (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from temoin.agent import build_investigation_cache_key, run_investigation
from temoin.config import Settings
from temoin.model import ChatAnswer, ToolCall
from temoin.schemas import Comparison, TrailEntry
from tests.fake_client import FakeChatClient


def _photos(tmp_path: Path) -> tuple[Path, Path]:
    """Create a sale photo and a delivery photo."""
    sale = tmp_path / "vente.jpg"
    delivery = tmp_path / "livraison.jpg"
    Image.new("RGB", (800, 600), color=(120, 80, 40)).save(sale, format="JPEG")
    Image.new("RGB", (800, 600), color=(40, 80, 120)).save(delivery, format="JPEG")
    return sale, delivery


def _delivery_draft(draft_json: str, code: str) -> str:
    """Return a constat answer reading the given delivery code."""
    payload = json.loads(draft_json)
    payload["challenge"]["read_code"] = code
    return json.dumps(payload)


def _call(name: str, arguments: dict[str, Any], id: str = "call-1") -> ToolCall:
    """Build one scripted tool call."""
    return ToolCall(id=id, name=name, arguments=json.dumps(arguments))


_DEVIL_CONFIRM = json.dumps({"objections": [], "stand_by_verdict": True})


def test_run_investigation_with_zoom(
    tmp_path: Path, test_settings: Settings, draft_json: str, comparison_json: str
) -> None:
    """The agent zooms once, then concludes; the trail shows the zoom."""
    sale, delivery = _photos(tmp_path)
    zoom = _call("zoom", {"photo": "sale", "bbox": [100, 100, 500, 500]})
    client = FakeChatClient(
        [
            draft_json,
            _delivery_draft(draft_json, "Q9"),
            ChatAnswer(content="", tool_calls=[zoom]),
            comparison_json,
            _DEVIL_CONFIRM,
        ]
    )
    comparison = run_investigation(
        sale,
        delivery,
        sale_code="K7",
        delivery_code="Q9",
        settings=test_settings,
        client=client,
    )
    assert isinstance(comparison, Comparison)
    assert comparison.verdict == "CONFORME"
    assert comparison.capped_reason is None
    actions = [entry.action for entry in comparison.trail]
    assert actions == ["constat_sale", "constat_delivery", "tool:zoom", "verdict", "redteam"]
    assert len(client.calls) == 5


def test_run_investigation_checks_both_codes(
    tmp_path: Path, test_settings: Settings, draft_json: str, comparison_json: str
) -> None:
    """check_challenge results appear in the trail."""
    sale, delivery = _photos(tmp_path)
    calls = [
        _call("check_challenge", {"photo": "sale", "code": "K7"}, id="call-1"),
        _call("check_challenge", {"photo": "delivery", "code": "Q9"}, id="call-2"),
    ]
    client = FakeChatClient(
        [
            draft_json,
            _delivery_draft(draft_json, "Q9"),
            ChatAnswer(content="", tool_calls=calls),
            comparison_json,
            _DEVIL_CONFIRM,
        ]
    )
    comparison = run_investigation(
        sale,
        delivery,
        sale_code="K7",
        delivery_code="Q9",
        settings=test_settings,
        client=client,
    )
    actions = [entry.action for entry in comparison.trail]
    assert actions.count("tool:check_challenge") == 2
    assert comparison.verdict == "CONFORME"


def test_run_investigation_request_photo_concludes_without_tools(
    tmp_path: Path, test_settings: Settings, draft_json: str, comparison_json: str
) -> None:
    """After request_photo, the agent concludes with tools disabled."""
    sale, delivery = _photos(tmp_path)
    payload = json.loads(comparison_json)
    payload["verdict"] = "À VÉRIFIER"
    payload["missing_proof"] = ["Rephotographiez le devant en gros plan."]
    final = json.dumps(payload)
    asked = _call("request_photo", {"zone": "devant", "instruction": "Gros plan."})
    client = FakeChatClient(
        [
            draft_json,
            _delivery_draft(draft_json, "Q9"),
            ChatAnswer(content="", tool_calls=[asked]),
            final,
            _DEVIL_CONFIRM,
        ]
    )
    comparison = run_investigation(
        sale,
        delivery,
        sale_code="K7",
        delivery_code="Q9",
        settings=test_settings,
        client=client,
    )
    assert comparison.verdict == "À VÉRIFIER"
    assert len(client.calls) == 5
    assert client.calls[2]["tools"] is not None
    assert client.calls[3]["tools"] is None


def test_run_investigation_retries_invalid_final_json(
    tmp_path: Path, test_settings: Settings, draft_json: str, comparison_json: str
) -> None:
    """An invalid final message is sent back for a new attempt."""
    sale, delivery = _photos(tmp_path)
    client = FakeChatClient(
        [draft_json, _delivery_draft(draft_json, "Q9"), "not json", comparison_json, _DEVIL_CONFIRM]
    )
    comparison = run_investigation(
        sale,
        delivery,
        sale_code="K7",
        delivery_code="Q9",
        settings=test_settings,
        client=client,
    )
    assert comparison.verdict == "CONFORME"
    assert len(client.calls) == 5


def test_run_investigation_falls_back_to_single_shot(
    tmp_path: Path, test_settings: Settings, draft_json: str, comparison_json: str
) -> None:
    """When the agent never concludes, a single-shot verdict rescues the run."""
    sale, delivery = _photos(tmp_path)
    script: list[str] = [draft_json, _delivery_draft(draft_json, "Q9")]
    script += ["junk"] * test_settings.agent_max_steps
    script.append(comparison_json)
    script.append(_DEVIL_CONFIRM)
    client = FakeChatClient(script)
    comparison = run_investigation(
        sale,
        delivery,
        sale_code="K7",
        delivery_code="Q9",
        settings=test_settings,
        client=client,
    )
    assert comparison.verdict == "CONFORME"
    actions = [entry.action for entry in comparison.trail]
    assert "fallback" in actions
    assert len(client.calls) == 2 + test_settings.agent_max_steps + 2


def test_run_investigation_without_codes_caps_verdict(
    tmp_path: Path, test_settings: Settings, draft_json: str, comparison_json: str
) -> None:
    """Unknown codes fail freshness honestly, capping the verdict."""
    sale, delivery = _photos(tmp_path)
    client = FakeChatClient([draft_json, draft_json, comparison_json, _DEVIL_CONFIRM])
    comparison = run_investigation(sale, delivery, settings=test_settings, client=client)
    assert comparison.verdict == "À VÉRIFIER"
    assert comparison.capped_reason == "preuve de fraîcheur manquante"


def test_run_investigation_uses_cache(
    tmp_path: Path, test_settings: Settings, draft_json: str, comparison_json: str
) -> None:
    """A second identical investigation replays from disk without new calls."""
    sale, delivery = _photos(tmp_path)
    first = FakeChatClient(
        [draft_json, _delivery_draft(draft_json, "Q9"), comparison_json, _DEVIL_CONFIRM]
    )
    out1 = run_investigation(
        sale,
        delivery,
        sale_code="K7",
        delivery_code="Q9",
        settings=test_settings,
        client=first,
    )
    assert out1.verdict == "CONFORME"
    second = FakeChatClient([])
    out2 = run_investigation(
        sale,
        delivery,
        sale_code="K7",
        delivery_code="Q9",
        settings=test_settings,
        client=second,
    )
    assert out2.verdict == "CONFORME"
    assert [entry.action for entry in out2.trail] == [entry.action for entry in out1.trail]
    assert second.calls == []


def test_build_investigation_cache_key_covers_constats() -> None:
    """Both constats shape the investigation cache key."""
    base = build_investigation_cache_key(sale_constat="a", delivery_constat="b")
    assert base == build_investigation_cache_key(sale_constat="a", delivery_constat="b")
    assert base != build_investigation_cache_key(sale_constat="X", delivery_constat="b")
    assert base != build_investigation_cache_key(sale_constat="a", delivery_constat="X")


def test_run_investigation_emits_trail_steps(
    tmp_path: Path, test_settings: Settings, draft_json: str, comparison_json: str
) -> None:
    """on_step receives every trail entry in order."""
    sale, delivery = _photos(tmp_path)
    zoom = _call("zoom", {"photo": "sale", "bbox": [100, 100, 500, 500]})
    client = FakeChatClient(
        [
            draft_json,
            _delivery_draft(draft_json, "Q9"),
            ChatAnswer(content="", tool_calls=[zoom]),
            comparison_json,
            _DEVIL_CONFIRM,
        ]
    )
    received: list[TrailEntry] = []
    comparison = run_investigation(
        sale,
        delivery,
        sale_code="K7",
        delivery_code="Q9",
        settings=test_settings,
        client=client,
        on_step=received.append,
    )
    assert [entry.action for entry in received] == [entry.action for entry in comparison.trail]
    assert received[2].action == "tool:zoom"


def test_run_investigation_without_redteam_skips_challenge(
    tmp_path: Path, test_settings: Settings, draft_json: str, comparison_json: str
) -> None:
    """redteam=False saves one call and leaves the trail without challenge."""
    sale, delivery = _photos(tmp_path)
    client = FakeChatClient([draft_json, _delivery_draft(draft_json, "Q9"), comparison_json])
    comparison = run_investigation(
        sale,
        delivery,
        sale_code="K7",
        delivery_code="Q9",
        settings=test_settings,
        client=client,
        redteam=False,
    )
    assert comparison.verdict == "CONFORME"
    assert "redteam" not in [entry.action for entry in comparison.trail]
    assert len(client.calls) == 3


def test_run_investigation_redteam_downgrades(
    tmp_path: Path, test_settings: Settings, draft_json: str, comparison_json: str
) -> None:
    """A major devil objection caps the investigation verdict."""
    sale, delivery = _photos(tmp_path)
    devil = json.dumps(
        {
            "objections": [{"objection": "Le match D1 est douteux.", "severity": "major"}],
            "stand_by_verdict": False,
        }
    )
    client = FakeChatClient([draft_json, _delivery_draft(draft_json, "Q9"), comparison_json, devil])
    comparison = run_investigation(
        sale,
        delivery,
        sale_code="K7",
        delivery_code="Q9",
        settings=test_settings,
        client=client,
    )
    assert comparison.verdict == "À VÉRIFIER"
    assert comparison.capped_reason == "avocat du diable : 1 majeure"
