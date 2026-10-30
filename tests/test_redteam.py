"""Tests for the devil's advocate challenge (offline, fake client)."""

from __future__ import annotations

import json

from temoin.config import Settings
from temoin.model import ModelError
from temoin.redteam import build_redteam_cache_key, challenge_comparison
from temoin.schemas import Comparison, Report, TrailEntry
from tests.fake_client import FakeChatClient


def _challenge_json(objections: list[dict[str, str]], stand_by: bool) -> str:
    """Build a devil answer payload."""
    return json.dumps({"objections": objections, "stand_by_verdict": stand_by})


def _major(text: str = "La rayure D1 peut être un reflet.") -> dict[str, str]:
    """Build one major objection."""
    return {"objection": text, "severity": "major"}


def _minor(text: str = "La photo est un peu floue.") -> dict[str, str]:
    """Build one minor objection."""
    return {"objection": text, "severity": "minor"}


def test_challenge_keeps_solid_verdict(
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    test_settings: Settings,
) -> None:
    """No objection: verdict kept, challenge logged in the trail."""
    client = FakeChatClient([_challenge_json([], True)])
    out = challenge_comparison(
        comparison,
        sale=sale_report,
        delivery=delivery_report,
        settings=test_settings,
        client=client,
    )
    assert out.verdict == "CONFORME"
    assert out.capped_reason is None
    assert out.trail[-1].action == "redteam"
    assert "0 objection" in out.trail[-1].summary


def test_challenge_downgrades_on_major(
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    test_settings: Settings,
) -> None:
    """One major objection caps the verdict to À VÉRIFIER."""
    client = FakeChatClient([_challenge_json([_major()], False)])
    out = challenge_comparison(
        comparison,
        sale=sale_report,
        delivery=delivery_report,
        settings=test_settings,
        client=client,
    )
    assert out.verdict == "À VÉRIFIER"
    assert out.capped_reason == "avocat du diable : 1 majeure"
    actions = [entry.action for entry in out.trail]
    assert actions[-2:] == ["redteam", "guardrail"]


def test_challenge_counts_minor_objections(
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    test_settings: Settings,
) -> None:
    """Two minors downgrade; a single minor keeps the verdict."""
    client = FakeChatClient([_challenge_json([_minor("Doute un."), _minor("Doute deux.")], False)])
    out = challenge_comparison(
        comparison,
        sale=sale_report,
        delivery=delivery_report,
        settings=test_settings,
        client=client,
    )
    assert out.verdict == "À VÉRIFIER"
    assert out.capped_reason == "avocat du diable : 2 mineures"
    other = comparison.model_copy(update={"justification": "Autre dossier."})
    lone = FakeChatClient([_challenge_json([_minor()], True)])
    kept = challenge_comparison(
        other,
        sale=sale_report,
        delivery=delivery_report,
        settings=test_settings,
        client=lone,
    )
    assert kept.verdict == "CONFORME"
    assert kept.capped_reason is None


def test_challenge_never_recaps_capped_verdict(
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    test_settings: Settings,
) -> None:
    """An already capped verdict keeps its reason; the attack is still logged."""
    capped = comparison.model_copy(
        update={"verdict": "À VÉRIFIER", "capped_reason": "preuve de fraîcheur manquante"}
    )
    client = FakeChatClient([_challenge_json([_major(), _minor()], False)])
    out = challenge_comparison(
        capped,
        sale=sale_report,
        delivery=delivery_report,
        settings=test_settings,
        client=client,
    )
    assert out.verdict == "À VÉRIFIER"
    assert out.capped_reason == "preuve de fraîcheur manquante"
    assert out.trail[-1].action == "redteam"


def test_challenge_skip_keeps_verdict(
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    test_settings: Settings,
) -> None:
    """A failing challenge is logged as skipped; the verdict survives."""
    client = FakeChatClient(
        [ModelError("Le service est injoignable."), ModelError("Toujours injoignable.")]
    )
    received: list[TrailEntry] = []
    out = challenge_comparison(
        comparison,
        sale=sale_report,
        delivery=delivery_report,
        settings=test_settings,
        client=client,
        on_step=received.append,
    )
    assert out.verdict == "CONFORME"
    assert "skipped" in out.trail[-1].summary
    assert [entry.action for entry in received] == ["redteam"]


def test_challenge_emits_new_trail_steps(
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    test_settings: Settings,
) -> None:
    """on_step receives the challenge and the downgrade entries."""
    client = FakeChatClient([_challenge_json([_major()], False)])
    received: list[TrailEntry] = []
    challenge_comparison(
        comparison,
        sale=sale_report,
        delivery=delivery_report,
        settings=test_settings,
        client=client,
        on_step=received.append,
    )
    assert [entry.action for entry in received] == ["redteam", "guardrail"]


def test_redteam_cache_key_covers_verdict(comparison: Comparison) -> None:
    """The verdict and code shape the redteam cache key."""
    base = build_redteam_cache_key(comparison)
    assert base == build_redteam_cache_key(comparison)
    other = comparison.model_copy(update={"verdict": "NON CONFORME"})
    assert base != build_redteam_cache_key(other)


def test_challenge_rejects_too_many_objections(
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    test_settings: Settings,
) -> None:
    """More than 3 objections is invalid; the retry path confirms instead."""
    crowded = _challenge_json([_minor(f"Doute {n}.") for n in range(4)], False)
    client = FakeChatClient([crowded, _challenge_json([], True)])
    out = challenge_comparison(
        comparison,
        sale=sale_report,
        delivery=delivery_report,
        settings=test_settings,
        client=client,
    )
    assert out.verdict == "CONFORME"
    assert len(client.calls) == 2
