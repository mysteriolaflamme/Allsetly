"""Tests for the FR/EN interface strings."""

from __future__ import annotations

import re
from pathlib import Path

from temoin.i18n import STRINGS, VERDICTS, t, verdict_label


def test_every_string_has_french_and_english() -> None:
    """No missing or empty translation."""
    assert len(STRINGS) > 0
    for key, entry in STRINGS.items():
        assert entry.get("fr"), f"missing French for {key}"
        assert entry.get("en"), f"missing English for {key}"


def test_verdicts_cover_all_cases() -> None:
    """The four verdicts translate both ways."""
    for verdict in ("CONFORME", "ENDOMMAGÉ", "À VÉRIFIER", "NON CONFORME"):
        assert VERDICTS[verdict]["fr"]
        assert VERDICTS[verdict]["en"]


def test_lookup_falls_back_gracefully() -> None:
    """Unknown keys and languages degrade to French, then the key."""
    assert t("app_title", "en") != t("app_title", "fr")
    assert t("app_title", "de") == t("app_title", "fr")
    assert t("no_such_key", "fr") == "no_such_key"
    assert verdict_label("CONFORME", "en") == "MATCH"
    assert verdict_label("???", "en") == "???"


def test_app_uses_only_defined_keys() -> None:
    """Every t("...") literal in app.py exists in STRINGS."""
    source = (Path(__file__).parent.parent / "app.py").read_text(encoding="utf-8")
    used = set(re.findall(r'\bt\("([a-z_]+)"[,)]', source))
    assert used, "no UI keys found in app.py"
    unknown = used - set(STRINGS)
    assert not unknown, f"undefined UI keys: {sorted(unknown)}"
