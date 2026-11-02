"""Smoke tests for the Streamlit interface (no model calls)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from temoin import config, qa, speech
from temoin.config import Settings
from temoin.schemas import CaseAnswer, Comparison, Report
from temoin.storage import (
    attach_constat,
    create_transaction,
    issue_delivery_code,
    seal_comparison_in_transaction,
    transaction_path,
)


def _run_app() -> AppTest:
    """Boot the app in French."""
    app = AppTest.from_file(Path(__file__).parent.parent / "app.py", default_timeout=30)
    app.run()
    assert not app.exception
    return app


def test_app_boots_in_french_and_english() -> None:
    """The app loads without error in both languages."""
    app = _run_app()
    assert app.title[0].value == "Témoin — témoin visuel neutre"
    app.sidebar.radio[0].set_value("English").run()
    assert not app.exception
    assert app.title[0].value == "Témoin — neutral visual witness"


def test_app_has_journey_tabs_and_lab() -> None:
    """Numbered Sale, Delivery, Verify journey, plus one advanced-evidence tab."""
    app = _run_app()
    assert [tab.label for tab in app.tabs] == [
        ":material/sell: 1 · Vente",
        ":material/local_shipping: 2 · Livraison",
        ":material/verified: 3 · Vérifier",
        ":material/science: Preuves avancées",
    ]


def test_advanced_evidence_tools_each_render() -> None:
    """Relief, Video and Depth each render from the advanced-evidence switch."""
    app = _run_app()
    for tool in ("video", "depth", "relief"):
        app.button_group(key="lab_tool").set_value(tool).run()
        assert not app.exception


def test_case_pickers_replace_free_text_ids() -> None:
    """Delivery, Verify and Relief offer a case list, with a manual-entry option."""
    app = _run_app()
    labels = [box.label for box in app.selectbox]
    assert "Identifiant du dossier" in labels
    assert "Identifiant du dossier à vérifier" in labels
    for box in app.selectbox:
        if box.label.startswith("Identifiant du dossier"):
            assert box.options[-1] == "Saisir un identifiant (lien reçu)…"


def test_typing_a_challenge_code_as_case_id_warns() -> None:
    """A 2-character challenge code typed as a case id gets a clear warning."""
    app = _run_app()
    picker = next(box for box in app.selectbox if box.key == "del_case")
    picker.set_value("__manual__").run()
    app.text_input(key="del_case_manual").set_value("K4").run()
    assert not app.exception
    assert any("Ceci est un code défi" in warning.value for warning in app.warning)


def test_voice_conversation_answers_and_speaks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    test_settings: Settings,
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
) -> None:
    """Talking to a sealed case shows both turns and reads the answer aloud."""
    settings = dataclasses.replace(test_settings, data_dir=tmp_path / "data")
    tx = create_transaction(settings, sale_code="K7")
    tx = issue_delivery_code(tx, settings, code="Q9")
    tx = attach_constat(tx, "sale", sale_report)
    tx = attach_constat(tx, "delivery", delivery_report)
    tx, _ = seal_comparison_in_transaction(tx, comparison, settings)
    for name in ("sale.jpg", "delivery.jpg"):
        Image.new("RGB", (320, 240), color=(200, 200, 190)).save(
            transaction_path(tx.id, settings) / name
        )
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    monkeypatch.setattr(speech, "synthesize", lambda text, lang: b"RIFF-fake-wav")
    monkeypatch.setattr(
        qa,
        "ask_question",
        lambda question, **_: CaseAnswer(
            answer=f"Réponse : {question}", citations=["S-D1"], confident=True
        ),
    )
    app = AppTest.from_file(Path(__file__).parent.parent / "app.py", default_timeout=30)
    app.session_state["del_tx_id"] = tx.id
    app.run()
    assert not app.exception
    app.chat_input(key=f"talk_in_{tx.id}").set_value("Est-ce le même objet ?").run()
    assert not app.exception
    assert [message.name for message in app.chat_message] == ["user", "assistant"]
    assert app.chat_message[1].markdown[0].value == "Réponse : Est-ce le même objet ?"
    assert len(app.get("audio")) == 1


def test_voice_falls_back_to_browser_when_meta_voice_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    test_settings: Settings,
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
) -> None:
    """Without the Meta voice, the answer is still read by the browser."""
    settings = dataclasses.replace(test_settings, data_dir=tmp_path / "data")
    tx = create_transaction(settings, sale_code="K7")
    tx = issue_delivery_code(tx, settings, code="Q9")
    tx = attach_constat(tx, "sale", sale_report)
    tx = attach_constat(tx, "delivery", delivery_report)
    tx, _ = seal_comparison_in_transaction(tx, comparison, settings)
    for name in ("sale.jpg", "delivery.jpg"):
        Image.new("RGB", (320, 240)).save(transaction_path(tx.id, settings) / name)

    def _no_voice(text: str, lang: str) -> bytes:
        raise speech.SpeechError("offline")

    monkeypatch.setattr(config, "load_settings", lambda: settings)
    monkeypatch.setattr(speech, "synthesize", _no_voice)
    app = AppTest.from_file(Path(__file__).parent.parent / "app.py", default_timeout=30)
    app.session_state["del_tx_id"] = tx.id
    app.run()
    app.button(key="verdict_listen").click().run()
    assert not app.exception
    assert any("speechSynthesis" in str(block.proto) for block in app.get("html"))
