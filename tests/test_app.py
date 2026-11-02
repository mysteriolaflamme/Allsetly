"""Smoke tests for the Streamlit interface (no model calls)."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


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
