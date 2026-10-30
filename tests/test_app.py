"""Smoke tests for the Streamlit interface (no model calls)."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def _run_app() -> AppTest:
    """Boot the app in French."""
    app = AppTest.from_file(Path(__file__).parent.parent / "app.py")
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


def test_app_has_four_tabs() -> None:
    """Sale, Delivery, Verify and Video tabs are present."""
    app = _run_app()
    assert [tab.label for tab in app.tabs] == ["Vente", "Livraison", "Vérifier", "Vidéo"]
