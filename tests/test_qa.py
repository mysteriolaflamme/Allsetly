"""Tests for case interrogation (offline, fake client)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from temoin.__main__ import main
from temoin.config import Settings
from temoin.qa import IDK_ANSWER, ask_question, build_qa_cache_key
from temoin.schemas import CaseAnswer, Comparison, Report
from temoin.seal import seal_comparison
from tests.fake_client import FakeChatClient


def _answer_json(answer: str, citations: list[str], confident: bool) -> str:
    """Build a case answer payload."""
    return json.dumps({"answer": answer, "citations": citations, "confident": confident})


def _sealed_cert(tmp_path: Path, sale: Report, delivery: Report, comparison: Comparison) -> Path:
    """Seal the fixtures into a certificate file."""
    certificate = seal_comparison(
        comparison, certificate_id="c1", previous_hash=None, sale=sale, delivery=delivery
    )
    path = tmp_path / "c1.json"
    path.write_text(certificate.model_dump_json(), encoding="utf-8")
    return path


def test_ask_returns_grounded_answer(
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    test_settings: Settings,
) -> None:
    """Valid citations pass through with the answer."""
    client = FakeChatClient([_answer_json("Oui, dès la vente.", ["sale:D1", "delivery:D1"], True)])
    out = ask_question(
        "D1 était-il là à la vente ?",
        sale=sale_report,
        delivery=delivery_report,
        comparison=comparison,
        settings=test_settings,
        client=client,
    )
    assert out.answer == "Oui, dès la vente."
    assert out.citations == ["sale:D1", "delivery:D1"]
    assert out.confident is True


def test_ask_retries_unknown_citation(
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    test_settings: Settings,
) -> None:
    """An invented defect id is sent back; the retry answers cleanly."""
    client = FakeChatClient(
        [
            _answer_json("Voir D9.", ["sale:D9"], True),
            _answer_json("Voir D1.", ["sale:D1"], True),
        ]
    )
    out = ask_question(
        "Où est la rayure ?",
        sale=sale_report,
        delivery=delivery_report,
        comparison=comparison,
        settings=test_settings,
        client=client,
    )
    assert out.citations == ["sale:D1"]
    assert len(client.calls) == 2


def test_ask_falls_back_when_ungrounded(
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    test_settings: Settings,
) -> None:
    """Persistent invention degrades to an explicit 'I don't know'."""
    client = FakeChatClient(
        [_answer_json("Voir D9.", ["sale:D9"], True), _answer_json("Voir D8.", ["sale:D8"], True)]
    )
    out = ask_question(
        "Où est la rayure ?",
        sale=sale_report,
        delivery=delivery_report,
        comparison=comparison,
        settings=test_settings,
        client=client,
    )
    assert out.answer == IDK_ANSWER
    assert out.citations == []
    assert out.confident is False


def test_ask_overrides_unconfident_answer(
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    test_settings: Settings,
) -> None:
    """An unsure model is replaced by the honest fallback sentence."""
    client = FakeChatClient([_answer_json("Peut-être bien.", ["sale:D1"], False)])
    out = ask_question(
        "Vaut-il cher ?",
        sale=sale_report,
        delivery=delivery_report,
        comparison=comparison,
        settings=test_settings,
        client=client,
    )
    assert out.answer == IDK_ANSWER
    assert out.citations == []
    assert out.confident is False


def test_ask_requires_citations_when_confident(
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    test_settings: Settings,
) -> None:
    """A confident answer without evidence is sent back for sources."""
    client = FakeChatClient(
        [
            _answer_json("Oui.", [], True),
            _answer_json("Oui, voir le verdict.", ["verdict"], True),
        ]
    )
    out = ask_question(
        "Est-ce conforme ?",
        sale=sale_report,
        delivery=delivery_report,
        comparison=comparison,
        settings=test_settings,
        client=client,
    )
    assert out.citations == ["verdict"]
    assert len(client.calls) == 2


def test_ask_rejects_empty_question(
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    test_settings: Settings,
) -> None:
    """An empty question is a programmer error."""
    client = FakeChatClient([])
    with pytest.raises(ValueError, match="empty"):
        ask_question(
            "  ",
            sale=sale_report,
            delivery=delivery_report,
            comparison=comparison,
            settings=test_settings,
            client=client,
        )
    assert client.calls == []


def test_qa_cache_key_covers_question(
    sale_report: Report, delivery_report: Report, comparison: Comparison
) -> None:
    """The question shapes the QA cache key."""
    base = build_qa_cache_key("Q1 ?", sale_report, delivery_report, comparison)
    assert base == build_qa_cache_key("Q1 ?", sale_report, delivery_report, comparison)
    assert base != build_qa_cache_key("Q2 ?", sale_report, delivery_report, comparison)


def test_ask_command_answers_from_certificate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    test_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ask prints the grounded answer with its sources (offline)."""
    monkeypatch.setattr("temoin.__main__.load_settings", lambda *args: test_settings)
    canned = CaseAnswer(answer="Oui, dès la vente.", citations=["sale:D1"], confident=True)
    monkeypatch.setattr("temoin.__main__.ask_question", lambda *args, **kwargs: canned)
    cert = _sealed_cert(tmp_path, sale_report, delivery_report, comparison)
    assert main(["ask", str(cert), "--question", "D1 ?"]) == 0
    out = capsys.readouterr().out
    assert "Oui, dès la vente." in out
    assert "Sources : sale:D1" in out


def test_ask_command_plays_voice_question(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    test_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ask --audio transcribes first, then answers (offline)."""

    class _Transcript:
        """Transcription stub with fixed text."""

        text = "Le défaut D1 était-il là ?"

    monkeypatch.setattr("temoin.__main__.load_settings", lambda *args: test_settings)
    monkeypatch.setattr("temoin.__main__.transcribe_audio", lambda *args, **kwargs: _Transcript())
    canned = CaseAnswer(answer="Oui.", citations=["verdict"], confident=True)
    monkeypatch.setattr("temoin.__main__.ask_question", lambda *args, **kwargs: canned)
    cert = _sealed_cert(tmp_path, sale_report, delivery_report, comparison)
    audio = tmp_path / "q.wav"
    audio.write_bytes(b"RIFF-fake-wav")
    assert main(["ask", str(cert), "--audio", str(audio)]) == 0
    out = capsys.readouterr().out
    assert "Question entendue" in out
    assert "Sources : verdict" in out


def test_ask_command_needs_question_or_audio(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    test_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ask without input and with missing audio exits 1 with guidance."""
    monkeypatch.setattr("temoin.__main__.load_settings", lambda *args: test_settings)
    cert = _sealed_cert(tmp_path, sale_report, delivery_report, comparison)
    assert main(["ask", str(cert)]) == 1
    assert "--question ou --audio" in capsys.readouterr().err
    assert main(["ask", str(cert), "--audio", str(tmp_path / "nope.wav")]) == 1
    assert "introuvable" in capsys.readouterr().err


def test_ask_command_refuses_tampered_certificate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
    test_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A tampered certificate exits 1 with a plain-language error."""
    monkeypatch.setattr("temoin.__main__.load_settings", lambda *args: test_settings)
    certificate = seal_comparison(
        comparison,
        certificate_id="c1",
        previous_hash=None,
        sale=sale_report,
        delivery=delivery_report,
    )
    cert = tmp_path / "c1.json"
    cert.write_text(
        certificate.model_dump_json().replace('"CONFORME"', '"NON CONFORME"'), encoding="utf-8"
    )
    assert main(["ask", str(cert), "--question", "D1 ?"]) == 1
    assert "MODIFIÉ" in capsys.readouterr().err
