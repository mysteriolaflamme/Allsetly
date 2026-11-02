"""Meta MMS-TTS voice: number spelling, chunking, tokenization and WAV output."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from temoin.speech import (
    SAMPLE_RATE,
    SpeechError,
    normalize_text,
    number_words,
    sentences,
    synthesize,
    tokenize,
)

VOCAB = {char: index for index, char in enumerate("î abcdefghijklmnopqrstuvwxyzéè-'")}


class FakeSession:
    """Return 0.1 s of signal per call and record the ids it received."""

    def __init__(self) -> None:
        self.calls: list[list[int]] = []

    def run(self, _outputs: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append(feeds["input_ids"][0].tolist())
        assert feeds["attention_mask"].shape == feeds["input_ids"].shape
        return [np.full((1, SAMPLE_RATE // 10), 0.5, dtype=np.float32)]


@pytest.mark.parametrize(
    ("number", "french"),
    [
        (0, "zéro"),
        (17, "dix-sept"),
        (21, "vingt et un"),
        (71, "soixante et onze"),
        (80, "quatre-vingts"),
        (81, "quatre-vingt-un"),
        (91, "quatre-vingt-onze"),
        (100, "cent"),
        (200, "deux cents"),
        (201, "deux cent un"),
        (1000, "mille"),
        (2026, "deux mille vingt-six"),
    ],
)
def test_french_numbers(number: int, french: str) -> None:
    """French spelling follows the usual rules (et un, quatre-vingts, cents)."""
    assert number_words(number, "fr") == french


def test_english_numbers() -> None:
    """English spelling covers tens, hundreds and thousands."""
    assert number_words(62, "en") == "sixty-two"
    assert number_words(100, "en") == "one hundred"
    assert number_words(2026, "en") == "two thousand twenty-six"


def test_scores_and_percentages_are_spelled_out() -> None:
    """Digits and symbols the voice cannot read become words."""
    assert " ".join(normalize_text("Confiance 62/100", "fr").split()) == (
        "Confiance soixante-deux sur cent"
    )
    assert "pour cent" in normalize_text("90%", "fr")
    assert "out of" in normalize_text("3/4", "en")


def test_sentences_split_and_keep_only_known_characters() -> None:
    """Punctuation splits chunks; characters outside the alphabet are dropped."""
    assert sentences("Verdict : CONFORME. Défauts (3) !", "fr", VOCAB) == [
        "verdict",
        "conforme",
        "défauts trois",
    ]


def test_tokenize_interleaves_blanks() -> None:
    """VITS add_blank: a pad id before, between and after every character."""
    assert tokenize("ab", VOCAB, 0) == [0, VOCAB["a"], 0, VOCAB["b"], 0]


def test_synthesize_returns_wav_with_pauses_between_sentences() -> None:
    """Two sentences give two model calls joined by a short pause, as 16 kHz WAV."""
    session = FakeSession()
    audio = synthesize("Bonjour. Au revoir.", "fr", session=session, vocab=VOCAB)
    assert len(session.calls) == 2
    with wave.open(io.BytesIO(audio)) as output:
        assert output.getframerate() == SAMPLE_RATE
        assert output.getnchannels() == 1
        assert output.getnframes() == SAMPLE_RATE // 10 * 2 + SAMPLE_RATE // 4


def test_synthesize_rejects_unknown_language_and_empty_text() -> None:
    """No voice for a language, or nothing readable, is a clear error."""
    with pytest.raises(SpeechError):
        synthesize("Bonjour", "xx", session=FakeSession(), vocab=VOCAB)
    with pytest.raises(SpeechError):
        synthesize("!!! ???", "fr", session=FakeSession(), vocab=VOCAB)
