"""Spoken answers with Meta MMS-TTS (open weights), on CPU.

Meta's Massively Multilingual Speech VITS voices, exported to ONNX, read
Témoin's answers aloud without any GPU. The model knows letters, not digits or
punctuation, so text is normalized first (numbers spelled out, one chunk per
sentence with a short pause between chunks). Weights are downloaded once per
language from a pinned revision and checked against their SHA-256.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import logging
import re
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional dependency
    ort = None  # type: ignore[assignment]

from temoin.config import Settings, load_settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
_PAUSE_SECONDS = 0.25
_MAX_CHARS = 700


class SpeechError(Exception):
    """Raised when a spoken answer cannot be produced, with a plain-language message."""


@dataclass(frozen=True)
class Voice:
    """One MMS-TTS voice: pinned Hugging Face revision and file fingerprints."""

    repo: str
    revision: str
    model_sha256: str
    vocab_sha256: str
    pad_token: str


VOICES: dict[str, Voice] = {
    "fr": Voice(
        repo="Xenova/mms-tts-fra",
        revision="c7aa283bf5b599d3c804e4accab07f6394a2b4ae",
        model_sha256="429b27e1cc954d7da4b88e6c73933ad8c3dc033f941e9a04fd2091c1e7ceba41",
        vocab_sha256="d11c7389cb4eec1237a964f18ed1973a47c8b6db6ad0726c065dd2ade6a9c5eb",
        pad_token="î",
    ),
    "en": Voice(
        repo="Xenova/mms-tts-eng",
        revision="3f8955a2adbd6487ce57420620d4916de22b9dac",
        model_sha256="cbfd1e634a7bd1fce6690176ab3bfdb2aeb3b3ed756591f85f654a6669cadd3d",
        vocab_sha256="304f7e71c40d601f9b89d81904609512eebe96b2a941f0a251fc0c9f397bcc45",
        pad_token="k",
    ),
}

_FR_UNITS = (
    "zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize"
).split()
_FR_TENS = {
    2: "vingt",
    3: "trente",
    4: "quarante",
    5: "cinquante",
    6: "soixante",
    8: "quatre-vingt",
}
_EN_UNITS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_EN_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_OUT_OF = {"fr": "sur", "en": "out of"}
_PERCENT = {"fr": "pour cent", "en": "percent"}


def _fr_below_100(n: int) -> str:
    if n < 17:
        return _FR_UNITS[n]
    if n < 20:
        return "dix-" + _FR_UNITS[n - 10]
    tens, unit = divmod(n, 10)
    if tens == 7 and unit == 1:
        return "soixante et onze"
    if tens in (7, 9):
        base = "soixante" if tens == 7 else "quatre-vingt"
        return f"{base}-{_fr_below_100(10 + unit)}"
    base = _FR_TENS[tens]
    if unit == 0:
        return base + ("s" if tens == 8 else "")
    if unit == 1 and tens != 8:
        return f"{base} et un"
    return f"{base}-{_FR_UNITS[unit]}"


def _fr_below_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    if hundreds == 0:
        return _fr_below_100(rest)
    head = "cent" if hundreds == 1 else f"{_FR_UNITS[hundreds]} cent" + ("s" if rest == 0 else "")
    return head if rest == 0 else f"{head} {_fr_below_100(rest)}"


def _en_below_1000(n: int) -> str:
    hundreds, rest = divmod(n, 100)
    if rest < 20:
        tail = _EN_UNITS[rest]
    else:
        tens, unit = divmod(rest, 10)
        tail = _EN_TENS[tens] + (f"-{_EN_UNITS[unit]}" if unit else "")
    if hundreds == 0:
        return tail
    head = f"{_EN_UNITS[hundreds]} hundred"
    return head if rest == 0 else f"{head} {tail}"


def number_words(n: int, lang: str) -> str:
    """Spell out 0..999 999 in French or English; larger numbers digit by digit."""
    if n > 999_999:
        units = _FR_UNITS if lang == "fr" else _EN_UNITS
        return " ".join(units[int(digit)] for digit in str(n))
    below = _fr_below_1000 if lang == "fr" else _en_below_1000
    thousands, rest = divmod(n, 1000)
    if thousands == 0:
        return below(rest)
    if lang == "fr":
        head = "mille" if thousands == 1 else f"{below(thousands)} mille"
    else:
        head = f"{below(thousands)} thousand"
    return head if rest == 0 else f"{head} {below(rest)}"


def normalize_text(text: str, lang: str) -> str:
    """Spell out fractions, percentages and numbers so the voice can read them."""
    text = re.sub(
        r"(\d+)\s*/\s*(\d+)", lambda match: f"{match[1]} {_OUT_OF[lang]} {match[2]}", text
    )
    text = text.replace("%", f" {_PERCENT[lang]}")
    return re.sub(r"\d+", lambda match: f" {number_words(int(match[0]), lang)} ", text)


def sentences(text: str, lang: str, vocab: dict[str, int]) -> list[str]:
    """Split into speakable chunks restricted to the voice's alphabet."""
    chunks = []
    for raw in re.split(r"[.!?;:\n]+", normalize_text(text[:_MAX_CHARS], lang)):
        kept = "".join(char for char in raw.lower() if char in vocab)
        cleaned = re.sub(r"\s+", " ", kept).strip(" -'")
        if cleaned:
            chunks.append(cleaned)
    return chunks


def tokenize(sentence: str, vocab: dict[str, int], pad_id: int) -> list[int]:
    """Map characters to ids with a blank between and around each (VITS add_blank)."""
    ids = [pad_id]
    for char in sentence:
        ids += [vocab[char], pad_id]
    return ids


def _sha256_ok(path: Path, expected: str) -> bool:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


def _fetch(voice: Voice, name: str, target: Path, expected: str) -> Path:
    """Return a checked local file, downloading it once from the pinned revision."""
    if target.is_file() and _sha256_ok(target, expected):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{voice.repo}/resolve/{voice.revision}/{name}"
    logger.info("Downloading Meta MMS voice file %s.", url)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, partial)
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise SpeechError(
            "Voix Meta indisponible : téléchargement impossible. Vérifiez la connexion."
        ) from error
    if not _sha256_ok(partial, expected):
        partial.unlink(missing_ok=True)
        raise SpeechError("Le fichier de voix téléchargé est corrompu : réessayez.")
    partial.replace(target)
    return target


def voice_files(lang: str, settings: Settings) -> tuple[Path, Path]:
    """Return (model.onnx, vocab.json) for a language, downloading them when missing."""
    voice = VOICES[lang]
    folder = Path(settings.data_dir) / "models" / "tts" / lang
    model = _fetch(voice, "onnx/model.onnx", folder / "model.onnx", voice.model_sha256)
    vocab = _fetch(voice, "vocab.json", folder / "vocab.json", voice.vocab_sha256)
    return model, vocab


@functools.lru_cache(maxsize=len(VOICES))
def _session(model_path: str) -> Any:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])


def _to_wav(samples: np.ndarray) -> bytes:
    peak = float(np.abs(samples).max()) or 1.0
    pcm = (np.clip(samples / peak * 0.9, -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())
    return buffer.getvalue()


def synthesize(
    text: str,
    lang: str,
    *,
    settings: Settings | None = None,
    session: Any = None,
    vocab: dict[str, int] | None = None,
) -> bytes:
    """Read text aloud with Meta MMS-TTS; return a 16 kHz mono WAV file."""
    if lang not in VOICES:
        raise SpeechError(f"Pas de voix Meta pour la langue « {lang} ».")
    if np is None or (session is None and ort is None):
        raise SpeechError("La voix Meta demande NumPy et onnxruntime, absents ici.")
    if session is None or vocab is None:
        model_path, vocab_path = voice_files(lang, settings or load_settings())
        session = session or _session(str(model_path))
        vocab = vocab or json.loads(vocab_path.read_text(encoding="utf-8"))
    pad_id = vocab[VOICES[lang].pad_token]
    pause = np.zeros(int(SAMPLE_RATE * _PAUSE_SECONDS), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for sentence in sentences(text, lang, vocab):
        ids = np.array([tokenize(sentence, vocab, pad_id)], dtype=np.int64)
        try:
            waveform = session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
        except Exception as error:
            raise SpeechError("La voix Meta n'a pas pu lire cette réponse.") from error
        pieces += [np.asarray(waveform[0], dtype=np.float32).reshape(-1), pause]
    if not pieces:
        raise SpeechError("Rien à lire à voix haute.")
    return _to_wav(np.concatenate(pieces[:-1]))
