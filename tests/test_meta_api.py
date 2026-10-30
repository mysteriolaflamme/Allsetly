"""Tests for the Meta Model API client (offline replay, no network)."""

from __future__ import annotations

import base64
import io
import json
import struct
import urllib.error
import wave
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from temoin.config import Settings
from temoin.meta_api import (
    MetaAPIError,
    Segmentation,
    SegmentMask,
    VoiceTurn,
    draw_segmentation,
    segment_image,
    to_voice_wav,
    transcribe_audio,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict[str, Any]:
    """Load a recorded Meta API response."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def meta_settings(test_settings: Settings) -> Settings:
    """Return settings with a fake Meta key."""
    return replace(test_settings, model_api_key="meta-test-key")


def _data_url(width: int, height: int) -> str:
    """Build a blank JPEG data URL of the given size."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buffer, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _wav_bytes(
    *, channels: int = 1, width: int = 2, rate: int = 22050, frames: int = 1000
) -> bytes:
    """Build a small deterministic PCM WAV."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        total = frames * channels
        if width == 1:
            writer.writeframes(bytes((index * 7) % 256 for index in range(total)))
        else:
            samples = [(index * 131) % 65536 - 32768 for index in range(total)]
            writer.writeframes(struct.pack(f"<{total}h", *samples))
    return buffer.getvalue()


def _wav_params(data: bytes) -> tuple[int, int, int, int]:
    """Return (channels, width, rate, frames) of WAV bytes."""
    with wave.open(io.BytesIO(data), "rb") as reader:
        return (
            reader.getnchannels(),
            reader.getsampwidth(),
            reader.getframerate(),
            reader.getnframes(),
        )


def _box_mask(
    *,
    object_id: str = "0",
    box: tuple[float, float, float, float] = (5.0, 5.0, 25.0, 20.0),
    size: tuple[int, int] = (20, 15),
) -> SegmentMask:
    """Build an all-foreground synthetic mask."""
    width, height = size
    return SegmentMask(
        object_id=object_id,
        left=box[0],
        top=box[1],
        right=box[2],
        bottom=box[3],
        width=width,
        height=height,
        pixels=bytes([1]) * (width * height),
    )


class _StatusError(Exception):
    """Stub transport failure carrying an HTTP status."""

    def __init__(self, status_code: int) -> None:
        """Store the status code."""
        super().__init__("remote exploded")
        self.status_code = status_code


class _FakeResponse:
    """Minimal urlopen response for voice tests."""

    def __init__(self, body: bytes) -> None:
        """Store the response body."""
        self._body = body

    def read(self) -> bytes:
        """Return the response body."""
        return self._body

    def __enter__(self) -> _FakeResponse:
        """Enter the response context."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Leave the response context."""
        return None


def test_segment_replays_recorded_masks(meta_settings: Settings) -> None:
    """A recorded SAM stream replays to the same boxes through the real parser."""
    recorded = _fixture("sam_headphones_events.json")
    seen: dict[str, Any] = {}

    def fake_transport(data_url: str, concept: str) -> list[dict[str, Any]]:
        seen["data_url"] = data_url
        seen["concept"] = concept
        return recorded["events"]

    url = _data_url(recorded["width"], recorded["height"])
    segmentation = segment_image(
        url, "headphones", settings=meta_settings, transport=fake_transport
    )
    assert seen == {"data_url": url, "concept": "headphones"}
    assert segmentation.concept == "headphones"
    assert (segmentation.image_width, segmentation.image_height) == (
        recorded["width"],
        recorded["height"],
    )
    assert len(segmentation.masks) == 1
    mask = segmentation.masks[0]
    assert mask.object_id == "0"
    assert 0 <= mask.left < mask.right <= recorded["width"]
    assert 0 <= mask.top < mask.bottom <= recorded["height"]
    assert len(mask.pixels) == mask.width * mask.height
    assert any(mask.pixels)


def test_segment_empty_match_is_valid(meta_settings: Settings) -> None:
    """A completed stream without records is an empty match, not an error."""
    events = [
        {
            "type": "response.output_text.done",
            "item_id": "m1",
            "output_index": 0,
            "content_index": 0,
            "text": "nothing found",
        },
        {"type": "response.completed"},
    ]
    segmentation = segment_image(
        _data_url(100, 80), "unicorn", settings=meta_settings, transport=lambda d, c: events
    )
    assert segmentation.masks == ()
    assert (segmentation.image_width, segmentation.image_height) == (100, 80)


def test_segment_rejects_truncated_stream(meta_settings: Settings) -> None:
    """An empty event list is an incomplete response, not an empty match."""
    with pytest.raises(MetaAPIError, match="incomplète"):
        segment_image(_data_url(10, 10), "x", settings=meta_settings, transport=lambda d, c: [])


def test_segment_requires_meta_key(test_settings: Settings) -> None:
    """A missing Meta key names MODEL_API_KEY, never the value."""
    with pytest.raises(MetaAPIError, match="MODEL_API_KEY"):
        segment_image(_data_url(10, 10), "x", settings=test_settings)


def test_segment_rejects_blank_concept(meta_settings: Settings) -> None:
    """A blank concept is rejected before any network call."""
    with pytest.raises(MetaAPIError, match="nom de l'objet"):
        segment_image(_data_url(10, 10), "  ", settings=meta_settings)


def test_segment_rejects_bad_data_url(meta_settings: Settings) -> None:
    """A non-image data URL is rejected before any network call."""
    with pytest.raises(MetaAPIError, match="invalide"):
        segment_image("not-a-url", "x", settings=meta_settings)


@pytest.mark.parametrize(
    ("error", "fragment"),
    [
        (_StatusError(401), "refusé"),
        (_StatusError(429), "débit"),
        (TimeoutError("slow"), "trop de temps"),
        (ConnectionError("down"), "joindre"),
        (RuntimeError("boom"), "boom"),
    ],
)
def test_segment_maps_transport_failures(
    meta_settings: Settings, error: Exception, fragment: str
) -> None:
    """Transport failures become plain-language French errors."""

    def boom(data_url: str, concept: str) -> list[dict[str, Any]]:
        raise error

    with pytest.raises(MetaAPIError, match=fragment):
        segment_image(_data_url(10, 10), "x", settings=meta_settings, transport=boom)


def test_draw_segmentation_paints_mask_and_box() -> None:
    """The overlay tints the mask red and leaves the background alone."""
    image = Image.new("RGB", (40, 30), color=(255, 255, 255))
    segmentation = Segmentation(concept="x", image_width=40, image_height=30, masks=(_box_mask(),))
    overlay = draw_segmentation(image, segmentation)
    assert overlay.size == (40, 30)
    red, green, blue = overlay.getpixel((15, 12))
    assert red > 200 and green < 200 and blue < 200
    assert overlay.getpixel((35, 25)) == (255, 255, 255)


def test_draw_segmentation_clamps_wild_boxes() -> None:
    """Out-of-range and degenerate boxes never crash the overlay."""
    image = Image.new("RGB", (40, 30), color=(1, 2, 3))
    wild = _box_mask(box=(-500.0, -500.0, 500.0, 500.0), size=(4, 4))
    flat = _box_mask(box=(5.0, 5.0, 5.0, 10.0), size=(4, 4))
    segmentation = Segmentation(concept="x", image_width=40, image_height=30, masks=(wild, flat))
    assert draw_segmentation(image, segmentation).size == (40, 30)


def test_transcribe_replays_recorded_response(meta_settings: Settings) -> None:
    """A recorded voice response replays to the same transcript."""
    payload = _fixture("voice_declaration_response.json")
    seen: dict[str, Any] = {}

    def fake_transport(wav: bytes, name: str, request: dict[str, Any]) -> dict[str, Any]:
        seen["wav"] = wav
        seen["name"] = name
        seen["request"] = request
        return payload

    transcript = transcribe_audio(
        _wav_bytes(),
        settings=meta_settings,
        filename="note.wav",
        language_bias=["french"],
        transport=fake_transport,
    )
    assert transcript.text == str(payload["transcript"]).strip()
    assert transcript.duration_seconds == float(payload["audioDurationMs"]) / 1000.0
    assert transcript.turns == ()
    assert _wav_params(seen["wav"])[:3] == (1, 2, 16000)
    assert seen["name"] == "note.wav"
    assert seen["request"] == {
        "mode": "PUSH_TO_TALK",
        "model": "muse-voice-transcribe-1.0",
        "audioEncoding": "WAV",
        "languageBias": ["french"],
    }


def test_transcribe_parses_turns_defensively(meta_settings: Settings) -> None:
    """Turns are parsed with defaults; junk entries are skipped."""
    payload = {
        "transcript": "  hello world ",
        "audioDurationMs": 1500,
        "turns": [
            {"speaker": "A", "startMs": 1520, "transcript": "How is the weather?"},
            {"startMs": 5900, "transcript": "It is raining."},
            {"speaker": 7, "startMs": "bad", "transcript": None},
            "junk",
        ],
    }
    transcript = transcribe_audio(
        _wav_bytes(), settings=meta_settings, transport=lambda w, n, r: payload
    )
    assert transcript.text == "hello world"
    assert transcript.duration_seconds == 1.5
    assert transcript.turns == (
        VoiceTurn(start_seconds=1.52, text="How is the weather?", speaker="A"),
        VoiceTurn(start_seconds=5.9, text="It is raining.", speaker=None),
        VoiceTurn(start_seconds=0.0, text="", speaker=None),
    )


def test_transcribe_requires_meta_key(test_settings: Settings) -> None:
    """A missing Meta key names MODEL_API_KEY, never the value."""
    with pytest.raises(MetaAPIError, match="MODEL_API_KEY"):
        transcribe_audio(_wav_bytes(), settings=test_settings)


def test_transcribe_rejects_missing_file(meta_settings: Settings, tmp_path: Path) -> None:
    """A missing audio path is a plain-language error."""
    with pytest.raises(MetaAPIError, match="introuvable"):
        transcribe_audio(str(tmp_path / "nope.wav"), settings=meta_settings)


def test_transcribe_rejects_unreadable_audio(meta_settings: Settings) -> None:
    """Non-WAV bytes are rejected before any network call."""
    with pytest.raises(MetaAPIError, match="illisible"):
        transcribe_audio(b"junk-bytes-here", settings=meta_settings)


def test_transcribe_wraps_transport_failure(meta_settings: Settings) -> None:
    """Transport crashes become plain-language errors."""

    def boom(wav: bytes, name: str, request: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    with pytest.raises(MetaAPIError, match="boom"):
        transcribe_audio(_wav_bytes(), settings=meta_settings, transport=boom)


def test_transcribe_rejects_non_dict_payload(meta_settings: Settings) -> None:
    """A non-JSON-object response is an unreadable answer."""
    with pytest.raises(MetaAPIError, match="illisible"):
        transcribe_audio(_wav_bytes(), settings=meta_settings, transport=lambda w, n, r: [])


@pytest.mark.parametrize(
    ("status", "fragment"),
    [
        (400, "trop longue"),
        (401, "refusé"),
        (404, "introuvable"),
        (413, "trop lourde"),
        (429, "saturé"),
        (500, "pas pu traiter"),
        (503, "erreur 503"),
    ],
)
def test_transcribe_maps_http_errors(
    monkeypatch: pytest.MonkeyPatch, meta_settings: Settings, status: int, fragment: str
) -> None:
    """Voice HTTP statuses become plain-language French errors."""

    def fake_urlopen(request: Any, timeout: Any = None) -> _FakeResponse:
        raise urllib.error.HTTPError(request.full_url, status, "x", {}, io.BytesIO())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(MetaAPIError, match=fragment):
        transcribe_audio(_wav_bytes(), settings=meta_settings)


def test_transcribe_maps_connection_failure(
    monkeypatch: pytest.MonkeyPatch, meta_settings: Settings
) -> None:
    """An unreachable voice endpoint names the connection, not the stack."""

    def fake_urlopen(request: Any, timeout: Any = None) -> _FakeResponse:
        raise urllib.error.URLError("down")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(MetaAPIError, match="joindre"):
        transcribe_audio(_wav_bytes(), settings=meta_settings)


@pytest.mark.parametrize("body", [b"nope", b"[1]"])
def test_transcribe_rejects_unreadable_response(
    monkeypatch: pytest.MonkeyPatch, meta_settings: Settings, body: bytes
) -> None:
    """A non-JSON-object body is an unreadable answer."""

    def fake_urlopen(request: Any, timeout: Any = None) -> _FakeResponse:
        return _FakeResponse(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(MetaAPIError, match="illisible"):
        transcribe_audio(_wav_bytes(), settings=meta_settings)


def test_transcribe_posts_json_part_and_audio(
    monkeypatch: pytest.MonkeyPatch, meta_settings: Settings
) -> None:
    """The file endpoint receives a JSON part plus the WAV part with a key."""
    seen: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: Any = None) -> _FakeResponse:
        seen["url"] = request.full_url
        seen["data"] = request.data
        seen["headers"] = dict(request.header_items())
        return _FakeResponse(b'{"transcript": "ok", "audioDurationMs": 1000}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    transcript = transcribe_audio(
        _wav_bytes(), settings=meta_settings, filename="note.wav", keywords=["casque"]
    )
    assert transcript.text == "ok"
    assert seen["url"] == "https://api.meta.ai/v1/asr/transcribe"
    body = seen["data"]
    assert b'name="request"' in body
    assert b'"audioEncoding": "WAV"' in body
    assert b'"keywords": ["casque"]' in body
    assert b'name="audio"; filename="note.wav"' in body
    content_type = seen["headers"].get("Content-type", "")
    assert content_type.startswith("multipart/form-data; boundary=")
    boundary = content_type.split("boundary=")[1].encode("ascii")
    assert body.startswith(b"--" + boundary)
    assert body.rstrip().endswith(b"--" + boundary + b"--")
    assert seen["headers"].get("Authorization") == "Bearer meta-test-key"


def test_transcribe_reads_audio_file(tmp_path: Path, meta_settings: Settings) -> None:
    """Audio paths are read and their file name is sent."""
    path = tmp_path / "memo.wav"
    path.write_bytes(_wav_bytes())
    seen: dict[str, Any] = {}

    def fake_transport(wav: bytes, name: str, request: dict[str, Any]) -> dict[str, Any]:
        seen["name"] = name
        return {"transcript": "file ok", "audioDurationMs": 500}

    transcript = transcribe_audio(str(path), settings=meta_settings, transport=fake_transport)
    assert transcript.text == "file ok"
    assert seen["name"] == "memo.wav"


def test_to_voice_wav_converts_to_16k_mono() -> None:
    """Stereo 8-bit 8 kHz audio becomes mono 16-bit 16 kHz."""
    converted = to_voice_wav(_wav_bytes(channels=2, width=1, rate=8000, frames=800))
    assert _wav_params(converted) == (1, 2, 16000, 1600)


def test_to_voice_wav_keeps_speech_length() -> None:
    """Resampling preserves the audio duration."""
    converted = to_voice_wav(_wav_bytes(channels=1, width=2, rate=22050, frames=2205))
    assert _wav_params(converted) == (1, 2, 16000, 1600)


def test_to_voice_wav_rejects_garbage_and_empty() -> None:
    """Non-WAV and empty audio are plain-language errors."""
    with pytest.raises(MetaAPIError, match="illisible"):
        to_voice_wav(b"junk-bytes-here")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
    with pytest.raises(MetaAPIError, match="vide ou invalide"):
        to_voice_wav(buffer.getvalue())
