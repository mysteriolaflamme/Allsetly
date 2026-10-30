"""Meta Model API client: SAM 3.1 segmentation and Voice Transcribe.

SAM follows the official Meta cookbook pattern (``07_sam/01_api_basics``): the
Responses API streams a segmentation grammar that only the official
``meta_sam_parser`` decodes. Voice follows ``06_muse_voice`` (one-shot file
endpoint). This module never parses the SAM grammar itself.

When the Meta service is unavailable, functions raise :class:`MetaAPIError`
with a plain-language French message so callers can degrade gracefully
(constat boxes, typed declaration) instead of inventing results.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import secrets
import struct
import urllib.error
import urllib.request
import wave
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meta_sam_parser import (
    ImageSegmentationResult,
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
)
from openai import AsyncOpenAI
from PIL import Image, ImageDraw

from temoin.config import Settings, load_settings
from temoin.images import decode_data_url

logger = logging.getLogger(__name__)

META_API_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"
VOICE_MODEL = "muse-voice-transcribe-1.0"
VOICE_TRANSCRIBE_PATH = "/asr/transcribe"
VOICE_SAMPLE_RATE = 16000
VOICE_MAX_BYTES = 32 * 1024 * 1024
VOICE_MODE = "PUSH_TO_TALK"
SAM_TIMEOUT_SECONDS = 120.0
VOICE_TIMEOUT_SECONDS = 300.0
_MASK_FILL = (255, 0, 0, 110)
_BOX_OUTLINE = (0, 200, 0)

SamTransport = Callable[[str, str], list[dict[str, Any]]]
"""Replace the SAM network call in tests: (data_url, concept) -> raw events."""

VoiceTransport = Callable[[bytes, str, dict[str, Any]], dict[str, Any]]
"""Replace the voice network call in tests: (wav, name, request) -> response."""


class MetaAPIError(Exception):
    """Raised when a Meta Model API call fails, with a plain-language message."""


@dataclass(frozen=True)
class SegmentMask:
    """One segmented object: box in sent-image pixels plus its binary raster."""

    object_id: str
    left: float
    top: float
    right: float
    bottom: float
    width: int
    height: int
    pixels: bytes
    confidence: float | None = None


@dataclass(frozen=True)
class Segmentation:
    """All masks found for one concept on one sent image."""

    concept: str
    image_width: int
    image_height: int
    masks: tuple[SegmentMask, ...] = ()


@dataclass(frozen=True)
class VoiceTurn:
    """One transcribed turn with its start time and optional speaker label."""

    start_seconds: float
    text: str
    speaker: str | None = None


@dataclass(frozen=True)
class VoiceTranscript:
    """Full transcription of a voice note, with optional turns."""

    text: str
    duration_seconds: float
    turns: tuple[VoiceTurn, ...] = ()


def _meta_key(settings: Settings) -> str:
    """Return the Meta key or raise a plain-language error."""
    if not settings.model_api_key:
        raise MetaAPIError(
            "La clé Meta (MODEL_API_KEY) est manquante. Ajoutez-la dans le "
            "fichier .env à la racine du projet, puis relancez."
        )
    return settings.model_api_key


def _sam_input(concept: str, image_data_url: str) -> list[dict[str, Any]]:
    """Build the Responses input: one noun phrase plus the image data URL."""
    return [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": concept},
                {"type": "input_image", "image_url": image_data_url},
            ],
        }
    ]


async def _collect_live_events(
    *, api_key: str, base_url: str, image_data_url: str, concept: str
) -> list[dict[str, Any]]:
    """Stream the Responses API and return plain JSON event dicts."""
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=SAM_TIMEOUT_SECONDS)
    async with client:
        stream = await client.responses.create(
            model=SAM_MODEL, stream=True, input=_sam_input(concept, image_data_url)
        )
        events: list[dict[str, Any]] = []
        async with stream:
            async for event in stream:
                dumped = event.model_dump(mode="json")
                events.append(dumped if isinstance(dumped, dict) else dict(dumped))
        return events


def _synthesize_lane_closings(events: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Replay the cookbook lane fix: the API streams content_part.done but not
    output_text.done, which the parser needs to close a lane, so synthesize it."""
    completed_lanes: set[tuple[Any, Any, Any]] = set()
    for data in events:
        lane = (data.get("item_id"), data.get("output_index"), data.get("content_index"))
        if data.get("type") == "response.output_text.done":
            if lane not in completed_lanes:
                completed_lanes.add(lane)
                yield data
            continue
        yield data
        part = data.get("part")
        if (
            data.get("type") == "response.content_part.done"
            and lane not in completed_lanes
            and isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
        ):
            completed_lanes.add(lane)
            yield {
                "type": "response.output_text.done",
                "item_id": data.get("item_id"),
                "output_index": data.get("output_index"),
                "content_index": data.get("content_index"),
                "text": part["text"],
            }


def _parse_recorded_events(events: list[dict[str, Any]]) -> ImageSegmentationResult:
    """Run the official parser over live or replayed event dicts."""
    adapted = list(_synthesize_lane_closings(events))

    async def _feed() -> Any:
        for event in adapted:
            yield event

    async def _final() -> ImageSegmentationResult:
        stream: Any = parse_responses_stream(_feed(), image_segmentation_format())
        return await stream.final_result()

    return asyncio.run(_final())


def _validate_result(result: ImageSegmentationResult) -> None:
    """Reject incomplete responses and parser errors like the cookbook does."""
    if result.outcome.status != "completed":
        raise MetaAPIError("La segmentation Meta est restée incomplète : réessayez.")
    errors = [item for item in result.diagnostics if item.severity == "error"]
    if errors:
        raise MetaAPIError("La segmentation Meta a produit des données invalides : réessayez.")


def _describe_sam_failure(error: Exception) -> str:
    """Turn a SAM transport failure into a plain-language French message."""
    status = getattr(error, "status_code", None)
    name = type(error).__name__.lower()
    if status == 401 or "auth" in name:
        return "Le service Meta a refusé la clé (MODEL_API_KEY). Vérifiez-la dans le fichier .env."
    if status == 429 or "ratelimit" in name:
        return "Le service Meta limite le débit : attendez une minute puis réessayez."
    if "timeout" in name:
        return "La segmentation Meta a pris trop de temps : réessayez."
    if "connect" in name:
        return "Impossible de joindre le service Meta : vérifiez votre connexion."
    detail = str(error).strip()
    if detail:
        return f"La segmentation Meta a échoué ({detail[:160]})."
    return "La segmentation Meta a échoué : réessayez."


def segment_image(
    image_data_url: str,
    concept: str,
    *,
    settings: Settings | None = None,
    transport: SamTransport | None = None,
) -> Segmentation:
    """Outline every instance of a concept on the sent image with SAM 3.1."""
    phrase = concept.strip()
    if not phrase:
        raise MetaAPIError("Donnez le nom de l'objet à détourer (par exemple « casque »).")
    if not image_data_url.startswith("data:image/"):
        raise MetaAPIError("La photo fournie pour la segmentation est invalide.")
    active = settings or load_settings()
    api_key = _meta_key(active)
    try:
        if transport is not None:
            events = transport(image_data_url, phrase)
        else:
            events = asyncio.run(
                _collect_live_events(
                    api_key=api_key,
                    base_url=active.model_api_url,
                    image_data_url=image_data_url,
                    concept=phrase,
                )
            )
        result = _parse_recorded_events(events)
        _validate_result(result)
        sent = decode_data_url(image_data_url)
    except MetaAPIError:
        raise
    except Exception as error:
        raise MetaAPIError(_describe_sam_failure(error)) from error
    masks = []
    for record in result.records:
        if record.kind != "mask":
            continue
        raster = decode_mask_to_raster(record.mask)
        masks.append(
            SegmentMask(
                object_id=record.object_id,
                left=float(record.bounds.left),
                top=float(record.bounds.top),
                right=float(record.bounds.right),
                bottom=float(record.bounds.bottom),
                width=record.mask.width,
                height=record.mask.height,
                pixels=raster,
                confidence=record.confidence,
            )
        )
    width, height = sent.size
    logger.info("SAM segmentation: %d mask(s) for %r.", len(masks), phrase)
    return Segmentation(concept=phrase, image_width=width, image_height=height, masks=tuple(masks))


def draw_segmentation(image: Image.Image, segmentation: Segmentation) -> Image.Image:
    """Draw translucent red masks, green boxes and ids on a copy of the image."""
    base = image.convert("RGB")
    width, height = base.size
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for mask in segmentation.masks:
        left = min(max(round(mask.left), 0), width)
        top = min(max(round(mask.top), 0), height)
        right = min(max(round(mask.right), 0), width)
        bottom = min(max(round(mask.bottom), 0), height)
        if right <= left or bottom <= top:
            continue
        cell = Image.new("L", (mask.width, mask.height))
        cell.putdata([255 if value else 0 for value in mask.pixels])
        alpha = cell.resize((right - left, bottom - top), Image.NEAREST)
        alpha = alpha.point(lambda value: _MASK_FILL[3] if value else 0)
        layer = Image.new("RGBA", (right - left, bottom - top), _MASK_FILL[:3] + (255,))
        layer.putalpha(alpha)
        tile = Image.new("RGBA", base.size, (0, 0, 0, 0))
        tile.paste(layer, (left, top))
        overlay = Image.alpha_composite(overlay, tile)
    combined = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(combined)
    pen = max(2, width // 300)
    for mask in segmentation.masks:
        left = min(max(round(mask.left), 0), width)
        top = min(max(round(mask.top), 0), height)
        right = min(max(round(mask.right), 0), width)
        bottom = min(max(round(mask.bottom), 0), height)
        if right <= left or bottom <= top:
            continue
        draw.rectangle([left, top, right, bottom], outline=_BOX_OUTLINE, width=pen)
        draw.text((left + pen, top + pen), str(mask.object_id), fill=_BOX_OUTLINE)
    return combined


def _decode_pcm_samples(frames: bytes, channels: int, width: int) -> list[float]:
    """Decode PCM frames to mono float samples in [-1, 1]."""
    count = len(frames) // (channels * width)
    samples: list[float] = []
    for index in range(count):
        total = 0.0
        for channel in range(channels):
            offset = (index * channels + channel) * width
            chunk = frames[offset : offset + width]
            if width == 1:
                total += (chunk[0] - 128) / 128.0
            elif width == 2:
                total += struct.unpack_from("<h", chunk)[0] / 32768.0
            elif width == 3:
                total += int.from_bytes(chunk, "little", signed=True) / 8388608.0
            else:
                total += struct.unpack_from("<i", chunk)[0] / 2147483648.0
        samples.append(total / channels)
    return samples


def _resample(samples: list[float], source_rate: int) -> list[float]:
    """Resample mono samples to the voice rate with linear interpolation."""
    if source_rate == VOICE_SAMPLE_RATE or not samples:
        return list(samples)
    count = max(1, round(len(samples) * VOICE_SAMPLE_RATE / source_rate))
    if len(samples) == 1 or count == 1:
        return [samples[0]] * count
    resampled: list[float] = []
    for index in range(count):
        position = index * (len(samples) - 1) / (count - 1)
        low = int(position)
        fraction = position - low
        high = min(low + 1, len(samples) - 1)
        resampled.append(samples[low] * (1.0 - fraction) + samples[high] * fraction)
    return resampled


def to_voice_wav(data: bytes) -> bytes:
    """Convert any PCM WAV bytes to mono 16-bit 16 kHz WAV bytes for the API."""
    try:
        reader = wave.open(io.BytesIO(bytes(data)), "rb")
    except (wave.Error, EOFError, OSError) as error:
        raise MetaAPIError(
            "Fichier audio illisible : fournissez une note vocale au format WAV."
        ) from error
    with reader:
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        rate = reader.getframerate()
        frames = reader.getnframes()
        compressed = reader.getcomptype() != "NONE"
        raw_frames = reader.readframes(frames)
    if compressed:
        raise MetaAPIError("Ce WAV est compressé : convertissez-le en WAV non compressé.")
    if width not in (1, 2, 3, 4):
        raise MetaAPIError("Ce WAV utilise un format d'échantillons non géré.")
    if channels <= 0 or rate <= 0 or frames <= 0 or not raw_frames:
        raise MetaAPIError("Ce fichier audio est vide ou invalide.")
    if len(raw_frames) > 64 * 1024 * 1024:
        raise MetaAPIError("Note vocale trop lourde (32 Mo maximum).")
    samples = _resample(_decode_pcm_samples(raw_frames, channels, width), rate)
    packed = struct.pack(
        f"<{len(samples)}h",
        *(max(-32768, min(32767, round(value * 32767))) for value in samples),
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(VOICE_SAMPLE_RATE)
        writer.writeframes(packed)
    return buffer.getvalue()


def _voice_request(
    *, keywords: list[str] | None, language_bias: list[str] | None
) -> dict[str, Any]:
    """Build the JSON part of the one-shot voice request."""
    request: dict[str, Any] = {
        "mode": VOICE_MODE,
        "model": VOICE_MODEL,
        "audioEncoding": "WAV",
    }
    if keywords:
        request["keywords"] = list(keywords)
    if language_bias:
        request["languageBias"] = list(language_bias)
    return request


def _build_multipart(
    *, wav_bytes: bytes, filename: str, request: dict[str, Any]
) -> tuple[bytes, str]:
    """Build a multipart body with a JSON part and the audio part."""
    boundary = "----temoin-" + secrets.token_hex(16)
    body = b""
    body += f"--{boundary}\r\n".encode("ascii")
    body += (
        b'Content-Disposition: form-data; name="request"\r\nContent-Type: application/json\r\n\r\n'
    )
    body += json.dumps(request).encode("utf-8")
    body += f"\r\n--{boundary}\r\n".encode("ascii")
    body += (
        f'Content-Disposition: form-data; name="audio"; filename="{filename}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode("ascii")
    body += wav_bytes
    body += f"\r\n--{boundary}--\r\n".encode("ascii")
    return body, boundary


def _describe_voice_http(status: int | None) -> str:
    """Turn a voice HTTP status into a plain-language French message."""
    if status == 400:
        return "Note vocale trop longue (10 minutes maximum) ou requête mal formée."
    if status == 401:
        return (
            "Le service vocal Meta a refusé la clé (MODEL_API_KEY). "
            "Vérifiez-la dans le fichier .env."
        )
    if status == 404:
        return "Service vocal Meta introuvable : réessayez plus tard."
    if status == 413:
        return "Note vocale trop lourde (32 Mo maximum)."
    if status == 429:
        return "Service vocal Meta saturé : réessayez dans une minute."
    if status == 500:
        return "Le service vocal Meta n'a pas pu traiter cette note."
    if status is None:
        return "Le service vocal Meta n'a pas répondu : réessayez."
    return f"Le service vocal Meta a échoué (erreur {status})."


def _post_voice_response(
    *, api_key: str, base_url: str, wav_bytes: bytes, filename: str, request: dict[str, Any]
) -> dict[str, Any]:
    """POST one recording to the file endpoint and return the JSON response."""
    body, boundary = _build_multipart(wav_bytes=wav_bytes, filename=filename, request=request)
    url = base_url.rstrip("/") + VOICE_TRANSCRIBE_PATH
    http_request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=VOICE_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        raise MetaAPIError(_describe_voice_http(error.code)) from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise MetaAPIError(
            "Impossible de joindre le service vocal Meta : vérifiez votre connexion."
        ) from error
    try:
        payload = json.loads(raw)
    except ValueError as error:
        raise MetaAPIError("Le service vocal Meta a renvoyé une réponse illisible.") from error
    if not isinstance(payload, dict):
        raise MetaAPIError("Le service vocal Meta a renvoyé une réponse illisible.")
    return payload


def _ms_to_seconds(value: Any) -> float:
    """Convert a millisecond count to seconds, defaulting to zero."""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int | float) and value >= 0:
        return float(value) / 1000.0
    return 0.0


def _parse_voice_payload(payload: dict[str, Any]) -> VoiceTranscript:
    """Read a voice response defensively: turns exist only in speech modes."""
    text = payload.get("transcript") or ""
    if not isinstance(text, str):
        text = ""
    turns: list[VoiceTurn] = []
    raw_turns = payload.get("turns") or []
    if isinstance(raw_turns, list):
        for turn in raw_turns:
            if not isinstance(turn, dict):
                continue
            speaker = turn.get("speaker")
            turn_text = turn.get("transcript") or ""
            turns.append(
                VoiceTurn(
                    start_seconds=_ms_to_seconds(turn.get("startMs")),
                    text=turn_text if isinstance(turn_text, str) else "",
                    speaker=speaker if isinstance(speaker, str) else None,
                )
            )
    return VoiceTranscript(
        text=text.strip(),
        duration_seconds=_ms_to_seconds(payload.get("audioDurationMs")),
        turns=tuple(turns),
    )


def transcribe_audio(
    audio: bytes | str | Path,
    *,
    settings: Settings | None = None,
    filename: str = "note.wav",
    language_bias: list[str] | None = None,
    keywords: list[str] | None = None,
    transport: VoiceTransport | None = None,
) -> VoiceTranscript:
    """Transcribe a voice note with Muse Voice Transcribe (one-shot file call)."""
    active = settings or load_settings()
    api_key = _meta_key(active)
    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if not path.is_file():
            raise MetaAPIError(f"Fichier audio introuvable : « {path} ».")
        raw = path.read_bytes()
        name = path.name
    else:
        raw = bytes(audio)
        name = filename
    wav_bytes = to_voice_wav(raw)
    if len(wav_bytes) > VOICE_MAX_BYTES:
        raise MetaAPIError("Note vocale trop lourde (32 Mo maximum).")
    request = _voice_request(keywords=keywords, language_bias=language_bias)
    try:
        if transport is not None:
            payload = transport(wav_bytes, name, request)
        else:
            payload = _post_voice_response(
                api_key=api_key,
                base_url=active.model_api_url,
                wav_bytes=wav_bytes,
                filename=name,
                request=request,
            )
        if not isinstance(payload, dict):
            raise MetaAPIError("Le service vocal Meta a renvoyé une réponse illisible.")
        transcript = _parse_voice_payload(payload)
    except MetaAPIError:
        raise
    except Exception as error:
        detail = str(error).strip()
        if detail:
            raise MetaAPIError(f"La transcription vocale a échoué ({detail[:160]}).") from error
        raise MetaAPIError("La transcription vocale a échoué : réessayez.") from error
    logger.info(
        "Voice transcription: %d chars in %.1fs.",
        len(transcript.text),
        transcript.duration_seconds,
    )
    return transcript


__all__ = [
    "META_API_URL",
    "MetaAPIError",
    "SAM_MODEL",
    "SamTransport",
    "SegmentMask",
    "Segmentation",
    "VOICE_MODEL",
    "VoiceTranscript",
    "VoiceTransport",
    "VoiceTurn",
    "draw_segmentation",
    "segment_image",
    "to_voice_wav",
    "transcribe_audio",
]
