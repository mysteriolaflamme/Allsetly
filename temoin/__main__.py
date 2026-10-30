"""Command line: python -m temoin constat photo.jpg --code K7."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from temoin.agent import run_investigation
from temoin.config import ConfigError, load_settings
from temoin.images import ImageError, decode_data_url, prepare_image
from temoin.meta_api import (
    MetaAPIError,
    draw_segmentation,
    segment_image,
    transcribe_audio,
)
from temoin.model import ModelError
from temoin.report import create_report
from temoin.seal import CertificateError, load_certificate, verify_certificate
from temoin.video import VideoError, create_video_report

logger = logging.getLogger("temoin")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (constat only for now)."""
    parser = argparse.ArgumentParser(prog="temoin", description="Neutral visual witness.")
    sub = parser.add_subparsers(dest="command", required=True)
    constat = sub.add_parser("constat", help="Produce the validated constat of one photo.")
    constat.add_argument("photo", help="Path to the photo, e.g. photo.jpg.")
    constat.add_argument("--code", required=True, help="Challenge code, e.g. K7.")
    compare = sub.add_parser("compare", help="Investigate a sale/delivery pair.")
    compare.add_argument("sale", help="Path to the sale photo, e.g. vente.jpg.")
    compare.add_argument("delivery", help="Path to the delivery photo, e.g. livraison.jpg.")
    compare.add_argument("--sale-code", default=None, help="Code issued for the sale photo.")
    compare.add_argument(
        "--delivery-code", default=None, help="Code issued for the delivery photo."
    )
    verify = sub.add_parser("verify", help="Check a certificate file.")
    verify.add_argument("certificate", help="Path to a certificate file, e.g. cert-1.json.")
    segment = sub.add_parser("segment", help="Outline one object on a photo with SAM 3.1.")
    segment.add_argument("photo", help="Path to the photo, e.g. photo.jpg.")
    segment.add_argument("--concept", required=True, help="Object to outline, e.g. headphones.")
    segment.add_argument("--output", default=None, help="Overlay PNG path.")
    transcribe = sub.add_parser("transcribe", help="Transcribe a voice note.")
    transcribe.add_argument("audio", help="Path to the voice note, e.g. note.wav.")
    transcribe.add_argument("--language-bias", action="append", default=None)
    transcribe.add_argument("--keywords", action="append", default=None)
    video = sub.add_parser("video-constat", help="Fuse a multi-view constat from a video.")
    video.add_argument("video", help="Path to the video, e.g. tour.mp4.")
    video.add_argument("--code", required=True, help="Challenge code, e.g. K7.")
    video.add_argument("--max-frames", type=int, default=None)
    return parser


def verify_main(path: str) -> int:
    """Verify one certificate file, print the outcome and return the exit code."""
    try:
        certificate = load_certificate(path)
    except CertificateError as error:
        sys.stderr.write(f"{error}\n")
        return 1
    intact = verify_certificate(certificate)
    comparison = certificate.comparison
    sale_short = comparison.sale_image_sha256[:8]
    delivery_short = comparison.delivery_image_sha256[:8]
    link = "first link" if certificate.previous_hash is None else "chained to previous"
    lines = [
        f"Certificat : {certificate.certificate_id}",
        f"Verdict scellé : {comparison.verdict} (confiance {comparison.confidence})",
        f"Photos : {sale_short}… / {delivery_short}…",
        f"Scellé le : {certificate.sealed_at_utc.isoformat()}",
        f"Enquête : {len(comparison.trail)} étapes journalisées",
        f"Maillon : {link}",
    ]
    if intact:
        lines.append("Statut : INTACT — ce certificat n'a pas été modifié.")
    else:
        lines.append("Statut : MODIFIÉ — ce certificat ne correspond plus à son sceau.")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0 if intact else 1


def segment_main(photo: str, concept: str, output: str | None) -> int:
    """Segment one photo, save the overlay image and print a summary."""
    settings = load_settings()
    prepared = prepare_image(
        photo,
        max_pixels=settings.image_max_pixels,
        max_megabytes=settings.image_max_megabytes,
        jpeg_quality=settings.image_jpeg_quality,
    )
    segmentation = segment_image(prepared.data_url, concept, settings=settings)
    overlay = draw_segmentation(decode_data_url(prepared.data_url), segmentation)
    out_path = Path(output) if output else Path(photo).parent / (Path(photo).stem + "-sam.png")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        overlay.save(out_path)
    except OSError as error:
        raise ImageError(f"Impossible d'écrire l'image : « {out_path} ».") from error
    count = len(segmentation.masks)
    noun = "masque" if count == 1 else "masques"
    lines = [f"{count} {noun} pour « {segmentation.concept} » -> {out_path}"]
    for mask in segmentation.masks:
        box = [round(mask.left), round(mask.top), round(mask.right), round(mask.bottom)]
        foreground = sum(mask.pixels)
        lines.append(f"  - objet {mask.object_id} : cadre {box}, {foreground} pixels.")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def transcribe_main(audio: str, language_bias: list[str] | None, keywords: list[str] | None) -> int:
    """Transcribe a voice note and print the text with its duration."""
    settings = load_settings()
    transcript = transcribe_audio(
        audio, settings=settings, language_bias=language_bias, keywords=keywords
    )
    if transcript.turns:
        for turn in transcript.turns:
            who = f"{turn.speaker}: " if turn.speaker else ""
            sys.stdout.write(f"[{turn.start_seconds:7.2f}s] {who}{turn.text}\n")
    elif transcript.text:
        sys.stdout.write(transcript.text + "\n")
    else:
        sys.stdout.write("(aucune parole détectée)\n")
    sys.stdout.write(f"Durée : {transcript.duration_seconds:.1f} s\n")
    return 0


def video_main(video: str, code: str, max_frames: int | None) -> int:
    """Analyse a video and print its fused constat."""
    settings = load_settings()
    report = create_video_report(video, code, settings=settings, max_frames=max_frames)
    lines = [
        f"Vues analysées : {len(report.frames)}",
        f"Objet : {report.description}",
        f"Code : lu sur {report.challenge.seen_frames}/{report.challenge.total_frames} vues",
        f"Défauts fusionnés : {len(report.fused_defects)}",
    ]
    for flaw in report.fused_defects:
        noun = "vue" if flaw.confirmations == 1 else "vues"
        lines.append(
            f"  - {flaw.id} ({flaw.type}, {flaw.zone}) : vu sur {flaw.confirmations} {noun}."
        )
    for alert in report.alerts:
        lines.append(f"Alerte : {alert.message}")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the exit code."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        return verify_main(args.certificate)
    try:
        if args.command == "compare":
            result = run_investigation(
                args.sale,
                args.delivery,
                sale_code=args.sale_code,
                delivery_code=args.delivery_code,
            )
        elif args.command == "segment":
            return segment_main(args.photo, args.concept, args.output)
        elif args.command == "transcribe":
            return transcribe_main(args.audio, args.language_bias, args.keywords)
        elif args.command == "video-constat":
            return video_main(args.video, args.code, args.max_frames)
        else:
            result = create_report(args.photo, args.code)
    except (ConfigError, ImageError, ModelError, MetaAPIError, VideoError) as error:
        sys.stderr.write(f"{error}\n")
        return 1
    sys.stdout.write(result.model_dump_json(indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
