"""Command line: python -m temoin constat photo.jpg --code K7."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image

from temoin.agent import run_investigation
from temoin.config import ConfigError, load_settings
from temoin.evidence import build_board, rows_from_reports
from temoin.forensics import analyze_photo, ela_map
from temoin.images import ImageError, decode_data_url, prepare_image
from temoin.meta_api import (
    MetaAPIError,
    draw_segmentation,
    segment_image,
    transcribe_audio,
)
from temoin.model import ModelError
from temoin.monodepth import MonoDepthError, estimate_depth
from temoin.report import create_report
from temoin.seal import CertificateError, load_certificate, verify_certificate
from temoin.stereo import (
    DepthMatrix,
    StereoError,
    compute_depth,
    defect_relief,
    depth_from_video,
    load_pair,
    relief_cloud,
    relief_wiggle,
)
from temoin.video import (
    SUPPORTED_VIDEO_SUFFIXES,
    VideoError,
    create_video_report,
    extract_frames,
)

logger = logging.getLogger("temoin")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (constat only for now)."""
    parser = argparse.ArgumentParser(prog="temoin", description="Neutral visual witness.")
    sub = parser.add_subparsers(dest="command", required=True)
    constat = sub.add_parser("constat", help="Produce the validated constat of one photo.")
    constat.add_argument("photo", help="Path to the photo, e.g. photo.jpg.")
    constat.add_argument("--code", required=True, help="Challenge code, e.g. K7.")
    constat.add_argument("--materials", default=None, help="Declared materials, e.g. 'cuir,bois'.")
    compare = sub.add_parser("compare", help="Investigate a sale/delivery pair.")
    compare.add_argument("sale", help="Path to the sale photo, e.g. vente.jpg.")
    compare.add_argument("delivery", help="Path to the delivery photo, e.g. livraison.jpg.")
    compare.add_argument("--sale-code", default=None, help="Code issued for the sale photo.")
    compare.add_argument(
        "--delivery-code", default=None, help="Code issued for the delivery photo."
    )
    compare.add_argument(
        "--no-redteam",
        dest="redteam",
        action="store_false",
        default=True,
        help="Skip the devil's advocate challenge.",
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
    forensics = sub.add_parser("forensics", help="Check a photo for retouching and quality.")
    forensics.add_argument("photo", help="Path to the photo, e.g. photo.jpg.")
    forensics.add_argument("--ela-map", default=None, help="Save the ELA map image here.")
    stereo = sub.add_parser("stereo", help="Measure relief from two views (Depth Matrix).")
    stereo.add_argument("left", help="Path to the left view, e.g. frame-00.jpg.")
    stereo.add_argument("right", help="Path to the right view, e.g. frame-01.jpg.")
    stereo.add_argument("--wiggle", default=None, help="Save the parallax GIF here.")
    stereo.add_argument("--cloud", default=None, help="Save the 3D cloud HTML here.")
    stereo.add_argument("--bbox", default=None, help="Defect box 0-1000, e.g. 400,400,600,600.")
    vdepth = sub.add_parser("video-depth", help="Measure relief from a walkaround video.")
    vdepth.add_argument("video", help="Path to the video, e.g. tour.mp4.")
    vdepth.add_argument("--wiggle", default=None, help="Save the parallax GIF here.")
    vdepth.add_argument("--cloud", default=None, help="Save the 3D cloud HTML here.")
    vdepth.add_argument("--bbox", default=None, help="Defect box 0-1000, e.g. 400,400,600,600.")
    vdepth.add_argument("--max-frames", type=int, default=None)
    mono = sub.add_parser("mono-depth", help="Estimate relief from one photo or video (AI).")
    mono.add_argument("source", help="Photo or video, e.g. photo.jpg or tour.mp4.")
    mono.add_argument("--wiggle", default=None, help="Save the parallax GIF here.")
    mono.add_argument("--cloud", default=None, help="Save the 3D cloud HTML here.")
    mono.add_argument("--bbox", default=None, help="Defect box 0-1000, e.g. 400,400,600,600.")
    board = sub.add_parser("board", help="Render the evidence board of a sealed certificate.")
    board.add_argument("certificate", help="Path to a certificate file, e.g. cert-1.json.")
    board.add_argument("--sale", required=True, help="Path to the sale photo.")
    board.add_argument("--delivery", required=True, help="Path to the delivery photo.")
    board.add_argument("--output", default=None, help="Board PNG path.")
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


def forensics_main(photo: str, ela_map_path: str | None) -> int:
    """Analyse a photo and print retouch signals with quality coaching."""
    settings = load_settings()
    report = analyze_photo(photo, settings=settings)
    lines = []
    if report.suspicious:
        lines.append("Photo suspecte :")
        lines.extend(f"  - {signal}" for signal in report.signals)
    else:
        lines.append("Photo saine.")
    if report.quality.advice:
        lines.append("Qualité :")
        lines.extend(f"  - {tip}" for tip in report.quality.advice)
    else:
        lines.append(f"Netteté : {report.quality.sharpness:.1f} (bonne).")
    if ela_map_path:
        with Image.open(photo) as handle:
            handle.load()
            ela_map(handle.convert("RGB"), quality=settings.forensics_ela_quality).save(
                ela_map_path
            )
        lines.append(f"Carte ELA : {ela_map_path}")
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def _depth_output_lines(
    first: Image.Image,
    matrix: DepthMatrix,
    wiggle: str | None,
    cloud: str | None,
    bbox: str | None,
    fallback: Path,
) -> list[str]:
    """Describe one relief box, save viewers and return the output lines."""
    lines = []
    if bbox:
        try:
            box = [int(item) for item in bbox.split(",")]
            if len(box) != 4:
                raise ValueError(box)
        except ValueError:
            raise StereoError("Cadre invalide : utilisez x1,y1,x2,y2 entre 0 et 1000.") from None
        stats = defect_relief(matrix, box)
        lines.append(
            f"Zone : relief {stats.inside:.2f}, "
            f"alentours {stats.ring:.2f} (écart {stats.delta:+.2f})."
        )
    wiggle_path = Path(wiggle) if wiggle else fallback
    wiggle_path.write_bytes(relief_wiggle(first, matrix))
    lines.append(f"Animation : {wiggle_path}")
    if cloud:
        relief_cloud(first, matrix).write_html(cloud, include_plotlyjs="cdn")
        lines.append(f"Nuage 3D : {cloud}")
    return lines


def stereo_main(
    left: str, right: str, wiggle: str | None, cloud: str | None, bbox: str | None
) -> int:
    """Measure relief from two views and save the parallax viewers."""
    settings = load_settings()
    first, second = load_pair(left, right)
    matrix = compute_depth(first, second, settings=settings)
    lines = [
        f"Vues : {left} / {right}",
        f"Relief mesuré sur {matrix.coverage * 100:.0f} % de l'image.",
    ]
    fallback = Path(left).parent / (Path(left).stem + "-relief.gif")
    lines.extend(_depth_output_lines(first, matrix, wiggle, cloud, bbox, fallback))
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def video_depth_main(
    video: str,
    wiggle: str | None,
    cloud: str | None,
    bbox: str | None,
    max_frames: int | None,
) -> int:
    """Measure relief from a walkaround video and save the parallax viewers."""
    settings = load_settings()
    matrix, first = depth_from_video(video, settings=settings, max_frames=max_frames)
    lines = [
        f"Vidéo : {video}",
        f"Vues utilisées : {matrix.left_index} et {matrix.right_index}.",
        f"Relief mesuré sur {matrix.coverage * 100:.0f} % de l'image.",
    ]
    fallback = Path(video).parent / (Path(video).stem + "-relief.gif")
    lines.extend(_depth_output_lines(first, matrix, wiggle, cloud, bbox, fallback))
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def mono_depth_main(source: str, wiggle: str | None, cloud: str | None, bbox: str | None) -> int:
    """Estimate AI relief from one photo or video and save the viewers."""
    settings = load_settings()
    path = Path(source)
    if path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES:
        frames = extract_frames(
            source,
            max_frames=1,
            max_pixels=settings.image_max_pixels,
            max_megabytes=settings.video_max_megabytes,
        )
        image = frames[0].image
        label = f"{source} (vue {frames[0].index})"
    else:
        prepared = prepare_image(
            source,
            max_pixels=settings.image_max_pixels,
            max_megabytes=settings.image_max_megabytes,
            jpeg_quality=settings.image_jpeg_quality,
        )
        image = decode_data_url(prepared.data_url)
        label = source
    matrix = estimate_depth(image, settings=settings)
    lines = [
        f"Vue : {label}",
        "Méthode : IA monoculaire (estimation indicative).",
    ]
    fallback = path.parent / (path.stem + "-mono-relief.gif")
    lines.extend(_depth_output_lines(image, matrix, wiggle, cloud, bbox, fallback))
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def board_main(certificate: str, sale: str, delivery: str, output: str | None) -> int:
    """Render the evidence board of an intact certificate and print its path."""
    certificate_data = load_certificate(certificate)
    if not verify_certificate(certificate_data):
        raise CertificateError("Certificat MODIFIÉ : planche refusée.")
    try:
        with Image.open(sale) as first, Image.open(delivery) as second:
            first.load()
            second.load()
            sale_image, delivery_image = first.convert("RGB"), second.convert("RGB")
    except (OSError, ValueError) as error:
        raise ImageError(f"Impossible de lire les photos « {sale} » et « {delivery} ».") from error
    rows = rows_from_reports(
        certificate_data.sale_constat,
        certificate_data.delivery_constat,
        certificate_data.comparison,
    )
    board = build_board(sale_image, delivery_image, rows)
    out_path = (
        Path(output)
        if output
        else Path(certificate).parent / (Path(certificate).stem + "-board.png")
    )
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        board.save(out_path)
    except OSError as error:
        raise ImageError(f"Impossible d'écrire l'image : « {out_path} ».") from error
    comparison = certificate_data.comparison
    sys.stdout.write(
        f"Planche : {len(rows)} correspondance(s), verdict {comparison.verdict} -> {out_path}\n"
    )
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
                redteam=args.redteam,
            )
        elif args.command == "segment":
            return segment_main(args.photo, args.concept, args.output)
        elif args.command == "transcribe":
            return transcribe_main(args.audio, args.language_bias, args.keywords)
        elif args.command == "video-constat":
            return video_main(args.video, args.code, args.max_frames)
        elif args.command == "forensics":
            return forensics_main(args.photo, args.ela_map)
        elif args.command == "stereo":
            return stereo_main(args.left, args.right, args.wiggle, args.cloud, args.bbox)
        elif args.command == "video-depth":
            return video_depth_main(args.video, args.wiggle, args.cloud, args.bbox, args.max_frames)
        elif args.command == "mono-depth":
            return mono_depth_main(args.source, args.wiggle, args.cloud, args.bbox)
        elif args.command == "board":
            return board_main(args.certificate, args.sale, args.delivery, args.output)
        else:
            declared = [item.strip() for item in (args.materials or "").split(",") if item.strip()]
            result = create_report(args.photo, args.code, declared_materials=declared)
    except (
        CertificateError,
        ConfigError,
        ImageError,
        ModelError,
        MetaAPIError,
        MonoDepthError,
        StereoError,
        VideoError,
    ) as error:
        sys.stderr.write(f"{error}\n")
        return 1
    sys.stdout.write(result.model_dump_json(indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
