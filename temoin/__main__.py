"""Command line: python -m temoin constat photo.jpg --code K7."""

from __future__ import annotations

import argparse
import logging
import sys

from temoin.agent import run_investigation
from temoin.config import ConfigError
from temoin.images import ImageError
from temoin.model import ModelError
from temoin.report import create_report
from temoin.seal import CertificateError, load_certificate, verify_certificate

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
        else:
            result = create_report(args.photo, args.code)
    except (ConfigError, ImageError, ModelError) as error:
        sys.stderr.write(f"{error}\n")
        return 1
    sys.stdout.write(result.model_dump_json(indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
