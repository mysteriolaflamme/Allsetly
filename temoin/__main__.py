"""Command line: python -m temoin constat photo.jpg --code K7."""

from __future__ import annotations

import argparse
import logging
import sys

from temoin.config import ConfigError
from temoin.images import ImageError
from temoin.model import ModelError
from temoin.report import create_report

logger = logging.getLogger("temoin")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (constat only for now)."""
    parser = argparse.ArgumentParser(prog="temoin", description="Neutral visual witness.")
    sub = parser.add_subparsers(dest="command", required=True)
    constat = sub.add_parser("constat", help="Produce the validated constat of one photo.")
    constat.add_argument("photo", help="Path to the photo, e.g. photo.jpg.")
    constat.add_argument("--code", required=True, help="Challenge code, e.g. K7.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return the exit code."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        report = create_report(args.photo, args.code)
    except (ConfigError, ImageError, ModelError) as error:
        sys.stderr.write(f"{error}\n")
        return 1
    sys.stdout.write(report.model_dump_json(indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
