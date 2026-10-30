"""Tests for the command-line parser."""

from __future__ import annotations

from temoin.__main__ import build_parser


def test_parser_constat_arguments() -> None:
    """python -m temoin constat photo.jpg --code K7 parses correctly."""
    args = build_parser().parse_args(["constat", "photo.jpg", "--code", "K7"])
    assert (args.command, args.photo, args.code) == ("constat", "photo.jpg", "K7")


def test_parser_compare_arguments() -> None:
    """compare with both photos and codes parses correctly."""
    args = build_parser().parse_args(
        [
            "compare",
            "vente.jpg",
            "livraison.jpg",
            "--sale-code",
            "K7",
            "--delivery-code",
            "Q9",
        ]
    )
    assert (args.command, args.sale, args.delivery) == ("compare", "vente.jpg", "livraison.jpg")
    assert (args.sale_code, args.delivery_code) == ("K7", "Q9")


def test_parser_compare_codes_are_optional() -> None:
    """compare works without codes (freshness then fails honestly)."""
    args = build_parser().parse_args(["compare", "vente.jpg", "livraison.jpg"])
    assert (args.sale_code, args.delivery_code) == (None, None)
