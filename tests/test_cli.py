"""Tests for the command-line parser."""

from __future__ import annotations

from temoin.__main__ import build_parser


def test_parser_constat_arguments() -> None:
    """python -m temoin constat photo.jpg --code K7 parses correctly."""
    args = build_parser().parse_args(["constat", "photo.jpg", "--code", "K7"])
    assert (args.command, args.photo, args.code) == ("constat", "photo.jpg", "K7")
