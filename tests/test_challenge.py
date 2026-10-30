"""Tests for challenge code generation, format and expiry (no network)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from temoin.challenge import generate_code, is_expired, is_well_formed
from temoin.config import Settings


def test_generate_code_uses_alphabet_only(test_settings: Settings) -> None:
    """Drawn codes have the right length and only unambiguous characters."""
    for _ in range(50):
        code = generate_code(
            length=test_settings.challenge_code_length,
            alphabet=test_settings.challenge_alphabet,
        )
        assert len(code) == test_settings.challenge_code_length
        assert all(char in test_settings.challenge_alphabet for char in code)


def test_generate_code_rejects_bad_parameters() -> None:
    """Invalid length or alphabet raise immediately."""
    with pytest.raises(ValueError):
        generate_code(length=0, alphabet="AB")
    with pytest.raises(ValueError):
        generate_code(length=2, alphabet="")


def test_alphabet_has_no_ambiguous_characters(test_settings: Settings) -> None:
    """O/0, I/1, S and 5 are excluded from the alphabet."""
    alphabet = test_settings.challenge_alphabet
    for char in ("O", "0", "I", "1", "S", "5"):
        assert char not in alphabet


def test_is_well_formed_accepts_generated_codes(test_settings: Settings) -> None:
    """Format check accepts drawn codes, ignoring case and spaces."""
    length = test_settings.challenge_code_length
    alphabet = test_settings.challenge_alphabet
    code = generate_code(length=length, alphabet=alphabet)
    assert is_well_formed(code, length=length, alphabet=alphabet)
    assert is_well_formed(f" {code.lower()} ", length=length, alphabet=alphabet)
    assert not is_well_formed(code + code, length=length, alphabet=alphabet)
    assert not is_well_formed("O" * length, length=length, alphabet=alphabet)


def test_is_expired_follows_validity_window() -> None:
    """Codes expire strictly after the validity window."""
    issued = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert not is_expired(issued, issued + timedelta(minutes=14, seconds=59), validity_minutes=15)
    assert not is_expired(issued, issued + timedelta(minutes=15), validity_minutes=15)
    assert is_expired(issued, issued + timedelta(minutes=15, seconds=1), validity_minutes=15)


def test_is_expired_rejects_naive_datetimes() -> None:
    """Naive datetimes raise instead of comparing silently."""
    naive = datetime(2026, 1, 1)
    aware = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)
    with pytest.raises(ValueError):
        is_expired(naive, aware, validity_minutes=15)
