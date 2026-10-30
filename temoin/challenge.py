"""Challenge codes: generation, format check and expiry."""

from __future__ import annotations

import secrets
from datetime import datetime


def generate_code(*, length: int, alphabet: str) -> str:
    """Generate an unpredictable code from the unambiguous alphabet."""
    if length < 1:
        raise ValueError("Code length must be at least 1.")
    if not alphabet:
        raise ValueError("Code alphabet must not be empty.")
    return "".join(secrets.choice(alphabet) for _ in range(length))


def is_well_formed(code: str, *, length: int, alphabet: str) -> bool:
    """Check a code against the expected format, ignoring case and spaces."""
    normalized = code.strip().upper()
    return len(normalized) == length and all(char in alphabet for char in normalized)


def is_expired(issued_at_utc: datetime, now_utc: datetime, *, validity_minutes: int) -> bool:
    """Return True when a code is older than its validity window."""
    if issued_at_utc.tzinfo is None or now_utc.tzinfo is None:
        raise ValueError("Datetimes must be timezone-aware.")
    return (now_utc - issued_at_utc).total_seconds() > validity_minutes * 60
