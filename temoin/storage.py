"""Local JSON storage of transactions under data/ (git-ignored)."""

from __future__ import annotations

import re
import secrets
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from temoin.challenge import generate_code, is_well_formed
from temoin.config import Settings
from temoin.schemas import Certificate, Comparison, IssuedCode, Report, Transaction
from temoin.seal import certificate_qr_png, seal_comparison

_TX_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


class StorageError(Exception):
    """Raised when stored data cannot be used, with a plain-language message."""


def transaction_path(tx_id: str, settings: Settings) -> Path:
    """Return the directory of one transaction, rejecting unsafe ids."""
    if not _TX_ID_PATTERN.fullmatch(tx_id):
        raise StorageError(f"Identifiant de transaction invalide : « {tx_id} ».")
    return settings.data_dir / "transactions" / tx_id


def _checked_code(code: str, settings: Settings) -> str:
    """Normalize an explicit code or raise when malformed."""
    normalized = code.strip().upper()
    if not is_well_formed(
        normalized,
        length=settings.challenge_code_length,
        alphabet=settings.challenge_alphabet,
    ):
        raise StorageError(f"Code défi invalide : « {code} ».")
    return normalized


def create_transaction(settings: Settings, *, sale_code: str | None = None) -> Transaction:
    """Create a transaction with a fresh sale code (kept in memory)."""
    now = datetime.now(UTC)
    code = (
        _checked_code(sale_code, settings)
        if sale_code
        else generate_code(
            length=settings.challenge_code_length, alphabet=settings.challenge_alphabet
        )
    )
    return Transaction(
        id=f"tx-{secrets.token_hex(4)}",
        created_at_utc=now,
        sale_code=IssuedCode(code=code, issued_at_utc=now),
    )


def issue_delivery_code(
    tx: Transaction, settings: Settings, *, code: str | None = None
) -> Transaction:
    """Issue a delivery code different from the sale code."""
    if code:
        wanted = _checked_code(code, settings)
        if wanted == tx.sale_code.code:
            raise StorageError("Le code livraison doit différer du code vente.")
        return tx.model_copy(
            update={"delivery_code": IssuedCode(code=wanted, issued_at_utc=datetime.now(UTC))}
        )
    for _ in range(settings.challenge_max_issues):
        drawn = generate_code(
            length=settings.challenge_code_length, alphabet=settings.challenge_alphabet
        )
        if drawn != tx.sale_code.code:
            return tx.model_copy(
                update={"delivery_code": IssuedCode(code=drawn, issued_at_utc=datetime.now(UTC))}
            )
    raise StorageError("Impossible de générer un code livraison distinct.")


def attach_constat(
    tx: Transaction, which: Literal["sale", "delivery"], report: Report
) -> Transaction:
    """Attach a constat whose code matches the issued code of that side."""
    if which == "sale":
        expected = tx.sale_code.code
    elif which == "delivery":
        if tx.delivery_code is None:
            raise StorageError("Émettez d'abord un code livraison.")
        expected = tx.delivery_code.code
    else:
        raise ValueError(f"Invalid side: {which!r}.")
    if report.challenge.expected_code != expected:
        actual = report.challenge.expected_code or "« ? »"
        raise StorageError(f"Le constat vise le code {actual} au lieu de {expected}.")
    field = "sale_constat" if which == "sale" else "delivery_constat"
    return tx.model_copy(update={field: report})


def save_transaction(tx: Transaction, settings: Settings) -> Path:
    """Write a transaction to data/; return the file path."""
    directory = transaction_path(tx.id, settings)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "transaction.json"
        path.write_text(tx.model_dump_json(indent=2), encoding="utf-8")
    except OSError as error:
        raise StorageError(f"Impossible d'enregistrer la transaction {tx.id}.") from error
    return path


def load_transaction(tx_id: str, settings: Settings) -> Transaction:
    """Load a transaction; French error when missing or corrupt."""
    path = transaction_path(tx_id, settings) / "transaction.json"
    if not path.is_file():
        raise StorageError(f"Transaction introuvable : « {tx_id} ».")
    try:
        return Transaction.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise StorageError(f"Transaction illisible : « {tx_id} ».") from error


def delete_transaction(tx_id: str, settings: Settings) -> None:
    """Delete a transaction directory (GDPR: right to erasure)."""
    directory = transaction_path(tx_id, settings)
    if not directory.exists():
        raise StorageError(f"Transaction introuvable : « {tx_id} ».")
    try:
        shutil.rmtree(directory)
    except OSError as error:
        raise StorageError(f"Impossible de supprimer la transaction {tx_id}.") from error


def add_certificate(
    tx: Transaction, certificate: Certificate, settings: Settings
) -> tuple[Transaction, Path]:
    """Append a sealed certificate and write its JSON + QR files."""
    number = len(tx.certificates) + 1
    updated = tx.model_copy(update={"certificates": [*tx.certificates, certificate]})
    directory = transaction_path(tx.id, settings)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        cert_path = directory / f"cert-{number}.json"
        cert_path.write_text(certificate.model_dump_json(indent=2), encoding="utf-8")
        qr_path = directory / f"cert-{number}.qr.png"
        qr_path.write_bytes(certificate_qr_png(certificate))
    except OSError as error:
        raise StorageError(f"Impossible d'enregistrer le certificat de {tx.id}.") from error
    save_transaction(updated, settings)
    return updated, cert_path


def seal_comparison_in_transaction(
    tx: Transaction, comparison: Comparison, settings: Settings
) -> tuple[Transaction, Path]:
    """Seal a comparison into the chain and persist JSON + QR files."""
    if tx.sale_constat is None or tx.delivery_constat is None:
        raise StorageError("Les deux constats sont requis avant de sceller.")
    number = len(tx.certificates) + 1
    previous = tx.certificates[-1].content_sha256 if tx.certificates else None
    certificate = seal_comparison(
        comparison,
        certificate_id=f"{tx.id}-c{number}",
        previous_hash=previous,
        sale=tx.sale_constat,
        delivery=tx.delivery_constat,
    )
    return add_certificate(tx, certificate, settings)
