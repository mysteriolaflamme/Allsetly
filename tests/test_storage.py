"""Tests for local transaction storage under data/ (no network)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from temoin.config import Settings
from temoin.schemas import Comparison, Report
from temoin.seal import verify_chain
from temoin.storage import (
    StorageError,
    attach_constat,
    create_transaction,
    delete_transaction,
    issue_delivery_code,
    load_transaction,
    save_transaction,
    seal_comparison_in_transaction,
    transaction_path,
)


@pytest.fixture
def storage_settings(tmp_path: Path, test_settings: Settings) -> Settings:
    """Return settings with an isolated data directory."""
    return dataclasses.replace(test_settings, data_dir=tmp_path / "data")


def test_create_transaction_issues_sale_code(storage_settings: Settings) -> None:
    """Creation assigns an id and a well-formed sale code, writing nothing."""
    tx = create_transaction(storage_settings)
    assert tx.id.startswith("tx-")
    assert len(tx.sale_code.code) == storage_settings.challenge_code_length
    assert tx.delivery_code is None
    assert tx.certificates == []
    assert not (storage_settings.data_dir / "transactions").exists()


def test_create_transaction_accepts_explicit_code(storage_settings: Settings) -> None:
    """An explicit sale code is kept; a malformed one is refused."""
    tx = create_transaction(storage_settings, sale_code="K7")
    assert tx.sale_code.code == "K7"
    with pytest.raises(StorageError, match="invalide"):
        create_transaction(storage_settings, sale_code="TOOLONG")


def test_issue_delivery_code_differs(storage_settings: Settings) -> None:
    """The delivery code differs from the sale code."""
    tx = create_transaction(storage_settings, sale_code="K7")
    tx2 = issue_delivery_code(tx, storage_settings)
    assert tx2.delivery_code is not None
    assert tx2.delivery_code.code != "K7"
    assert len(tx2.delivery_code.code) == storage_settings.challenge_code_length
    tx3 = issue_delivery_code(tx, storage_settings, code="Q9")
    assert tx3.delivery_code is not None
    assert tx3.delivery_code.code == "Q9"
    with pytest.raises(StorageError, match="différer"):
        issue_delivery_code(tx, storage_settings, code="K7")


def test_save_and_load_roundtrip(storage_settings: Settings, sale_report: Report) -> None:
    """A saved transaction reloads identical."""
    tx = create_transaction(storage_settings, sale_code="K7")
    tx = attach_constat(tx, "sale", sale_report)
    path = save_transaction(tx, storage_settings)
    assert path.is_file()
    assert load_transaction(tx.id, storage_settings) == tx


def test_load_missing_transaction(storage_settings: Settings) -> None:
    """An unknown id gives a French error."""
    with pytest.raises(StorageError, match="introuvable"):
        load_transaction("tx-nope", storage_settings)


def test_transaction_path_rejects_traversal(storage_settings: Settings) -> None:
    """Unsafe ids cannot escape the data directory."""
    with pytest.raises(StorageError, match="invalide"):
        transaction_path("../evil", storage_settings)


def test_attach_constat_checks_codes(
    storage_settings: Settings, sale_report: Report, delivery_report: Report
) -> None:
    """Attached constats must match the issued codes of their side."""
    tx = create_transaction(storage_settings, sale_code="K7")
    ok = attach_constat(tx, "sale", sale_report)
    assert ok.sale_constat is not None
    other = sale_report.model_copy(deep=True)
    other.challenge.expected_code = "XX"
    with pytest.raises(StorageError, match="au lieu de"):
        attach_constat(tx, "sale", other)
    with pytest.raises(StorageError, match="code livraison"):
        attach_constat(tx, "delivery", delivery_report)
    tx2 = issue_delivery_code(tx, storage_settings, code="Q9")
    ok2 = attach_constat(tx2, "delivery", delivery_report)
    assert ok2.delivery_constat is not None


def test_seal_flow_writes_json_and_qr(
    storage_settings: Settings,
    sale_report: Report,
    delivery_report: Report,
    comparison: Comparison,
) -> None:
    """Sealing persists JSON + QR files and chains a second seal."""
    tx = create_transaction(storage_settings, sale_code="K7")
    tx = issue_delivery_code(tx, storage_settings, code="Q9")
    tx = attach_constat(tx, "sale", sale_report)
    tx = attach_constat(tx, "delivery", delivery_report)
    tx, cert_path = seal_comparison_in_transaction(tx, comparison, storage_settings)
    assert cert_path.is_file()
    assert cert_path.with_name("cert-1.qr.png").is_file()
    assert len(tx.certificates) == 1
    assert tx.certificates[0].previous_hash is None
    tx, cert_path2 = seal_comparison_in_transaction(tx, comparison, storage_settings)
    assert cert_path2.name == "cert-2.json"
    assert tx.certificates[1].previous_hash == tx.certificates[0].content_sha256
    assert verify_chain(tx.certificates) is True
    reloaded = load_transaction(tx.id, storage_settings)
    assert len(reloaded.certificates) == 2


def test_seal_requires_both_constats(
    storage_settings: Settings, sale_report: Report, comparison: Comparison
) -> None:
    """Sealing refuses a transaction missing one constat."""
    tx = create_transaction(storage_settings, sale_code="K7")
    tx = attach_constat(tx, "sale", sale_report)
    with pytest.raises(StorageError, match="deux constats"):
        seal_comparison_in_transaction(tx, comparison, storage_settings)


def test_voice_declaration_persists(storage_settings: Settings) -> None:
    """A transcribed voice note survives save/load with its transaction."""
    tx = create_transaction(storage_settings, sale_code="K7")
    assert tx.sale_voice_declaration == ""
    tx.sale_voice_declaration = "Je vends ce casque en l'état."
    save_transaction(tx, storage_settings)
    assert load_transaction(tx.id, storage_settings).sale_voice_declaration == (
        "Je vends ce casque en l'état."
    )


def test_delete_transaction(storage_settings: Settings) -> None:
    """Deletion erases the directory; unknown ids give a French error."""
    tx = create_transaction(storage_settings, sale_code="K7")
    save_transaction(tx, storage_settings)
    delete_transaction(tx.id, storage_settings)
    with pytest.raises(StorageError, match="introuvable"):
        load_transaction(tx.id, storage_settings)
    with pytest.raises(StorageError, match="introuvable"):
        delete_transaction(tx.id, storage_settings)
