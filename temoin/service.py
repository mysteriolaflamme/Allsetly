"""Case workflow shared by the Streamlit app and the business API.

Each function takes raw photo bytes, stores them in the case folder, runs the
model-backed step and persists the updated transaction, so both front doors
(human UI, partner API) follow exactly the same rules.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from temoin.agent import run_investigation
from temoin.config import Settings
from temoin.report import create_report
from temoin.schemas import Certificate, Report, TrailEntry, Transaction
from temoin.storage import (
    StorageError,
    attach_constat,
    save_transaction,
    seal_comparison_in_transaction,
    transaction_path,
)


def _store_photo(tx: Transaction, name: str, photo: bytes, settings: Settings) -> Path:
    """Write a case photo (sale.jpg / delivery.jpg) and return its path."""
    path = transaction_path(tx.id, settings) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(photo)
    return path


def record_sale(
    tx: Transaction,
    photo: bytes,
    settings: Settings,
    *,
    declared_materials: list[str] | None = None,
) -> tuple[Transaction, Report]:
    """Store the seller photo, build the sale constat and attach it to the case."""
    path = _store_photo(tx, "sale.jpg", photo, settings)
    report = create_report(
        path, tx.sale_code.code, settings=settings, declared_materials=declared_materials or []
    )
    tx = attach_constat(tx, "sale", report)
    save_transaction(tx, settings)
    return tx, report


def investigate_delivery(
    tx: Transaction,
    photo: bytes,
    settings: Settings,
    *,
    on_step: Callable[[TrailEntry], None] | None = None,
) -> tuple[Transaction, Certificate]:
    """Store the buyer photo, run the investigator agent and seal its verdict."""
    if tx.sale_constat is None:
        raise StorageError("Le constat de vente manque : photographiez d'abord la vente.")
    if tx.delivery_code is None:
        raise StorageError("Émettez d'abord un code livraison.")
    delivery_path = _store_photo(tx, "delivery.jpg", photo, settings)
    comparison = run_investigation(
        transaction_path(tx.id, settings) / "sale.jpg",
        delivery_path,
        sale_code=tx.sale_code.code,
        delivery_code=tx.delivery_code.code,
        settings=settings,
        on_step=on_step,
    )
    report = create_report(delivery_path, tx.delivery_code.code, settings=settings)
    tx = attach_constat(tx, "delivery", report)
    save_transaction(tx, settings)
    tx, _ = seal_comparison_in_transaction(tx, comparison, settings)
    return tx, tx.certificates[-1]
