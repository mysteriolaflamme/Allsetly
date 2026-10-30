"""Streamlit interface: Sale / Delivery / Verify flows (FR/EN)."""

from __future__ import annotations

import urllib.parse
from pathlib import Path

import streamlit as st

from temoin.agent import run_investigation
from temoin.compare import is_fresh
from temoin.config import ConfigError, Settings, load_settings
from temoin.i18n import t, verdict_label
from temoin.images import ImageError, draw_defects
from temoin.model import ModelError
from temoin.report import create_report
from temoin.schemas import Certificate, Comparison, Report, TrailEntry
from temoin.seal import verify_certificate, verify_chain
from temoin.storage import (
    StorageError,
    attach_constat,
    create_transaction,
    issue_delivery_code,
    load_transaction,
    save_transaction,
    seal_comparison_in_transaction,
    transaction_path,
)


def show_report(report: Report, lang: str, photo: Path | None) -> None:
    """Render a constat: description, freshness, framed flaws, checkpoints."""
    st.write(f"**{t('report_object', lang)}** : {report.description}")
    if is_fresh(report):
        st.success(t("fresh_ok", lang).format(code=report.challenge.read_code or "?"))
    else:
        st.warning(t("fresh_ko", lang))
    if report.alerts:
        st.subheader(t("alerts", lang))
        for alert in report.alerts:
            st.warning(alert.message)
    st.subheader(t("report_defects", lang))
    if not report.defects:
        st.write(t("report_none", lang))
    else:
        severity_names = {
            "minor": t("severity_minor", lang),
            "moderate": t("severity_moderate", lang),
            "major": t("severity_major", lang),
        }
        st.table(
            [
                {
                    "ID": defect.id,
                    "Type": defect.type,
                    "Zone": defect.zone,
                    t("col_severity", lang): severity_names.get(defect.severity, defect.severity),
                }
                for defect in report.defects
            ]
        )
    if photo is not None and photo.is_file():
        st.image(draw_defects(photo, report.defects), caption=t("photo_caption", lang))
    if report.checkpoints:
        st.subheader(t("report_checkpoints", lang))
        for number, point in enumerate(report.checkpoints, start=1):
            st.write(f"{number}. [{point.defect_id}] {point.instruction}")


def show_comparison(comparison: Comparison, lang: str) -> None:
    """Render a verdict with its evidence and trail."""
    st.subheader(t("del_verdict", lang))
    label = (
        f"{verdict_label(comparison.verdict, lang)} "
        f"({t('del_confidence', lang)} {comparison.confidence})"
    )
    if comparison.verdict == "CONFORME":
        st.success(label)
    elif comparison.verdict == "NON CONFORME":
        st.error(label)
    else:
        st.warning(label)
    st.progress(comparison.confidence)
    st.write(f"**{t('del_justification', lang)}** : {comparison.justification}")
    if comparison.matches:
        st.subheader(t("del_matches", lang))
        status_names = {
            "matched": t("st_matched", lang),
            "missing": t("st_missing", lang),
            "new": t("st_new", lang),
            "changed": t("st_changed", lang),
        }
        st.table(
            [
                {
                    t("col_sale", lang): match.sale_defect_id or "—",
                    t("col_delivery", lang): match.delivery_defect_id or "—",
                    t("col_status", lang): status_names.get(match.status, match.status),
                    "Note": match.note,
                }
                for match in comparison.matches
            ]
        )
    if comparison.contradictions:
        st.subheader(t("del_contradictions", lang))
        for contradiction in comparison.contradictions:
            st.write(f"- {contradiction}")
    if comparison.missing_proof:
        st.subheader(t("del_missing_proof", lang))
        for missing in comparison.missing_proof:
            st.info(missing)
    if comparison.alerts:
        st.subheader(t("alerts", lang))
        for alert in comparison.alerts:
            st.warning(alert.message)
    with st.expander(t("del_trail", lang)):
        for entry in comparison.trail:
            st.write(f"{entry.step}. **{entry.action}** — {entry.summary}")


def show_certificate(certificate: Certificate, qr_path: Path, lang: str) -> None:
    """Render a sealed certificate with download and QR code."""
    st.subheader(t("del_certificate", lang))
    st.download_button(
        t("del_download", lang),
        data=certificate.model_dump_json(indent=2),
        file_name=f"{certificate.certificate_id}.json",
        mime="application/json",
    )
    if qr_path.is_file():
        st.image(str(qr_path), caption=t("del_qr", lang))


def show_verify_result(certificate: Certificate, lang: str) -> None:
    """Render the integrity outcome and summary of one certificate."""
    comparison = certificate.comparison
    if verify_certificate(certificate):
        st.success(t("ver_intact", lang))
    else:
        st.error(t("ver_tampered", lang))
    label = (
        f"{verdict_label(comparison.verdict, lang)} "
        f"({t('del_confidence', lang)} {comparison.confidence})"
    )
    st.write(f"**{t('del_verdict', lang)}** : {label}")
    st.write(
        f"**{t('ver_photos', lang)}** : "
        f"{comparison.sale_image_sha256[:8]}… / {comparison.delivery_image_sha256[:8]}…"
    )
    st.write(f"**{t('ver_sealed_on', lang)}** : {certificate.sealed_at_utc.isoformat()}")
    st.write(f"**{t('del_trail', lang)}** : {len(comparison.trail)} {t('ver_steps', lang)}")
    link = t("ver_link_first", lang)
    if certificate.previous_hash is not None:
        link = t("ver_link_chained", lang)
    st.write(f"**{t('ver_link', lang)}** : {link}")


def sale_flow(settings: Settings, lang: str) -> None:
    """Run the seller flow: new case, code, photo, constat, share link."""
    if st.button(t("sale_new", lang)):
        tx = create_transaction(settings)
        save_transaction(tx, settings)
        st.session_state.sale_tx_id = tx.id
        st.rerun()
    tx_id = st.session_state.get("sale_tx_id")
    if not tx_id:
        st.info(t("sale_start_hint", lang))
        return
    try:
        tx = load_transaction(tx_id, settings)
    except StorageError as error:
        st.session_state.pop("sale_tx_id", None)
        st.error(str(error))
        return
    st.subheader(t("sale_code_title", lang))
    st.markdown(f"# `{tx.sale_code.code}`")
    upload = st.file_uploader(t("sale_upload", lang), type=["jpg", "jpeg", "png", "webp"])
    if st.button(t("sale_analyze", lang)):
        if upload is None:
            st.warning(t("err_photo_first", lang))
        else:
            photo_path = transaction_path(tx.id, settings) / "sale.jpg"
            photo_path.parent.mkdir(parents=True, exist_ok=True)
            photo_path.write_bytes(upload.getvalue())
            with st.spinner(t("sale_analyzing", lang)):
                try:
                    report = create_report(photo_path, tx.sale_code.code, settings=settings)
                except (ConfigError, ImageError, ModelError) as error:
                    st.error(str(error))
                    return
            tx = attach_constat(tx, "sale", report)
            save_transaction(tx, settings)
            st.rerun()
    if tx.sale_constat is None:
        return
    show_report(tx.sale_constat, lang, transaction_path(tx.id, settings) / "sale.jpg")
    st.subheader(t("sale_share_title", lang))
    st.code(tx.id)
    st.caption(t("sale_share_help", lang))
    text = t("sale_share_text", lang).format(tx=tx.id)
    st.link_button(t("sale_whatsapp", lang), "https://wa.me/?text=" + urllib.parse.quote(text))
    st.caption(t("sale_simulated", lang))


def delivery_flow(settings: Settings, lang: str) -> None:
    """Run the buyer flow: load case, code, photo, investigation, seal."""
    tx_id = st.text_input(t("del_tx_label", lang), value=st.session_state.get("del_tx_id", ""))
    if st.button(t("del_load", lang)):
        st.session_state.del_tx_id = tx_id.strip()
        st.rerun()
    active_id = st.session_state.get("del_tx_id", "").strip()
    if not active_id:
        st.info(t("del_hint", lang))
        return
    try:
        tx = load_transaction(active_id, settings)
    except StorageError:
        st.session_state.pop("del_tx_id", None)
        st.error(t("del_not_found", lang))
        return
    tx_dir = transaction_path(tx.id, settings)
    st.subheader(t("del_recap", lang))
    if tx.sale_constat is None:
        st.warning(t("del_no_sale", lang))
        return
    show_report(tx.sale_constat, lang, tx_dir / "sale.jpg")
    if tx.delivery_code is None:
        if st.button(t("del_gen_code", lang)):
            tx = issue_delivery_code(tx, settings)
            save_transaction(tx, settings)
            st.rerun()
        return
    st.subheader(t("del_code_title", lang))
    st.markdown(f"# `{tx.delivery_code.code}`")
    upload = st.file_uploader(t("del_upload", lang), type=["jpg", "jpeg", "png", "webp"])
    if st.button(t("del_investigate", lang)):
        if upload is None:
            st.warning(t("err_photo_first", lang))
        else:
            delivery_path = tx_dir / "delivery.jpg"
            delivery_path.write_bytes(upload.getvalue())
            status = st.status(t("del_investigating", lang), expanded=True)

            def _live(entry: TrailEntry) -> None:
                """Show one investigation step as it completes."""
                status.write(f"{entry.step}. {entry.action} — {entry.summary}")

            try:
                comparison = run_investigation(
                    tx_dir / "sale.jpg",
                    delivery_path,
                    sale_code=tx.sale_code.code,
                    delivery_code=tx.delivery_code.code,
                    settings=settings,
                    on_step=_live,
                )
            except (ConfigError, ImageError, ModelError) as error:
                status.update(label=t("del_investigating", lang), state="error")
                st.error(str(error))
                return
            status.update(label=t("del_done", lang), state="complete", expanded=False)
            delivery_report = create_report(delivery_path, tx.delivery_code.code, settings=settings)
            tx = attach_constat(tx, "delivery", delivery_report)
            save_transaction(tx, settings)
            tx, _cert_path = seal_comparison_in_transaction(tx, comparison, settings)
            st.rerun()
    if not tx.certificates:
        return
    latest = tx.certificates[-1]
    show_comparison(latest.comparison, lang)
    show_certificate(latest, tx_dir / f"cert-{len(tx.certificates)}.qr.png", lang)


def verify_flow(settings: Settings, lang: str) -> None:
    """Verify a dropped certificate file or a whole chain by id."""
    upload = st.file_uploader(t("ver_upload", lang), type=["json"])
    if st.button(t("ver_check", lang)):
        if upload is None:
            st.warning(t("ver_no_file", lang))
        else:
            try:
                certificate = Certificate.model_validate_json(upload.getvalue().decode("utf-8"))
            except ValueError:
                st.error(t("ver_invalid", lang))
            else:
                show_verify_result(certificate, lang)
    st.subheader(t("ver_chain_title", lang))
    chain_id = st.text_input(t("ver_tx_label", lang))
    if st.button(t("ver_chain_check", lang)):
        try:
            tx = load_transaction(chain_id.strip(), settings)
        except StorageError:
            st.error(t("del_not_found", lang))
            return
        if not tx.certificates:
            st.info(t("ver_no_certs", lang))
        elif verify_chain(tx.certificates):
            st.success(t("ver_chain_ok", lang).format(n=len(tx.certificates)))
        else:
            st.error(t("ver_chain_ko", lang))


st.set_page_config(page_title="Témoin", layout="centered")

choice = st.sidebar.radio("Langue / Language", ["Français", "English"])
lang = "fr" if choice == "Français" else "en"

st.title(t("app_title", lang))
st.caption(t("tagline", lang))
st.caption(t("disclaimer", lang))

try:
    settings = load_settings()
except ConfigError as error:
    st.error(str(error))
    st.stop()

sale_tab, delivery_tab, verify_tab = st.tabs(
    [t("tab_sale", lang), t("tab_delivery", lang), t("tab_verify", lang)]
)
with sale_tab:
    sale_flow(settings, lang)
with delivery_tab:
    delivery_flow(settings, lang)
with verify_tab:
    verify_flow(settings, lang)
