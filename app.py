"""Streamlit interface: Sale / Delivery / Verify flows (FR/EN)."""

from __future__ import annotations

import urllib.parse
from pathlib import Path

import streamlit as st

from temoin.agent import run_investigation
from temoin.compare import is_fresh
from temoin.config import ConfigError, Settings, load_settings
from temoin.forensics import analyze_photo
from temoin.i18n import t, verdict_label
from temoin.images import ImageError, decode_data_url, draw_defects, prepare_image
from temoin.meta_api import MetaAPIError, draw_segmentation, segment_image, transcribe_audio
from temoin.model import ModelError
from temoin.report import create_report
from temoin.schemas import Certificate, Comparison, Report, TrailEntry, VideoReport
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
from temoin.video import VideoError, create_video_report, frames_dir_for


def show_report(report: Report, lang: str, photo: Path | None) -> None:
    """Render a constat: description, freshness, framed flaws, checkpoints."""
    st.write(f"**{t('report_object', lang)}** : {report.description}")
    if report.materials:
        st.write(f"**{t('report_materials', lang)}** : {', '.join(report.materials)}")
    if report.declared_materials:
        st.write(f"**{t('report_declared', lang)}** : {', '.join(report.declared_materials)}")
    if report.condition_grade:
        st.write(f"**{t('report_grade', lang)}** : {report.condition_grade}")
        st.caption(t("val_basis", lang).format(n=len(report.defects)))
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
        impact_names = {
            "minor": t("impact_low", lang),
            "moderate": t("impact_medium", lang),
            "major": t("impact_high", lang),
        }
        st.table(
            [
                {
                    "ID": defect.id,
                    "Type": defect.type,
                    "Zone": defect.zone,
                    t("col_severity", lang): severity_names.get(defect.severity, defect.severity),
                    t("col_impact", lang): impact_names.get(defect.severity, defect.severity),
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
    declared_raw = st.text_input(
        t("sale_materials", lang), value=st.session_state.get("sale_materials", "")
    )
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
                    forensics = analyze_photo(photo_path, settings=settings)
                    if forensics.suspicious or forensics.quality.advice:
                        st.subheader(t("for_title", lang))
                        for signal in forensics.signals:
                            st.warning(signal)
                        for tip in forensics.quality.advice:
                            st.info(tip)
                    declared = [item.strip() for item in declared_raw.split(",") if item.strip()]
                    st.session_state.sale_materials = declared_raw
                    report = create_report(
                        photo_path,
                        tx.sale_code.code,
                        settings=settings,
                        declared_materials=declared,
                    )
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
    st.subheader(t("voice_title", lang))
    record = st.audio_input(t("voice_record", lang))
    upload_voice = st.file_uploader(t("voice_upload", lang), type=["wav"])
    if st.button(t("voice_button", lang)):
        audio_bytes: bytes | None = None
        if record is not None:
            audio_bytes = record.getvalue()
        elif upload_voice is not None:
            audio_bytes = upload_voice.getvalue()
        if audio_bytes is None:
            st.warning(t("voice_no_audio", lang))
        else:
            with st.spinner(t("voice_running", lang)):
                try:
                    transcript = transcribe_audio(audio_bytes, settings=settings)
                except (MetaAPIError, ConfigError) as error:
                    st.error(str(error))
                    return
            if not transcript.text:
                st.warning(t("voice_empty", lang))
            else:
                tx.sale_voice_declaration = transcript.text
                save_transaction(tx, settings)
                st.rerun()
    if tx.sale_voice_declaration:
        st.write(f"« {tx.sale_voice_declaration} »")


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
    st.subheader(t("sam_title", lang))
    overlay_path = tx_dir / "segment-delivery.png"
    concept = st.text_input(t("sam_concept", lang), value=st.session_state.get("sam_concept", ""))
    if st.button(t("sam_button", lang)):
        st.session_state.sam_concept = concept.strip()
        if not concept.strip():
            st.warning(t("sam_need_concept", lang))
        elif not (tx_dir / "delivery.jpg").is_file():
            st.warning(t("err_photo_first", lang))
        else:
            with st.spinner(t("sam_running", lang)):
                try:
                    prepared = prepare_image(
                        tx_dir / "delivery.jpg",
                        max_pixels=settings.image_max_pixels,
                        max_megabytes=settings.image_max_megabytes,
                        jpeg_quality=settings.image_jpeg_quality,
                    )
                    segmentation = segment_image(
                        prepared.data_url, concept.strip(), settings=settings
                    )
                except (ImageError, MetaAPIError) as error:
                    st.error(str(error))
                else:
                    draw_segmentation(decode_data_url(prepared.data_url), segmentation).save(
                        overlay_path
                    )
                    st.session_state.sam_masks = len(segmentation.masks)
                    st.rerun()
    if overlay_path.is_file():
        if st.session_state.get("sam_masks") == 0:
            st.info(t("sam_none", lang).format(concept=st.session_state.get("sam_concept", "")))
        else:
            st.image(str(overlay_path), caption=t("sam_caption", lang))


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


def show_video_report(report: VideoReport, lang: str, settings: Settings) -> None:
    """Render a fused video constat: filmstrip, freshness, confirmed flaws."""
    st.write(f"**{t('report_object', lang)}** : {report.description}")
    frames_dir = frames_dir_for(settings, report.video_sha256)
    st.subheader(t("vid_views", lang))
    columns = st.columns(len(report.frames))
    for column, frame in zip(columns, report.frames, strict=True):
        path = frames_dir / f"frame-{frame.frame_index:02d}.jpg"
        if path.is_file():
            column.image(
                draw_defects(path, frame.report.defects),
                caption=f"{t('vid_view', lang)} {frame.frame_index + 1} · {frame.timestamp_s}s",
            )
    seen = report.challenge.seen_frames
    total = report.challenge.total_frames
    if report.challenge.verified:
        st.success(t("vid_fresh_ok", lang).format(seen=seen, total=total))
    else:
        st.warning(t("vid_fresh_ko", lang))
    if report.alerts:
        st.subheader(t("alerts", lang))
        for alert in report.alerts:
            st.warning(alert.message)
    st.subheader(t("vid_fused", lang))
    if not report.fused_defects:
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
                    "ID": flaw.id,
                    "Type": flaw.type,
                    "Zone": flaw.zone,
                    t("col_severity", lang): severity_names.get(flaw.severity, flaw.severity),
                    t("vid_confirmed", lang): f"{flaw.confirmations}/{len(report.frames)}",
                }
                for flaw in report.fused_defects
            ]
        )
    if report.checkpoints:
        st.subheader(t("report_checkpoints", lang))
        for number, point in enumerate(report.checkpoints, start=1):
            st.write(f"{number}. [{point.defect_id}] {point.instruction}")


def video_flow(settings: Settings, lang: str) -> None:
    """Run the multi-view video flow: upload, analyse, filmstrip, fusion."""
    st.info(t("vid_hint", lang))
    code = st.text_input(t("vid_code", lang), value=st.session_state.get("vid_code", ""))
    upload = st.file_uploader(t("vid_upload", lang), type=["mp4", "mov", "webm", "avi", "m4v"])
    if st.button(t("vid_analyze", lang)):
        st.session_state.vid_code = code.strip()
        if upload is None:
            st.warning(t("vid_no_file", lang))
        elif not code.strip():
            st.warning(t("vid_need_code", lang))
        else:
            suffix = Path(upload.name).suffix.lower() or ".mp4"
            video_path = settings.data_dir / "videos" / f"upload{suffix}"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(upload.getvalue())
            with st.spinner(t("vid_running", lang)):
                progress = st.progress(0)

                def _tick(done: int, total: int) -> None:
                    """Advance the frame progress bar."""
                    progress.progress(done / total)

                try:
                    report = create_video_report(
                        video_path, code.strip(), settings=settings, on_frame=_tick
                    )
                except (VideoError, ConfigError, ImageError, ModelError) as error:
                    st.error(str(error))
                    return
            st.session_state.vid_report = report.model_dump_json()
            st.rerun()
    saved = st.session_state.get("vid_report")
    if not saved:
        return
    show_video_report(VideoReport.model_validate_json(saved), lang, settings)


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

sale_tab, delivery_tab, verify_tab, video_tab = st.tabs(
    [t("tab_sale", lang), t("tab_delivery", lang), t("tab_verify", lang), t("tab_video", lang)]
)
with sale_tab:
    sale_flow(settings, lang)
with delivery_tab:
    delivery_flow(settings, lang)
with verify_tab:
    verify_flow(settings, lang)
with video_tab:
    video_flow(settings, lang)
