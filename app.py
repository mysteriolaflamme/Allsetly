"""Streamlit interface: Sale / Delivery / Verify flows (FR/EN)."""

from __future__ import annotations

import dataclasses
import json
import urllib.parse
from pathlib import Path
from typing import Any

try:
    import plotly.io as plotly_io
except ImportError:  # pragma: no cover - optional dependency
    plotly_io = None  # type: ignore[assignment]

import streamlit as st
from PIL import Image

from temoin.compare import is_fresh
from temoin.config import ConfigError, Settings, load_settings
from temoin.evidence import build_board, rows_from_reports
from temoin.forensics import analyze_photo
from temoin.i18n import defect_type_label, t, verdict_label
from temoin.images import ImageError, decode_data_url, draw_defects, prepare_image
from temoin.meta_api import MetaAPIError, draw_segmentation, segment_image, transcribe_audio
from temoin.model import ModelError
from temoin.monodepth import MonoDepthError, estimate_depth
from temoin.qa import ask_question
from temoin.relief import ReliefError, run_relief
from temoin.report import create_report
from temoin.schemas import (
    Certificate,
    Comparison,
    ReliefReport,
    Report,
    TrailEntry,
    Transaction,
    VideoReport,
)
from temoin.seal import verify_certificate, verify_chain
from temoin.service import investigate_delivery
from temoin.speech import SpeechError, synthesize
from temoin.stereo import (
    StereoError,
    compute_depth,
    depth_from_video,
    load_pair,
    relief_cloud,
    relief_wiggle,
)
from temoin.storage import (
    StorageError,
    attach_constat,
    create_transaction,
    issue_delivery_code,
    list_transactions,
    load_transaction,
    save_transaction,
    transaction_path,
)
from temoin.video import VideoError, create_video_report, extract_frames, frames_dir_for


def show_report(report: Report, lang: str, photo: Path | None) -> None:
    """Render a constat: description, freshness, framed flaws, checkpoints."""
    with st.container(border=True):
        if is_fresh(report):
            code = report.challenge.read_code or "?"
            st.badge(
                t("fresh_badge_ok", lang).format(code=code),
                icon=":material/verified:",
                color="green",
                help=t("fresh_ok", lang).format(code=code),
            )
        else:
            st.badge(
                t("fresh_badge_ko", lang),
                icon=":material/warning:",
                color="orange",
                help=t("fresh_ko", lang),
            )
        st.markdown(f"**{t('report_object', lang)}** — {report.description}")
        with st.container(horizontal=True):
            if report.condition_grade:
                st.metric(
                    t("report_grade", lang),
                    report.condition_grade,
                    help=t("val_basis", lang).format(n=len(report.defects)),
                )
            st.metric(t("metric_defects", lang), len(report.defects))
            st.metric(t("metric_checkpoints", lang), len(report.checkpoints))
        if report.materials:
            st.caption(f"{t('report_materials', lang)} : {', '.join(report.materials)}")
        if report.declared_materials:
            st.caption(f"{t('report_declared', lang)} : {', '.join(report.declared_materials)}")
    for alert in report.alerts:
        st.warning(alert.message, icon=":material/gpp_maybe:")
    if photo is not None and photo.is_file():
        st.image(draw_defects(photo, report.defects), caption=t("photo_caption", lang))
    st.subheader(t("report_defects", lang), icon=":material/troubleshoot:")
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
                    "Type": defect_type_label(defect.type, lang),
                    "Zone": defect.zone,
                    t("col_severity", lang): severity_names.get(defect.severity, defect.severity),
                    t("col_impact", lang): impact_names.get(defect.severity, defect.severity),
                }
                for defect in report.defects
            ]
        )
    if report.checkpoints:
        st.subheader(t("report_checkpoints", lang), icon=":material/my_location:")
        for number, point in enumerate(report.checkpoints, start=1):
            st.write(f"{number}. [{point.defect_id}] {point.instruction}")


_VERDICT_STYLE = {
    "CONFORME": (st.success, ":material/verified:"),
    "ENDOMMAGÉ": (st.warning, ":material/build:"),
    "À VÉRIFIER": (st.info, ":material/help:"),
    "NON CONFORME": (st.error, ":material/gpp_bad:"),
}


def show_comparison(comparison: Comparison, lang: str) -> None:
    """Render a verdict with its evidence and trail."""
    st.subheader(t("del_verdict", lang), icon=":material/gavel:")
    alert, icon = _VERDICT_STYLE.get(comparison.verdict, (st.warning, ":material/help:"))
    alert(
        comparison.justification,
        title=verdict_label(comparison.verdict, lang),
        icon=icon,
    )
    statuses = [match.status for match in comparison.matches]
    with st.container(horizontal=True):
        st.metric(t("del_confidence", lang), f"{comparison.confidence}/100", border=True)
        st.metric(t("st_matched", lang), statuses.count("matched"), border=True)
        st.metric(t("st_missing", lang), statuses.count("missing"), border=True)
        st.metric(t("st_new", lang), statuses.count("new"), border=True)
    if comparison.matches:
        st.subheader(t("del_matches", lang), icon=":material/compare_arrows:")
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
        st.subheader(t("del_contradictions", lang), icon=":material/report:")
        for contradiction in comparison.contradictions:
            st.write(f"- {contradiction}")
    if comparison.missing_proof:
        st.subheader(t("del_missing_proof", lang), icon=":material/add_a_photo:")
        for missing in comparison.missing_proof:
            st.info(missing, icon=":material/photo_camera:")
    for warning in comparison.alerts:
        st.warning(warning.message, icon=":material/gpp_maybe:")
    if st.button(t("verdict_listen", lang), icon=":material/volume_up:", key="verdict_listen"):
        speak(f"{verdict_label(comparison.verdict, lang)}. {comparison.justification}", lang)
    with st.expander(
        f"{t('del_trail', lang)} · {len(comparison.trail)} {t('ver_steps', lang)}",
        icon=":material/timeline:",
    ):
        for entry in comparison.trail:
            st.write(f"{entry.step}. **{entry.action}** — {entry.summary}")


def show_challenge_code(title: str, code: str) -> None:
    """Show a challenge code big and centered, readable from arm's length."""
    with st.container(border=True, horizontal_alignment="center"):
        st.caption(title, text_alignment="center")
        st.header(code, anchor=False, text_alignment="center")


def show_certificate(certificate: Certificate, qr_path: Path, lang: str) -> None:
    """Render a sealed certificate with download and QR code."""
    st.subheader(t("del_certificate", lang), icon=":material/workspace_premium:")
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
    if st.button(t("sale_new", lang), icon=":material/add_circle:", type="primary"):
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
    show_challenge_code(t("sale_code_title", lang), tx.sale_code.code)
    declared_raw = st.text_input(
        t("sale_materials", lang), value=st.session_state.get("sale_materials", "")
    )
    upload = st.file_uploader(t("sale_upload", lang), type=["jpg", "jpeg", "png", "webp"])
    if st.button(t("sale_analyze", lang), icon=":material/document_scanner:", type="primary"):
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
                        st.subheader(t("for_title", lang), icon=":material/image_search:")
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
    st.subheader(t("sale_share_title", lang), icon=":material/share:")
    st.code(tx.id)
    st.caption(t("sale_share_help", lang))
    text = t("sale_share_text", lang).format(tx=tx.id)
    st.link_button(
        t("sale_whatsapp", lang),
        icon=":material/send:",
        url="https://wa.me/?text=" + urllib.parse.quote(text),
    )
    st.caption(t("sale_simulated", lang))
    st.subheader(t("voice_title", lang), icon=":material/mic:")
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


_MANUAL_CASE = "__manual__"


def case_label(tx: Transaction, lang: str) -> str:
    """Describe a case for the picker: id, local date/time, short item description."""
    when = tx.created_at_utc.astimezone().strftime("%d/%m %H:%M")
    constat = tx.sale_constat or tx.delivery_constat
    about = constat.description if constat else t("case_no_description", lang)
    if len(about) > 50:
        about = about[:49] + "…"
    return f"{tx.id} · {when} · {about}"


def looks_like_challenge_code(raw: str, settings: Settings) -> bool:
    """True when the text is a challenge code typed where a case id was expected."""
    value = raw.strip()
    return len(value) == settings.challenge_code_length and not value.startswith("tx-")


def pick_case(settings: Settings, lang: str, *, label_key: str, key: str) -> str:
    """Let the user pick a case from a list (or type one); return its id or ''."""
    cases = list_transactions(settings)
    labels = {tx.id: case_label(tx, lang) for tx in cases}
    options = [*labels, _MANUAL_CASE]
    preferred = st.session_state.get("sale_tx_id") or st.session_state.get("del_tx_id")
    index = options.index(preferred) if preferred in labels else 0
    choice = st.selectbox(
        t(label_key, lang),
        options,
        index=index,
        format_func=lambda option: labels.get(option, t("case_manual", lang)),
        key=key,
    )
    if choice != _MANUAL_CASE:
        return choice
    raw = st.text_input(t("case_manual_input", lang), key=f"{key}_manual").strip()
    if looks_like_challenge_code(raw, settings):
        st.warning(t("case_is_code", lang))
        return ""
    return raw


@st.cache_data(max_entries=64, show_spinner=False)
def _meta_voice(text: str, lang: str) -> bytes:
    """Synthesize once per (text, language); failures are not cached."""
    return synthesize(text, lang)


def _browser_voice(text: str, lang: str) -> None:
    """Fallback: read text aloud with the viewer's browser (Web Speech API)."""
    payload = json.dumps(text).replace("</", r"<\/")
    voice = "fr-FR" if lang == "fr" else "en-US"
    st.html(
        "<script>(() => { const s = window.speechSynthesis; if (!s) return; s.cancel();"
        f" const u = new SpeechSynthesisUtterance({payload}); u.lang = '{voice}';"
        " s.speak(u); })();</script>",
        unsafe_allow_javascript=True,
    )


def speak(text: str, lang: str) -> bytes | None:
    """Read text aloud with Meta MMS-TTS on CPU; fall back to the browser voice."""
    try:
        with st.spinner(t("voice_preparing", lang)):
            audio = _meta_voice(text, lang)
    except SpeechError:
        _browser_voice(text, lang)
        st.caption(t("voice_browser", lang))
        return None
    st.audio(audio, format="audio/wav", autoplay=True)
    st.caption(t("voice_meta", lang))
    return audio


def voice_conversation(
    tx: Transaction, comparison: Comparison, settings: Settings, lang: str
) -> None:
    """Talk with the sealed case: Muse Voice listens, Muse Spark answers, the browser speaks."""
    st.subheader(t("ask_title", lang), icon=":material/record_voice_over:")
    st.caption(t("talk_caption", lang))
    history: list[dict[str, Any]] = st.session_state.setdefault(f"talk_{tx.id}", [])
    for turn in history:
        with st.chat_message(turn["role"], avatar=_AVATARS[turn["role"]]):
            st.markdown(turn["text"])
            if turn.get("sources"):
                st.caption(f"{t('ask_sources', lang)} : {turn['sources']}")
            if turn.get("audio"):
                st.audio(turn["audio"], format="audio/wav")
    aloud = st.toggle(t("talk_aloud", lang), value=True, key="talk_aloud")
    message = st.chat_input(t("ask_question", lang), accept_audio=True, key=f"talk_in_{tx.id}")
    if message is None:
        return
    question = message.text.strip()
    if message.audio is not None:
        with st.spinner(t("ask_transcribing", lang)):
            try:
                transcript = transcribe_audio(message.audio.getvalue(), settings=settings)
            except (MetaAPIError, ConfigError) as error:
                st.error(str(error))
                return
        question = transcript.text.strip() or question
    if not question:
        st.warning(t("ask_need", lang))
        return
    with st.chat_message("user", avatar=_AVATARS["user"]):
        st.markdown(question)
    with st.chat_message("assistant", avatar=_AVATARS["assistant"]):
        with st.spinner(t("ask_running", lang)):
            try:
                answer = ask_question(
                    question,
                    sale=tx.sale_constat,
                    delivery=tx.delivery_constat,
                    comparison=comparison,
                    settings=settings,
                    lang=lang,
                )
            except (ModelError, ConfigError) as error:
                st.error(str(error))
                return
        sources = ", ".join(answer.citations)
        st.markdown(answer.answer)
        if sources:
            st.caption(f"{t('ask_sources', lang)} : {sources}")
        audio = speak(answer.answer, lang) if aloud else None
    history.append({"role": "user", "text": question})
    history.append({"role": "assistant", "text": answer.answer, "sources": sources, "audio": audio})


_AVATARS = {"user": ":material/person:", "assistant": ":material/verified_user:"}


def delivery_flow(settings: Settings, lang: str) -> None:
    """Run the buyer flow: load case, code, photo, investigation, seal."""
    tx_id = pick_case(settings, lang, label_key="del_tx_label", key="del_case")
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
    st.subheader(t("del_recap", lang), icon=":material/inventory_2:")
    if tx.sale_constat is None:
        st.warning(t("del_no_sale", lang))
        return
    show_report(tx.sale_constat, lang, tx_dir / "sale.jpg")
    if tx.delivery_code is None:
        if st.button(t("del_gen_code", lang), icon=":material/pin:", type="primary"):
            tx = issue_delivery_code(tx, settings)
            save_transaction(tx, settings)
            st.rerun()
        return
    show_challenge_code(t("del_code_title", lang), tx.delivery_code.code)
    upload = st.file_uploader(t("del_upload", lang), type=["jpg", "jpeg", "png", "webp"])
    if st.button(t("del_investigate", lang), icon=":material/policy:", type="primary"):
        if upload is None:
            st.warning(t("err_photo_first", lang))
        else:
            status = st.status(t("del_investigating", lang), expanded=True)

            def _live(entry: TrailEntry) -> None:
                """Show one investigation step as it completes."""
                status.write(f"{entry.step}. {entry.action} — {entry.summary}")

            try:
                investigate_delivery(tx, upload.getvalue(), settings, on_step=_live)
            except (ConfigError, ImageError, ModelError, StorageError) as error:
                status.update(label=t("del_investigating", lang), state="error")
                st.error(str(error))
                return
            status.update(label=t("del_done", lang), state="complete", expanded=False)
            st.rerun()
    if not tx.certificates:
        return
    latest = tx.certificates[-1]
    show_comparison(latest.comparison, lang)
    sale_path = tx_dir / "sale.jpg"
    delivery_path = tx_dir / "delivery.jpg"
    if (
        tx.sale_constat is not None
        and tx.delivery_constat is not None
        and sale_path.is_file()
        and delivery_path.is_file()
    ):
        st.subheader(t("board_title", lang), icon=":material/compare:")
        rows = rows_from_reports(tx.sale_constat, tx.delivery_constat, latest.comparison, lang=lang)
        sale_img = Image.open(sale_path).convert("RGB")
        delivery_img = Image.open(delivery_path).convert("RGB")
        st.image(
            build_board(sale_img, delivery_img, rows, lang=lang), caption=t("board_title", lang)
        )
        voice_conversation(tx, latest.comparison, settings, lang)
    show_certificate(latest, tx_dir / f"cert-{len(tx.certificates)}.qr.png", lang)
    with st.expander(t("sam_title", lang), icon=":material/select_all:"):
        overlay_path = tx_dir / "segment-delivery.png"
        concept = st.text_input(
            t("sam_concept", lang), value=st.session_state.get("sam_concept", "")
        )
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
    st.subheader(t("ver_chain_title", lang), icon=":material/link:")
    chain_id = pick_case(settings, lang, label_key="ver_tx_label", key="ver_case")
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
                    "Type": defect_type_label(flaw.type, lang),
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


def depth_flow(settings: Settings, lang: str) -> None:
    """Run the Depth Matrix flow: stereo or mono AI, animated relief, 3D cloud."""
    st.info(t("depth_hint", lang))
    method = st.radio(
        t("depth_method", lang), [t("depth_method_stereo", lang), t("depth_method_mono", lang)]
    )
    mono = method == t("depth_method_mono", lang)
    left_upload = st.file_uploader(t("depth_left", lang), type=["jpg", "jpeg", "png", "webp"])
    right_upload = st.file_uploader(t("depth_right", lang), type=["jpg", "jpeg", "png", "webp"])
    video_upload = st.file_uploader(
        t("depth_video", lang), type=["mp4", "mov", "webm", "avi", "m4v"]
    )
    if not st.button(t("depth_button", lang)):
        return
    depth_dir = settings.data_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    with st.spinner(t("depth_running", lang)):
        try:
            if mono:
                if video_upload is not None:
                    suffix = Path(video_upload.name).suffix.lower() or ".mp4"
                    video_path = depth_dir / f"tour{suffix}"
                    video_path.write_bytes(video_upload.getvalue())
                    frames = extract_frames(
                        video_path,
                        max_frames=1,
                        max_pixels=settings.image_max_pixels,
                        max_megabytes=settings.video_max_megabytes,
                    )
                    left = frames[0].image
                elif left_upload is not None or right_upload is not None:
                    chosen = left_upload if left_upload is not None else right_upload
                    mono_path = depth_dir / "mono.jpg"
                    mono_path.write_bytes(chosen.getvalue())
                    prepared = prepare_image(
                        mono_path,
                        max_pixels=settings.image_max_pixels,
                        max_megabytes=settings.image_max_megabytes,
                        jpeg_quality=settings.image_jpeg_quality,
                    )
                    left = decode_data_url(prepared.data_url)
                else:
                    st.warning(t("depth_mono_need", lang))
                    return
                matrix = estimate_depth(left, settings=settings)
                picked: str | None = t("depth_mono_note", lang)
            elif video_upload is not None:
                suffix = Path(video_upload.name).suffix.lower() or ".mp4"
                video_path = depth_dir / f"tour{suffix}"
                video_path.write_bytes(video_upload.getvalue())
                matrix, left = depth_from_video(video_path, settings=settings)
                picked = t("depth_frames", lang).format(a=matrix.left_index, b=matrix.right_index)
            elif left_upload is not None and right_upload is not None:
                left_path = depth_dir / "left.jpg"
                right_path = depth_dir / "right.jpg"
                left_path.write_bytes(left_upload.getvalue())
                right_path.write_bytes(right_upload.getvalue())
                left, right = load_pair(left_path, right_path)
                matrix = compute_depth(left, right, settings=settings)
                picked = None
            else:
                st.warning(t("depth_need", lang))
                return
        except (StereoError, VideoError, ConfigError, MonoDepthError, ImageError) as error:
            st.error(str(error))
            return
    if mono:
        st.success(t("depth_mono_ok", lang))
    else:
        st.success(t("depth_coverage", lang).format(pct=round(matrix.coverage * 100)))
    if picked is not None:
        st.caption(picked)
    st.subheader(t("depth_wiggle", lang))
    st.image(relief_wiggle(left, matrix))
    st.subheader(t("depth_cloud", lang))
    try:
        figure = relief_cloud(left, matrix)
    except StereoError as error:
        st.warning(str(error))
    else:
        st.plotly_chart(figure, use_container_width=True)
    st.caption(t("depth_unknown", lang))


def relief_flow(settings: Settings, lang: str) -> None:
    """Run the raking-light relief flow: upload, measure, read, seal."""
    st.info(t("relief_hint", lang))
    done = st.session_state.pop("relief_done", None)
    if done:
        st.success(t("relief_attached", lang).format(side=done))
    tx_id = pick_case(settings, lang, label_key="relief_tx_label", key="relief_tx")
    side_label = st.radio(
        t("relief_side", lang),
        [t("board_sale", lang), t("board_delivery", lang)],
        horizontal=True,
        key="relief_side",
    )
    side = "sale" if side_label == t("board_sale", lang) else "delivery"
    concept = st.text_input(
        t("relief_concept", lang),
        value=st.session_state.get("relief_concept", "object"),
        key="relief_concept",
    )
    uploads = st.file_uploader(
        t("relief_uploads", lang),
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="relief_uploads",
    )
    uncalibrated = st.checkbox(t("relief_uncalibrated", lang), key="relief_uncalibrated")
    if st.button(t("relief_button", lang), key="relief_run"):
        st.session_state.relief_tx_id = tx_id.strip()
        files = list(uploads or [])
        if len(files) < settings.relief_min_photos or len(files) > settings.relief_max_photos:
            st.warning(t("relief_need_photos", lang))
            return
        try:
            tx = load_transaction(tx_id.strip(), settings)
        except StorageError as error:
            st.error(str(error))
            return
        constat = tx.sale_constat if side == "sale" else tx.delivery_constat
        if constat is None:
            st.warning(t("relief_no_constat", lang).format(side=side_label))
            return
        photo_dir = transaction_path(tx.id, settings) / "relief" / side
        photo_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for number, upload in enumerate(files):
            suffix = Path(upload.name).suffix.lower() or ".jpg"
            target = photo_dir / f"rake-{number:02d}{suffix}"
            target.write_bytes(upload.getvalue())
            paths.append(target)
        with st.spinner(t("relief_running", lang)):
            try:
                analysis = run_relief(
                    paths,
                    settings=settings,
                    concept=concept.strip() or "object",
                    uncalibrated=uncalibrated,
                    defects=constat.defects,
                )
            except (ReliefError, ConfigError, ImageError, ModelError) as error:
                st.error(str(error))
                return
        try:
            analysis.relief_image.save(photo_dir / "relief.png")
            analysis.normals_image.save(photo_dir / "normals.png")
            (photo_dir / "wiggle.gif").write_bytes(relief_wiggle(analysis.first, analysis.matrix))
            (photo_dir / "cloud.json").write_text(
                relief_cloud(analysis.first, analysis.matrix).to_json(), encoding="utf-8"
            )
            (photo_dir / "report.json").write_text(
                analysis.report.model_dump_json(indent=2), encoding="utf-8"
            )
        except StereoError as error:
            st.error(str(error))
            return
        st.session_state.relief_pending = {"tx": tx.id, "side": side, "side_label": side_label}
        st.rerun()
    pending = st.session_state.get("relief_pending")
    if not pending:
        return
    try:
        tx = load_transaction(pending["tx"], settings)
    except StorageError as error:
        st.error(str(error))
        return
    photo_dir = transaction_path(tx.id, settings) / "relief" / pending["side"]
    report_file = photo_dir / "report.json"
    if not report_file.is_file():
        return
    try:
        relief_report = ReliefReport.model_validate_json(report_file.read_text(encoding="utf-8"))
    except ValueError:
        st.error(t("relief_invalid", lang))
        return
    st.success(t("relief_ok", lang).format(pct=round(relief_report.coverage * 100)))
    method_key = (
        "relief_method_uncalibrated"
        if relief_report.method == "raking-uncalibrated"
        else "relief_method_calibrated"
    )
    st.caption(t(method_key, lang))
    mask_key = "relief_mask_sam" if relief_report.mask_source == "sam" else "relief_mask_threshold"
    st.caption(t(mask_key, lang))
    st.caption(t("relief_elapsed", lang).format(s=relief_report.compute_seconds))
    if relief_report.flat:
        st.error(t("relief_flat", lang))
    st.subheader(t("relief_map", lang))
    st.image(str(photo_dir / "relief.png"))
    st.subheader(t("relief_normals", lang))
    st.image(str(photo_dir / "normals.png"))
    st.subheader(t("depth_wiggle", lang))
    st.image(str(photo_dir / "wiggle.gif"))
    st.subheader(t("depth_cloud", lang))
    if plotly_io is None:
        st.info(t("cloud_unavailable", lang))
    else:
        st.plotly_chart(
            plotly_io.from_json((photo_dir / "cloud.json").read_text(encoding="utf-8")),
            use_container_width=True,
        )
    description = relief_report.description
    if description is not None:
        st.subheader(t("relief_reading", lang))
        st.write(description.summary)
        if description.features:
            st.table(
                [
                    {
                        "ID": feature.id,
                        t("relief_col_shape", lang): feature.kind,
                        t("relief_col_zone", lang): feature.zone,
                        t("relief_col_defect", lang): feature.defect_id or "—",
                    }
                    for feature in description.features
                ]
            )
        else:
            st.write(t("relief_features_none", lang))
        if description.note:
            st.caption(description.note)
    current = tx.sale_constat if pending["side"] == "sale" else tx.delivery_constat
    if current is not None and current.relief is not None:
        st.caption(t("relief_already", lang))
    if st.button(t("relief_attach", lang).format(side=pending["side_label"]), key="relief_attach"):
        try:
            fresh = load_transaction(pending["tx"], settings)
        except StorageError as error:
            st.error(str(error))
            return
        target = fresh.sale_constat if pending["side"] == "sale" else fresh.delivery_constat
        if target is None:
            st.warning(t("relief_no_constat", lang).format(side=pending["side_label"]))
            return
        try:
            fresh = attach_constat(
                fresh, pending["side"], target.model_copy(update={"relief": relief_report})
            )
        except StorageError as error:
            st.error(str(error))
            return
        save_transaction(fresh, settings)
        st.session_state.pop("relief_pending", None)
        st.session_state.relief_done = pending["side_label"]
        st.rerun()
    st.caption(t("relief_seal_note", lang))


st.set_page_config(page_title="Témoin", page_icon=":material/verified_user:", layout="centered")

_LAB_TOOLS = {
    "relief": ("tab_relief", ":material/light_mode:"),
    "video": ("tab_video", ":material/videocam:"),
    "depth": ("lab_experimental", ":material/view_in_ar:"),
}

with st.sidebar:
    choice = st.radio("Langue / Language", ["Français", "English"])
    lang = "fr" if choice == "Français" else "en"
    st.subheader(t("how_title", lang), icon=":material/route:")
    for step in ("how_1", "how_2", "how_3"):
        st.markdown(t(step, lang))
    st.subheader(t("meta_models", lang), icon=":material/hub:")
    with st.container(horizontal=True):
        st.badge("Muse Spark Open", icon=":material/psychology:", color="primary")
        st.badge("SAM 3.1", icon=":material/select_all:", color="violet")
        st.badge("Muse Voice Transcribe", icon=":material/mic:", color="blue")
        st.badge("Muse Code", icon=":material/code:", color="gray")
    st.caption(t("disclaimer", lang))

st.title(t("app_title", lang), icon=":material/verified_user:")
st.caption(t("tagline", lang))

try:
    settings = dataclasses.replace(load_settings(), content_language=lang)
except ConfigError as error:
    st.error(str(error))
    st.stop()

sale_tab, delivery_tab, verify_tab, lab_tab = st.tabs(
    [
        f":material/sell: 1 · {t('tab_sale', lang)}",
        f":material/local_shipping: 2 · {t('tab_delivery', lang)}",
        f":material/verified: 3 · {t('tab_verify', lang)}",
        f":material/science: {t('tab_lab', lang)}",
    ]
)
with sale_tab:
    sale_flow(settings, lang)
with delivery_tab:
    delivery_flow(settings, lang)
with verify_tab:
    verify_flow(settings, lang)
with lab_tab:
    st.caption(t("lab_hint", lang))
    tool = st.segmented_control(
        t("lab_pick", lang),
        list(_LAB_TOOLS),
        default="relief",
        required=True,
        format_func=lambda key: f"{_LAB_TOOLS[key][1]} {t(_LAB_TOOLS[key][0], lang)}",
        key="lab_tool",
    )
    if tool == "relief":
        relief_flow(settings, lang)
    elif tool == "video":
        video_flow(settings, lang)
    else:
        depth_flow(settings, lang)
