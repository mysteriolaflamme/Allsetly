"""Evidence board: side-by-side flaw crops proving a verdict visually."""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from temoin.i18n import t
from temoin.schemas import Comparison, Report

logger = logging.getLogger(__name__)

_STATUS_COLORS = {
    "matched": (46, 204, 113),
    "missing": (231, 76, 60),
    "new": (230, 126, 34),
    "changed": (241, 196, 15),
}
_UNKNOWN_COLOR = (149, 165, 166)
_TEXT_COLOR = (40, 40, 40)
_PLACEHOLDER_COLOR = (238, 238, 238)
_CELL_SIZE = 280
_CAPTION_HEIGHT = 44
_STATUS_WIDTH = 260
_PAD = 16
_ROW_GAP = 16
_FRAME = 6
_FONT_SIZE = 14
_FONT_PATH = Path(__file__).resolve().parent / "fonts" / "DejaVuSans.ttf"


def _board_font() -> ImageFont.ImageFont:
    """Load the bundled font, falling back to the bitmap default."""
    try:
        return ImageFont.truetype(str(_FONT_PATH), _FONT_SIZE)
    except OSError as error:
        logger.warning("Bundled font missing, using bitmap fallback: %s", error)
        return ImageFont.load_default()


@dataclass(frozen=True)
class BoardRow:
    """One match row: captions, boxes and status of a flaw pair."""

    label: str
    status: str
    note: str
    sale_caption: str
    delivery_caption: str
    sale_bbox: list[int] | None
    delivery_bbox: list[int] | None
    sale_missing: bool = False
    delivery_missing: bool = False


def rows_from_reports(
    sale: Report, delivery: Report, comparison: Comparison, *, lang: str = "fr"
) -> list[BoardRow]:
    """Align each match with its sale and delivery boxes."""
    sale_defects = {defect.id: defect for defect in sale.defects}
    delivery_defects = {defect.id: defect for defect in delivery.defects}
    rows = []
    for match in comparison.matches:
        sale_defect = sale_defects.get(match.sale_defect_id or "")
        delivery_defect = delivery_defects.get(match.delivery_defect_id or "")
        if sale_defect is None:
            sale_caption = f"{t('board_sale', lang)} · {t('board_missing', lang)}"
        else:
            sale_caption = f"{t('board_sale', lang)} · {sale_defect.id} ({sale_defect.type})"
        if delivery_defect is None:
            delivery_caption = f"{t('board_delivery', lang)} · {t('board_missing', lang)}"
        else:
            caption = f"{t('board_delivery', lang)} · {delivery_defect.id} ({delivery_defect.type})"
            delivery_caption = caption
        rows.append(
            BoardRow(
                label=f"{match.sale_defect_id or '?'} <-> {match.delivery_defect_id or '?'}",
                status=match.status,
                note=match.note,
                sale_caption=sale_caption,
                delivery_caption=delivery_caption,
                sale_bbox=list(sale_defect.bbox) if sale_defect and sale_defect.bbox else None,
                delivery_bbox=(
                    list(delivery_defect.bbox) if delivery_defect and delivery_defect.bbox else None
                ),
                sale_missing=sale_defect is None,
                delivery_missing=delivery_defect is None,
            )
        )
    return rows


def _crop_cell(image: Image.Image, bbox: list[int] | None) -> tuple[Image.Image, bool]:
    """Crop around a 0-1000 box with margin; whole image when box is missing."""
    width, height = image.size
    if bbox is None:
        return image.copy(), True
    x0 = min(max(round(bbox[0] / 1000 * width), 0), width)
    y0 = min(max(round(bbox[1] / 1000 * height), 0), height)
    x1 = min(max(round(bbox[2] / 1000 * width), 0), width)
    y1 = min(max(round(bbox[3] / 1000 * height), 0), height)
    if x1 <= x0 or y1 <= y0:
        return image.copy(), True
    pad_x = round((x1 - x0) * 0.3)
    pad_y = round((y1 - y0) * 0.3)
    return (
        image.crop(
            (
                max(x0 - pad_x, 0),
                max(y0 - pad_y, 0),
                min(x1 + pad_x, width),
                min(y1 + pad_y, height),
            )
        ),
        False,
    )


def _fit_cell(crop: Image.Image) -> Image.Image:
    """Center a crop on a white square cell."""
    thumb = crop.copy()
    thumb.thumbnail((_CELL_SIZE, _CELL_SIZE), Image.LANCZOS)
    cell = Image.new("RGB", (_CELL_SIZE, _CELL_SIZE), (255, 255, 255))
    cell.paste(thumb, ((_CELL_SIZE - thumb.width) // 2, (_CELL_SIZE - thumb.height) // 2))
    return cell


def build_board(
    sale_image: Image.Image,
    delivery_image: Image.Image,
    rows: list[BoardRow],
    *,
    lang: str = "fr",
) -> Image.Image:
    """Assemble match rows: sale crop, delivery crop, status panel."""
    font = _board_font()
    if not rows:
        board = Image.new("RGB", (640, 120), (255, 255, 255))
        draw = ImageDraw.Draw(board)
        text = t("board_none", lang)
        box = draw.textbbox((0, 0), text, font=font)
        draw.text(
            ((640 - (box[2] - box[0])) // 2, (120 - (box[3] - box[1])) // 2),
            text,
            font=font,
            fill=_TEXT_COLOR,
        )
        return board
    row_height = _CELL_SIZE + _CAPTION_HEIGHT
    width = _PAD * 3 + _CELL_SIZE * 2 + _STATUS_WIDTH + _PAD
    height = _PAD + len(rows) * (row_height + _ROW_GAP) - _ROW_GAP + _PAD
    board = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(board)
    for index, row in enumerate(rows):
        top = _PAD + index * (row_height + _ROW_GAP)
        color = _STATUS_COLORS.get(row.status, _UNKNOWN_COLOR)
        _draw_side(
            draw,
            board,
            sale_image,
            row.sale_bbox,
            row.sale_caption,
            row.sale_missing,
            0,
            top,
            color,
            font,
            lang,
        )
        _draw_side(
            draw,
            board,
            delivery_image,
            row.delivery_bbox,
            row.delivery_caption,
            row.delivery_missing,
            1,
            top,
            color,
            font,
            lang,
        )
        panel_left = _PAD * 3 + _CELL_SIZE * 2
        draw.text((panel_left, top), row.label, font=font, fill=_TEXT_COLOR)
        status_names = {
            "matched": t("st_matched", lang),
            "missing": t("st_missing", lang),
            "new": t("st_new", lang),
            "changed": t("st_changed", lang),
        }
        draw.text(
            (panel_left, top + 20),
            status_names.get(row.status, row.status),
            font=font,
            fill=color,
        )
        line_top = top + 44
        for line in textwrap.wrap(row.note, width=30)[:8]:
            draw.text((panel_left, line_top), line, font=font, fill=_TEXT_COLOR)
            line_top += 17
    return board


def _draw_side(
    draw: ImageDraw.ImageDraw,
    board: Image.Image,
    image: Image.Image,
    bbox: list[int] | None,
    caption: str,
    missing: bool,
    column: int,
    top: int,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    lang: str,
) -> None:
    """Draw one crop cell with its status frame and caption."""
    left = _PAD + column * (_CELL_SIZE + _PAD)
    if missing:
        cell = Image.new("RGB", (_CELL_SIZE, _CELL_SIZE), _PLACEHOLDER_COLOR)
    else:
        crop, whole = _crop_cell(image, bbox)
        cell = _fit_cell(crop)
        if whole:
            caption = f"{caption} ({t('board_full', lang)})"
    board.paste(cell, (left, top))
    draw.rectangle(
        [left, top, left + _CELL_SIZE - 1, top + _CELL_SIZE - 1], outline=color, width=_FRAME
    )
    caption_top = top + _CELL_SIZE + 4
    for line in textwrap.wrap(caption, width=34)[:2]:
        draw.text((left, caption_top), line, font=font, fill=_TEXT_COLOR)
        caption_top += 17
