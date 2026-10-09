'OpenCV pipeline/replay_trajectories.py'

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

try:
    from PIL import Image as PilImage
    from PIL import ImageDraw as PilDraw
    from PIL import ImageFont as PilFont

    _PIL_OK = True
except ImportError:
    _PIL_OK = False

_pil_font_cache: dict[int, Any] = {}


def pil_available() -> bool:
    return _PIL_OK


def _get_pil_font(size: int = 14) -> Any:
    if size in _pil_font_cache:
        return _pil_font_cache[size]
    font: Any = None
    for name in (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ):
        if Path(name).exists():
            font = PilFont.truetype(name, size)
            break
    if font is None:
        font = PilFont.load_default()
    _pil_font_cache[size] = font
    return font


def text_size(text: str, font_size: int = 14) -> tuple[int, int, int]:
    '(width, height, baseline)'
    if _PIL_OK:
        font = _get_pil_font(font_size)
        try:
            bbox = font.getbbox(text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            baseline = max(0, bbox[3])
        except AttributeError:
            tw, th = font.getsize(text)
            baseline = 2
        return int(tw), int(th), int(baseline)
    ascii_text = text.encode("ascii", errors="replace").decode("ascii")
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = font_size / 32.0
    (tw, th), baseline = cv.getTextSize(ascii_text, font, scale, 1)
    return int(tw), int(th), int(baseline)


def put_text(
    img: np.ndarray,
    text: str,
    xy: tuple[int, int],
    color_bgr: tuple[int, int, int],
    *,
    bg: bool = True,
    font_size: int = 14,
) -> None:
    'img Pillow CJK ASCII'
    if _PIL_OK:
        font = _get_pil_font(font_size)
        pil = PilImage.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        draw = PilDraw.Draw(pil)
        x, y = xy
        r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
        if bg:
            try:
                bbox = font.getbbox(text)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                tw, th = font.getsize(text)
            draw.rectangle([x - 1, y - 1, x + tw + 2, y + th + 2], fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=(r, g, b))
        img[:] = cv.cvtColor(np.array(pil), cv.COLOR_RGB2BGR)
        return

    ascii_text = text.encode("ascii", errors="replace").decode("ascii")
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = max(0.35, font_size / 32.0)
    thick = 1
    x, y = xy
    if bg:
        (tw, th), _ = cv.getTextSize(ascii_text, font, scale, thick)
        cv.rectangle(img, (x - 1, y - th - 2), (x + tw + 2, y + 2), (0, 0, 0), -1)
    cv.putText(img, ascii_text, (x, y), font, scale, color_bgr, thick, cv.LINE_AA)


def draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    *,
    font_size: int = 14,
) -> None:
    'origin origin draw_label'
    tw, th, baseline = text_size(text, font_size)
    x, y = origin
    x = max(0, min(x, image.shape[1] - tw - 4))
    y = max(th + baseline + 2, min(y, image.shape[0] - 2))
    top_y = y - th - baseline - 2
    put_text(image, text, (x + 2, top_y), color, bg=True, font_size=font_size)


def draw_text_block(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (12, 28),
    *,
    color_bgr: tuple[int, int, int] = (240, 240, 240),
    font_size: int = 14,
    line_height: int | None = None,
    pad: int = 8,
    bg: bool = True,
) -> None:
    'See the HSG-SRL reproduction notes.'
    if not lines:
        return
    x, y = origin
    lh = line_height or int(font_size * 1.85)
    max_w = max(text_size(line, font_size)[0] for line in lines)
    box_h = len(lines) * lh + pad * 2
    box_w = max_w + pad * 2
    if bg:
        cv.rectangle(
            image,
            (x - pad, y - font_size - pad),
            (x - pad + box_w, y - font_size - pad + box_h),
            (0, 0, 0),
            -1,
        )
    for i, line in enumerate(lines):
        put_text(
            image,
            line,
            (x, y + i * lh - font_size),
            color_bgr,
            bg=False,
            font_size=font_size,
        )


def warn_if_no_pil(context: str = "HUD") -> None:
    if not _PIL_OK:
        print(f" Pillow{context} : pip install Pillow")
