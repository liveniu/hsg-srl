#!/usr/bin/env python3
"""Print a ChArUco board at 1:1 physical size (PNG preview plus SVG/PDF).

The board captures use a 4-by-5 board with 0.40 m squares:

    python tools/print_ChArUco.py --square-length 0.40 --format svg,pdf --start-id 0
    python tools/print_ChArUco.py --square-length 0.40 --format svg --start-id 10
    python tools/print_ChArUco.py --square-length 0.40 --format svg --start-id 20

A 4-by-5 board uses 10 ArUco IDs. Copy squares_x, squares_y, square_length,
marker_length, aruco_dictionary, and first_marker_id into the scene YAML.
Measure one square after printing and correct square_length if needed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.image_io import imwrite
from utils.project_paths import PROJECT_ROOT, resolve_project_path

# Small-board defaults. Override them on the command line for a 0.40 m board.
DEFAULT_SQUARES_X = 4
DEFAULT_SQUARES_Y = 5
DEFAULT_SQUARE_LENGTH = 0.13
DEFAULT_MARKER_RATIO = 0.75
DEFAULT_DICT_NAME = "DICT_4X4_50"
DEFAULT_MARGIN_MM = 50.0
DEFAULT_PNG_DPI = 150

NEXT_ID_FILE = PROJECT_ROOT / "data" / "charuco" / "next_start_id.txt"
OUT_DIR = PROJECT_ROOT / "data" / "charuco"

_DICT_MAX_IDS: dict[str, int] = {
    "DICT_4X4_50": 49,
    "DICT_4X4_100": 99,
    "DICT_4X4_250": 249,
    "DICT_4X4_1000": 999,
    "DICT_5X5_50": 49,
    "DICT_5X5_100": 99,
    "DICT_5X5_250": 249,
    "DICT_5X5_1000": 999,
    "DICT_6X6_50": 49,
    "DICT_6X6_100": 99,
    "DICT_6X6_250": 249,
    "DICT_6X6_1000": 999,
    "DICT_7X7_50": 49,
    "DICT_7X7_100": 99,
    "DICT_7X7_250": 249,
    "DICT_7X7_1000": 999,
    "DICT_ARUCO_ORIGINAL": 1023,
}


def get_dictionary(dict_name: str) -> Any:
    if not hasattr(cv.aruco, dict_name):
        raise ValueError(f"Unknown ArUco dictionary: {dict_name}")
    return cv.aruco.getPredefinedDictionary(getattr(cv.aruco, dict_name))


def dict_max_id(dict_name: str) -> int:
    return _DICT_MAX_IDS.get(dict_name, 249)


def marker_count(squares_x: int, squares_y: int, square_length: float, marker_length: float, dict_name: str) -> int:
    dictionary = get_dictionary(dict_name)
    board = cv.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
    return int(board.getIds().size)  # type: ignore[union-attr]


def read_next_start_id() -> int:
    if NEXT_ID_FILE.is_file():
        return int(NEXT_ID_FILE.read_text(encoding="utf-8").strip())
    return 0


def save_next_start_id(current_start: int, step: int, dict_name: str) -> None:
    max_id = dict_max_id(dict_name)
    next_id = current_start + step
    if next_id + step > max_id + 1:
        next_id = 0
    NEXT_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    NEXT_ID_FILE.write_text(str(next_id), encoding="utf-8")


def resolve_board_geometry(args: argparse.Namespace) -> tuple[int, int, float, float]:
    """Resolve square count and physical size in meters."""
    sx = args.squares_x
    sy = args.squares_y

    if args.board_height is not None and args.board_width is not None:
        sq_from_h = args.board_height / sy
        sq_from_w = args.board_width / sx
        if abs(sq_from_h - sq_from_w) > 1e-4:
            raise ValueError(
                f"--board-height {args.board_height}  and --board-width {args.board_width} "
                f"imply different square lengths ({sq_from_h:.4f} m vs {sq_from_w:.4f} m)."
                f"Set only one side, or change squares_x/y."
            )
        square_length = sq_from_h
    elif args.board_height is not None:
        square_length = args.board_height / sy
    elif args.board_width is not None:
        square_length = args.board_width / sx
    elif args.square_length is not None:
        square_length = args.square_length
    else:
        square_length = DEFAULT_SQUARE_LENGTH

    if square_length <= 0:
        raise ValueError("square_length must be > 0")

    marker_length = args.marker_length if args.marker_length is not None else square_length * DEFAULT_MARKER_RATIO
    if marker_length <= 0 or marker_length >= square_length:
        raise ValueError(
            f"marker_length={marker_length} is invalid; require 0 < marker_length < square_length={square_length})"
        )
    return sx, sy, square_length, marker_length


def build_board(
    start_id: int,
    squares_x: int,
    squares_y: int,
    square_length: float,
    marker_length: float,
    dict_name: str,
) -> Any:
    dictionary = get_dictionary(dict_name)
    n = marker_count(squares_x, squares_y, square_length, marker_length, dict_name)
    max_id = dict_max_id(dict_name)
    if start_id + n > max_id + 1:
        raise ValueError(
            f"start ID {start_id} + {n} markers exceed {dict_name} range (0-{max_id})"
        )
    marker_ids = np.arange(start_id, start_id + n, dtype=np.int32).reshape(-1, 1)
    return cv.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        dictionary,
        marker_ids,
    )


def build_board_raster(
    board: Any,
    margin_mm: float,
    squares_x: int,
    squares_y: int,
    square_length: float,
    dpi: int,
) -> np.ndarray:
    """Render a high-resolution PNG at physical size."""
    board_w_mm = squares_x * square_length * 1000.0
    board_h_mm = squares_y * square_length * 1000.0
    total_w_mm = board_w_mm + 2.0 * margin_mm
    total_h_mm = board_h_mm + 2.0 * margin_mm
    out_w = max(1, int(round(total_w_mm / 25.4 * dpi)))
    out_h = max(1, int(round(total_h_mm / 25.4 * dpi)))
    margin_px = max(1, int(round(margin_mm / 25.4 * dpi)))
    if hasattr(board, "generateImage"):
        return board.generateImage((out_w, out_h), marginSize=margin_px, borderBits=1)
    return cv.aruco.CharucoBoard.generateImage(board, (out_w, out_h), marginSize=margin_px, borderBits=1)


def aruco_marker_black_grid(dictionary: Any, marker_id: int) -> np.ndarray:
    """Return an (N, N) boolean array. True is a black module, including the one-module quiet border."""
    n = int(dictionary.markerSize) + 2
    side_px = n * 64
    if hasattr(cv.aruco, "generateImageMarker"):
        img = cv.aruco.generateImageMarker(dictionary, int(marker_id), side_px)
    else:
        img = np.zeros((side_px, side_px), np.uint8)
        cv.aruco.drawMarker(dictionary, int(marker_id), side_px, img, 1)
    small = cv.resize(img, (n, n), interpolation=cv.INTER_AREA)
    return small < 127


def _iter_board_rects_mm(
    board: Any,
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    margin_mm: float,
) -> list[tuple[float, float, float, float, str]]:
    """Return 1:1 mm rectangles (x, y, w, h, fill). Origin is the top left."""
    sq_mm = square_length_m * 1000.0
    rects: list[tuple[float, float, float, float, str]] = []

    for iy in range(squares_y):
        for ix in range(squares_x):
            x = margin_mm + ix * sq_mm
            y = margin_mm + iy * sq_mm
            fill = "#000000" if (ix + iy) % 2 == 0 else "#ffffff"
            rects.append((x, y, sq_mm, sq_mm, fill))

    dictionary = board.getDictionary()
    ids = board.getIds().flatten()
    for marker_id, obj_pts in zip(ids, board.getObjPoints()):
        xs = obj_pts[:, 0] * 1000.0 + margin_mm
        ys = obj_pts[:, 1] * 1000.0 + margin_mm
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        marker_w = x1 - x0
        cells = aruco_marker_black_grid(dictionary, int(marker_id))
        n = cells.shape[0]
        cell_mm = marker_w / n
        for row in range(n):
            for col in range(n):
                if cells[row, col]:
                    rects.append((x0 + col * cell_mm, y0 + row * cell_mm, cell_mm, cell_mm, "#000000"))

    return rects


def export_board_svg(
    path: Path,
    board: Any,
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    margin_mm: float,
    start_id: int,
    dict_name: str,
) -> None:
    """Write a 1:1 SVG in millimeters."""
    sq_mm = square_length_m * 1000.0
    board_w_mm = squares_x * sq_mm
    board_h_mm = squares_y * sq_mm
    total_w = board_w_mm + 2.0 * margin_mm
    total_h = board_h_mm + 2.0 * margin_mm
    n_markers = len(board.getIds())

    rects = _iter_board_rects_mm(board, squares_x, squares_y, square_length_m, margin_mm)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{total_w:.3f}mm" height="{total_h:.3f}mm" '
        f'viewBox="0 0 {total_w:.3f} {total_h:.3f}">',
        f'  <title>ChArUco {squares_x}x{squares_y} square={square_length_m*1000:.1f}mm '
        f'ids={start_id}-{start_id + n_markers - 1}</title>',
        f'  <desc>Dictionary {dict_name}. Print at 100% scale. Measure one square = {square_length_m:.4f} m.</desc>',
        f'  <rect x="0" y="0" width="{total_w:.3f}" height="{total_h:.3f}" fill="#ffffff"/>',
    ]
    for x, y, w, h, fill in rects:
        lines.append(
            f'  <rect x="{x:.6f}" y="{y:.6f}" width="{w:.6f}" height="{h:.6f}" fill="{fill}"/>'
        )
    lines.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def export_board_pdf(
    path: Path,
    board: Any,
    squares_x: int,
    squares_y: int,
    square_length_m: float,
    margin_mm: float,
    start_id: int,
    dict_name: str,
) -> None:
    """Write a 1:1 PDF. The PDF y axis points up."""
    sq_mm = square_length_m * 1000.0
    board_w_mm = squares_x * sq_mm
    board_h_mm = squares_y * sq_mm
    total_w_mm = board_w_mm + 2.0 * margin_mm
    total_h_mm = board_h_mm + 2.0 * margin_mm
    n_markers = len(board.getIds())

    mm_to_pt = 72.0 / 25.4
    page_w = total_w_mm * mm_to_pt
    page_h = total_h_mm * mm_to_pt

    rects = _iter_board_rects_mm(board, squares_x, squares_y, square_length_m, margin_mm)
    stream_parts: list[str] = ["0.0 0.0 0.0 rg", "0.0 0.0 0.0 RG"]

    def mm_rect_to_pdf(x_mm: float, y_mm: float, w_mm: float, h_mm: float, fill_hex: str) -> None:
        x = x_mm * mm_to_pt
        y_top = y_mm * mm_to_pt
        w = w_mm * mm_to_pt
        h = h_mm * mm_to_pt
        y = page_h - y_top - h
        if fill_hex == "#000000":
            stream_parts.append("0.0 0.0 0.0 rg")
        else:
            stream_parts.append("1.0 1.0 1.0 rg")
        stream_parts.append(f"{x:.4f} {y:.4f} {w:.4f} {h:.4f} re f")

    stream_parts.append("1.0 1.0 1.0 rg")
    stream_parts.append(f"0.0 0.0 {page_w:.4f} {page_h:.4f} re f")
    for x, y, w, h, fill in rects:
        mm_rect_to_pdf(x, y, w, h, fill)

    content = "\n".join(stream_parts).encode("latin-1", errors="replace")
    title = (
        f"ChArUco {squares_x}x{squares_y} {square_length_m*1000:.0f}mm "
        f"{dict_name} ids {start_id}-{start_id + n_markers - 1}"
    )

    objects: list[bytes] = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(
        f"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n".encode("ascii")
    )
    objects.append(
        f"3 0 obj << /Type /Page /Parent 2 0 R "
        f"/MediaBox [0 0 {page_w:.4f} {page_h:.4f}] "
        f"/Contents 4 0 R /Resources << >> >> endobj\n".encode("ascii")
    )
    objects.append(
        f"4 0 obj << /Length {len(content)} >> stream\n".encode("ascii") + content + b"\nendstream endobj\n"
    )
    objects.append(
        f"5 0 obj << /Title ({_pdf_escape(title)}) /Producer (print_ChArUco.py) >> endobj\n".encode("latin-1", errors="replace")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"%PDF-1.4\n")
        offsets = [0]
        for i, obj in enumerate(objects, start=1):
            offsets.append(f.tell())
            f.write(obj)
        xref_pos = f.tell()
        f.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        f.write(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            f.write(f"{off:010d} 00000 n \n".encode("ascii"))
        f.write(
            f"trailer << /Size {len(objects) + 1} /Root 1 0 R /Info 5 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n".encode("ascii")
        )


def print_board_summary(
    *,
    squares_x: int,
    squares_y: int,
    square_length: float,
    marker_length: float,
    dict_name: str,
    start_id: int,
    n_markers: int,
    margin_mm: float,
) -> None:
    board_w = squares_x * square_length
    board_h = squares_y * square_length
    end_id = start_id + n_markers - 1
    inner_corners = (squares_x - 1) * (squares_y - 1)

    print(f"Board: {squares_x}×{squares_y} squares")
    print(f"Physical size: {board_w:.3f} m × {board_h:.3f} m (margin {margin_mm:.0f} mm)")
    print(f"square_length: {square_length:.4f} m ({square_length * 100:.1f} cm)")
    print(f"marker_length: {marker_length:.4f} m (about square_length x {marker_length / square_length:.2f})")
    print(f"Dictionary: {dict_name}, inner corners {inner_corners}, ArUco IDs {start_id}–{end_id} ({n_markers} markers)")
    print("Copy into the YAML: squares_x/y, square_length, marker_length, aruco_dictionary, first_marker_id")
    print("Measure one printed square and correct square_length.")


def parse_formats(raw: str) -> set[str]:
    formats = {part.strip().lower() for part in raw.split(",") if part.strip()}
    allowed = {"png", "svg", "pdf"}
    unknown = formats - allowed
    if unknown:
        raise ValueError(f"Unknown --format: {unknown}; choices {sorted(allowed)}")
    if not formats:
        raise ValueError("--format needs at least one of png,svg,pdf")
    return formats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a ChArUco board: PNG preview plus 1:1 SVG/PDF."
    )
    parser.add_argument("--squares-x", type=int, default=DEFAULT_SQUARES_X, help="squares across")
    parser.add_argument("--squares-y", type=int, default=DEFAULT_SQUARES_Y, help="squares down")
    parser.add_argument(
        "--square-length",
        type=float,
        default=None,
        help="Square side in meters. Do not combine with both board height and width.",
    )
    parser.add_argument(
        "--board-height",
        type=float,
        default=None,
        help="Pattern height in meters. square_length = height / squares_y.",
    )
    parser.add_argument(
        "--board-width",
        type=float,
        default=None,
        help="Pattern width in meters. square_length = width / squares_x.",
    )
    parser.add_argument(
        "--marker-length",
        type=float,
        default=None,
        help=f"Marker side in meters. Default is square_length x {DEFAULT_MARKER_RATIO}",
    )
    parser.add_argument(
        "--dict",
        dest="dict_name",
        default=DEFAULT_DICT_NAME,
        help=f"ArUco dictionary (default {DEFAULT_DICT_NAME})",
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        default=DEFAULT_MARGIN_MM,
        help=f"Outer margin in millimeters (default {DEFAULT_MARGIN_MM})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_PNG_DPI,
        help=f"PNG DPI (default {DEFAULT_PNG_DPI}; 100-150 is enough for a large board)",
    )
    parser.add_argument(
        "--format",
        default="png,svg",
        help="Comma-separated output: png,svg,pdf. Use svg or pdf for printing.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=None,
        help="First ArUco ID. Defaults advance automatically. Separate boards must not share IDs.",
    )
    parser.add_argument("--out-dir", default=None, help=f"Output directory (default {OUT_DIR})")
    parser.add_argument(
        "--no-advance",
        action="store_true",
        help="Do not update next_start_id.txt when --start-id is set.",
    )
    args = parser.parse_args()

    formats = parse_formats(args.format)
    squares_x, squares_y, square_length, marker_length = resolve_board_geometry(args)
    out_dir = resolve_project_path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    n_markers = marker_count(squares_x, squares_y, square_length, marker_length, args.dict_name)
    start_id = args.start_id if args.start_id is not None else read_next_start_id()

    board = build_board(
        start_id, squares_x, squares_y, square_length, marker_length, args.dict_name
    )
    stamp = time.strftime("%Y%m%d%H%M%S")
    stem = f"charuco_{squares_x}x{squares_y}_sq{int(round(square_length * 1000)):03d}mm_id{start_id:03d}_{stamp}"

    if "png" in formats:
        png_path = out_dir / f"{stem}.png"
        board_img = build_board_raster(
            board, args.margin_mm, squares_x, squares_y, square_length, args.dpi
        )
        if not imwrite(png_path, board_img):
            raise RuntimeError(f"Failed to write PNG: {png_path}")
        print(f"PNG:  {png_path.resolve()}")

    if "svg" in formats:
        svg_path = out_dir / f"{stem}.svg"
        export_board_svg(
            svg_path,
            board,
            squares_x,
            squares_y,
            square_length,
            args.margin_mm,
            start_id,
            args.dict_name,
        )
        print(f"SVG:  {svg_path.resolve()} (1:1 mm; print at actual size)")

    if "pdf" in formats:
        pdf_path = out_dir / f"{stem}.pdf"
        export_board_pdf(
            pdf_path,
            board,
            squares_x,
            squares_y,
            square_length,
            args.margin_mm,
            start_id,
            args.dict_name,
        )
        print(f"PDF:  {pdf_path.resolve()} (1:1 mm vector)")

    print()
    print_board_summary(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length=square_length,
        marker_length=marker_length,
        dict_name=args.dict_name,
        start_id=start_id,
        n_markers=n_markers,
        margin_mm=args.margin_mm,
    )

    if not args.no_advance:
        save_next_start_id(start_id, n_markers, args.dict_name)
        print(f"The next unmarked --start-id begins at {read_next_start_id()}")


if __name__ == "__main__":
    main()
