#!/usr/bin/env python3
"""Check a ChArUco YAML against its images and write an annotated debug view.

If automatic detection fails, mark corners with tools/mark_charuco_corners.py.
This tool draws manual_charuco_corners on top of any automatic detections.
The stitch procedure prefers the manual corners.

    python tools/diagnose_charuco.py --config configs/board_capture/scene.yaml
    python tools/diagnose_charuco.py --config configs/board_capture/scene.yaml --view
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import argparse
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np
import yaml

from utils.image_text import draw_text_block, warn_if_no_pil
from utils.project_paths import resolve_project_path
from utils.image_io import imwrite
from pipeline.multi_camera_bev_stitch import (
    ARUCO_DICTS,
    _create_charuco_board,
    _detect_charuco_corners_raw,
    _detection_variants,
    _make_aruco_detector_params,
    charuco_inner_corner_count,
    load_manual_charuco_corners,
    validate_charuco_board_cfg,
)

# One BGR color per camera and board.
BOARD_COLORS: dict[str, tuple[int, int, int]] = {
    "AB_overlap": (0, 220, 0),
    "BC_overlap": (0, 180, 255),
    "CD_overlap": (255, 140, 0),
}
DEFAULT_BOARD_COLOR = (255, 0, 255)


def configured_marker_ids(boards_cfg: dict[str, dict[str, Any]]) -> set[int]:
    """Collect ArUco marker IDs from every board in the YAML."""
    ids: set[int] = set()
    for bcfg in boards_cfg.values():
        board = _create_charuco_board(bcfg)
        ids.update(int(x) for x in board.getIds().flatten())
    return ids


def detect_all_aruco_markers(
    image: np.ndarray,
    dictionary_name: str,
) -> tuple[list[np.ndarray], list[int]]:
    """Detect every ArUco marker, not one board.

    Uses the same preprocessing attempts as _detect_charuco_corners_raw:
    original, CLAHE, CLAHE plus sharpening, then 2x upsample. Keeps the attempt with the most markers.
    Returned coordinates are original-image pixels.
    """
    if dictionary_name not in ARUCO_DICTS:
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")

    dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary_name])
    det_params = _make_aruco_detector_params()

    best_corners: list[np.ndarray] = []
    best_ids: list[int] = []
    best_n = 0

    for _variant_name, variant_img, coord_scale in _detection_variants(image):
        if hasattr(cv.aruco, "ArucoDetector"):
            detector = cv.aruco.ArucoDetector(dictionary, det_params)
            corners, ids, _ = detector.detectMarkers(variant_img)
        else:
            corners, ids, _ = cv.aruco.detectMarkers(  # type: ignore[attr-defined]
                variant_img, dictionary, parameters=det_params
            )

        if ids is None or corners is None or len(ids) == 0:
            continue

        n = int(ids.size)
        if n > best_n:
            best_n = n
            if coord_scale != 1.0:
                best_corners = [c / coord_scale for c in corners]
            else:
                best_corners = list(corners)
            best_ids = [int(x) for x in ids.flatten()]

    return best_corners, best_ids


MANUAL_COLOR_OVERLAY = (0, 180, 255)   # Manual corners use one orange-blue overlay.


def board_color(board_name: str) -> tuple[int, int, int]:
    return BOARD_COLORS.get(board_name, DEFAULT_BOARD_COLOR)


def draw_manual_charuco_corners(
    canvas: np.ndarray,
    corners_dict: dict[int, list[float]],
    color: tuple[int, int, int],
    board_name: str,
) -> None:
    """Draw manually marked ChArUco corners.

    Open squares distinguish them from automatic filled circles. The ID prefix M means manual.
    """
    if not corners_dict:
        return

    for cid, xy in corners_dict.items():
        x, y = int(round(xy[0])), int(round(xy[1]))
        cv.drawMarker(canvas, (x, y), color, cv.MARKER_SQUARE, 14, 2, cv.LINE_AA)
        cv.putText(
            canvas,
            f"M{int(cid)}",
            (x + 9, y - 9),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv.LINE_AA,
        )

    # Connect corners in ID order to show the board outline.
    pts_sorted = [corners_dict[cid] for cid in sorted(corners_dict)]
    if len(pts_sorted) >= 2:
        pts_arr = np.asarray(pts_sorted, dtype=np.int32).reshape(-1, 1, 2)
        cv.polylines(canvas, [pts_arr], False, color, 1, cv.LINE_AA)

    first = corners_dict[min(corners_dict)]
    cv.putText(
        canvas,
        f"{board_name}(manual)",
        (int(first[0]), int(first[1]) - 22),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv.LINE_AA,
    )


def draw_charuco_corners(
    canvas: np.ndarray,
    corners: np.ndarray,
    ids: np.ndarray,
    color: tuple[int, int, int],
    board_name: str,
) -> None:
    for corner, cid in zip(corners, ids.flatten()):
        x, y = int(round(corner[0])), int(round(corner[1]))
        cv.circle(canvas, (x, y), 6, color, -1)
        cv.circle(canvas, (x, y), 8, (255, 255, 255), 1)
        cv.putText(
            canvas,
            str(int(cid)),
            (x + 6, y - 6),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv.LINE_AA,
        )
    # Connect corners so the board outline is visible.
    if len(corners) >= 2:
        pts = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
        cv.polylines(canvas, [pts], False, color, 1, cv.LINE_AA)
    cv.putText(
        canvas,
        board_name,
        (int(corners[0, 0]), int(corners[0, 1]) - 18),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv.LINE_AA,
    )


def draw_all_markers(
    canvas: np.ndarray,
    marker_corners: list[np.ndarray],
    marker_ids: list[int],
) -> None:
    """Draw every detected ArUco marker.

    Call OpenCV drawDetectedMarkers once so IDs are not drawn twice.
    A second putText used to stack another ID on the same marker.
    """
    if not marker_ids:
        return
    cv.aruco.drawDetectedMarkers(canvas, marker_corners, np.asarray(marker_ids).reshape(-1, 1))


def draw_legend(
    canvas: np.ndarray,
    board_names: list[str],
    has_manual: bool = False,
) -> None:
    extra = 1 if has_manual else 0
    x, y = 12, canvas.shape[0] - 16 - 22 * (len(board_names) + 2 + extra)
    lines = ["Legend:"]
    for bname in board_names:
        color = board_color(bname)
        lines.append(f"  o {bname}")
        idx = len(lines) - 1
        cv.rectangle(canvas, (x + 8, y + idx * 22 - 14), (x + 28, y + idx * 22 + 2), color, -1)
    lines.append("  all ArUco markers (diagnostic IDs only)")
    lines.append("  automatic ChArUco inner corners")
    if has_manual:
        lines.append("  manual corners (manual_charuco_corners; preferred by calibration)")
        idx = len(lines) - 1
        cv.rectangle(
            canvas,
            (x + 8, y + idx * 22 - 14),
            (x + 28, y + idx * 22 + 2),
            MANUAL_COLOR_OVERLAY,
            -1,
        )
    draw_text_block(canvas, lines, origin=(x, y), font_size=18, line_height=26)


def annotate_camera(
    image: np.ndarray,
    cam_name: str,
    board_names: list[str],
    boards_cfg: dict[str, dict[str, Any]],
    dictionary_name: str,
    manual_corners_by_board: dict[str, dict[int, list[float]]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Draw detections and manual corners. Return the image and summary lines."""
    canvas = image.copy()
    summary_lines = [f"camera {cam_name}  {image.shape[1]}x{image.shape[0]}"]

    marker_corners, marker_ids = detect_all_aruco_markers(image, dictionary_name)
    valid_marker_ids = configured_marker_ids(boards_cfg)
    if marker_ids:
        id_range = f"{min(marker_ids)}-{max(marker_ids)}"
        summary_lines.append(f"all ArUco: {len(marker_ids)} markers, IDs={id_range} (diagnostic only; not used for calibration)")
        spurious = sorted(mid for mid in marker_ids if mid not in valid_marker_ids)
        if spurious:
            summary_lines.append(
                f"  ignored IDs outside every board: {spurious}"
            )
        draw_all_markers(canvas, marker_corners, marker_ids)
    else:
        summary_lines.append("all ArUco: 0 markers")

    has_manual_any = bool(manual_corners_by_board)

    for bname in board_names:
        bcfg = boards_cfg[bname]
        corners, ids = _detect_charuco_corners_raw(image, bcfg)
        need = charuco_inner_corner_count(bcfg["squares_x"], bcfg["squares_y"])
        n = 0 if ids is None else len(ids)
        color = board_color(bname)
        if n >= need:
            status = "complete"
        elif n >= 4:
            status = f"partial({n}/{need})"
        else:
            status = f"short({n}/{need})"
        line = f"{bname}: {n}/{need} inner corners [automatic:{status}]"
        if ids is not None and len(ids):
            sample = ids.flatten()[:6].tolist()
            line += f"  id={sample}{'…' if n > 6 else ''}"
            draw_charuco_corners(canvas, corners, ids, color, bname)
        summary_lines.append(line)

        # Overlay manual corners when present.
        if manual_corners_by_board and bname in manual_corners_by_board:
            manual = manual_corners_by_board[bname]
            n_m = len(manual)
            if n_m >= need:
                m_status = "complete"
            elif n_m >= 4:
                m_status = f"partial({n_m}/{need})"
            else:
                m_status = f"short({n_m}/{need})"
            summary_lines.append(
                f"  manual: {n_m}/{need} corners [{m_status}]"
                f" (calibration prefers these)"
            )
            draw_manual_charuco_corners(canvas, manual, MANUAL_COLOR_OVERLAY, bname)

    if has_manual_any:
        summary_lines.append("  orange-blue squares are manual corners; stitch prefers them")

    draw_text_block(canvas, summary_lines, font_size=18, line_height=26)
    draw_legend(canvas, board_names, has_manual=has_manual_any)
    return canvas, summary_lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ChArUco detection check")
    parser.add_argument("--config", default="config/my_scene.yaml", help="Scene YAML")
    parser.add_argument(
        "--out-dir",
        default="outputs/debug_charuco",
        help="Annotated-image directory (default outputs/debug_charuco)",
    )
    parser.add_argument("--view", action="store_true", help="Open an OpenCV preview window")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    warn_if_no_pil("ChArUco diagnose")
    config_path = resolve_project_path(args.config)
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    boards = cfg["charuco_boards"]
    dict_name = cfg.get("aruco_dictionary", "DICT_4X4_50")
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== boards ===")
    for name, bcfg in boards.items():
        validate_charuco_board_cfg(name, bcfg)
        n = charuco_inner_corner_count(bcfg["squares_x"], bcfg["squares_y"])
        board = _create_charuco_board(bcfg)
        ids = board.getIds().flatten().tolist()
        print(
            f"  {name}: {bcfg['squares_x']}x{bcfg['squares_y']} "
            f"square={bcfg['square_length']} marker={bcfg['marker_length']} "
            f"first_id={bcfg.get('first_marker_id')} "
            f"inner={n} markers={len(ids)} ID={min(ids)}-{max(ids)}"
        )

    print(f"\n=== images (debug -> {out_dir}/) ===")
    saved_paths: list[Path] = []
    for cam in cfg["cameras"]:
        cam_name = cam["name"]
        img_path = str(resolve_project_path(cam["image"]))
        img = cv.imread(img_path)
        print(f"\n  camera {cam_name}: {img_path}")
        if img is None:
            print("    [error] cannot read image")
            continue

        board_names = cam.get("charuco_boards", [])
        manual_corners = load_manual_charuco_corners(cam)
        if manual_corners:
            boards_with_manual = list(manual_corners.keys())
            print(f"    [manual] loaded boards: {boards_with_manual}")

        annotated, summary_lines = annotate_camera(
            img, cam_name, board_names, boards, dict_name,
            manual_corners_by_board=manual_corners,
        )
        for line in summary_lines:
            print(f"    {line}")

        out_path = out_dir / f"{cam_name}_detected.jpg"
        imwrite(out_path, annotated)
        saved_paths.append(out_path)
        print(f"    [saved] {out_path}")

        if args.view:
            scale = 1280 / max(annotated.shape[1], 1)
            if scale < 1.0:
                preview = cv.resize(
                    annotated,
                    (int(annotated.shape[1] * scale), int(annotated.shape[0] * scale)),
                )
            else:
                preview = annotated
            cv.imshow(f"ChArUco diagnose: {cam_name}", preview)

    if args.view and saved_paths:
        print("\n  [preview] press any key to close")
        cv.waitKey(0)
        cv.destroyAllWindows()

    print(f"\n=== saved {len(saved_paths)} debug images to {out_dir}/ ===")


if __name__ == "__main__":
    main()
