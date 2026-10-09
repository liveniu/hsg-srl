#!/usr/bin/env python3
"""Optional local-frame setup from a ChArUco board chain.

This is the board procedure in supplementary Section S5. The main experiments
use the datasets' calibrated metric frames and do not call this file.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2 as cv
import numpy as np
import yaml

from utils.project_paths import resolve_project_path
from utils.image_io import imwrite

                                               
                                               
                                                                  
GEOMETRY_VERSION = "fab1_metric_jacobian_v1"
CALIBRATION_DIAGNOSTICS_DIR = "outputs/calibration_diagnostics"


ARUCO_DICTS = {
    name: getattr(cv.aruco, name)
    for name in dir(cv.aruco)
    if name.startswith("DICT_")
}


def load_camera_matrix(path: str | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    'npz/yml/yaml None'
    if not path:
        return None, None

    p = resolve_project_path(path)
    if p.suffix.lower() == ".npz":
        data = np.load(str(p))
        camera_matrix = data["camera_matrix"] if "camera_matrix" in data else data["mtx"]
        dist_coeffs = data["dist_coeffs"] if "dist_coeffs" in data else data["dist"]
        return camera_matrix.astype(np.float64), dist_coeffs.astype(np.float64)

    fs = cv.FileStorage(str(p), cv.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calibration file: {path}")
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    if dist_coeffs is None:
        dist_coeffs = fs.getNode("dist_coefficients").mat()
    if dist_coeffs is None:
        dist_coeffs = fs.getNode("dist").mat()
    fs.release()
    if camera_matrix is None or dist_coeffs is None:
        raise ValueError(f"{path} must contain camera_matrix and dist_coeffs")
    return camera_matrix.astype(np.float64), dist_coeffs.astype(np.float64)


def marker_world_corners(marker_cfg: dict[str, Any]) -> np.ndarray:
    'See the HSG-SRL reproduction notes.'
    if "corners" in marker_cfg:
        return np.asarray(marker_cfg["corners"], dtype=np.float32)

    center_x, center_y = marker_cfg["center"]
    size = float(marker_cfg["size"])
    yaw = math.radians(float(marker_cfg.get("yaw_deg", 0.0)))
    half = size / 2.0
    local = np.asarray(
        [[-half, half], [half, half], [half, -half], [-half, -half]],
        dtype=np.float32,
    )
    rot = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float32,
    )
    return local @ rot.T + np.asarray([center_x, center_y], dtype=np.float32)


def charuco_inner_corner_count(squares_x: int, squares_y: int) -> int:
    'ChArUco = (squares_x - 1) * (squares_y - 1)'
    return max(0, (int(squares_x) - 1) * (int(squares_y) - 1))


def load_manual_charuco_corners(
    cam_cfg: dict[str, Any],
) -> dict[str, dict[int, list[float]]] | None:
    'YAML manual_charuco_corners \n\n tools/mark_charuco_corners.py \n manual_charuco_corners:\n AB_overlap:\n - [corner_id, image_x, image_y]\n\n {board_name: {corner_id: [x, y]}} None\n stitch / auto_locate ArUco'
    raw = cam_cfg.get("manual_charuco_corners")
    if not raw:
        return None
    result: dict[str, dict[int, list[float]]] = {}
    for board_name, entries in raw.items():
        if entries:
            result[str(board_name)] = {
                int(e[0]): [float(e[1]), float(e[2])] for e in entries
            }
    return result or None


def validate_charuco_board_cfg(name: str, board_cfg: dict[str, Any]) -> None:
    'marker'
    sx = int(board_cfg["squares_x"])
    sy = int(board_cfg["squares_y"])
    sq_len = float(board_cfg["square_length"])
    mk_len = float(board_cfg["marker_length"])
    corners = charuco_inner_corner_count(sx, sy)
    if corners < 4:
        raise ValueError(
            f" '{name}'  {sx}×{sy}  {corners}  ChArUco "
            f" 4  4×512  6×835 "
        )
    border = (sq_len - mk_len) * 0.5
    min_border = mk_len * 0.15                                            
    if border < min_border:
        suggested = round(sq_len * 0.75, 4)
        print(
            f"  []  '{name}': marker  {border:.4f} m square={sq_len}, marker={mk_len}"
            f" marker_length ≈ {suggested} m square_length × 0.75"
        )


def validate_charuco_board_topology(boards_cfg: dict[str, dict[str, Any]]) -> None:
    """Validate multi-board world-coordinate setup before homography fitting."""
    if not boards_cfg:
        return

    world_refs = [
        name for name, cfg in boards_cfg.items()
        if bool(cfg.get("world_reference", False))
    ]
    auto_located = [
        name for name, cfg in boards_cfg.items()
        if bool(cfg.get("auto_locate", False))
    ]

    if len(world_refs) > 1:
        joined = ", ".join(world_refs)
        raise ValueError(
            "ChArUco config error: only one board may set world_reference: true.\n"
            f"  Current world_reference boards: {joined}\n"
            "Every world_reference board is interpreted as the same world origin and "
            "orientation. Multiple physical boards configured this way create conflicting "
            "point pairs, so RANSAC may silently keep only one board.\n"
            "Keep exactly one world_reference board. Configure the other boards with "
            "auto_locate: true, or set world_origin / world_yaw_deg for their measured "
            "world poses."
        )

    if auto_located and not world_refs:
        joined = ", ".join(auto_located)
        raise ValueError(
            "ChArUco config error: auto_locate boards require one world_reference board.\n"
            f"  auto_locate boards: {joined}\n"
            "Set one board as world_reference: true to anchor the world coordinate system."
        )

    for name, cfg in boards_cfg.items():
        modes = [
            "world_reference" if cfg.get("world_reference", False) else None,
            "auto_locate" if cfg.get("auto_locate", False) else None,
            "manual_world" if "world_origin" in cfg else None,
        ]
        active_modes = [m for m in modes if m is not None]
        if len(active_modes) > 1:
            raise ValueError(
                f"ChArUco config error: board '{name}' has multiple pose modes: "
                f"{', '.join(active_modes)}.\n"
                "Use exactly one of: world_reference, auto_locate, or "
                "world_origin/world_yaw_deg."
            )
        if not active_modes:
            raise ValueError(
                f"ChArUco config error: board '{name}' has no world pose mode.\n"
                "Set world_reference: true, auto_locate: true, or "
                "world_origin / world_yaw_deg."
            )


def world_to_bev_points(world_xy: np.ndarray, scale_px_per_meter: float) -> np.ndarray:
    'BEV \n\n \n bev_x = world_x * scale X \n bev_y = world_y * scale Y world_reference row-0 \n world_y BEV'
    xy = np.asarray(world_xy, dtype=np.float32).copy()
    xy[:, 0] *= scale_px_per_meter
    xy[:, 1] *= scale_px_per_meter
    return xy


def load_manual_point_pairs(
    cam_cfg: dict[str, Any],
    scale_px_per_meter: float,
) -> tuple[np.ndarray, np.ndarray]:
    'See the HSG-SRL reproduction notes.'
    image_pts = []
    world_pts = []
    for pair in cam_cfg.get("point_pairs", []):
        image_pts.append(pair["image"])
        world_pts.append(pair["world"])
    if not image_pts:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    return (
        np.asarray(image_pts, dtype=np.float32),
        world_to_bev_points(np.asarray(world_pts, dtype=np.float32), scale_px_per_meter),
    )


def detect_aruco_points(
    image: np.ndarray,
    marker_world_by_id: dict[int, np.ndarray],
    dictionary_name: str,
    scale_px_per_meter: float,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    'ArUco BEV ID\n\n 4 marker_world_by_id \n ③ ChArUco'
    if dictionary_name not in ARUCO_DICTS:
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")

    dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICTS[dictionary_name])
    params = cv.aruco.DetectorParameters()
    if hasattr(cv.aruco, "ArucoDetector"):
        detector = cv.aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(image)
    else:
        corners, ids, _ = cv.aruco.detectMarkers(image, dictionary, parameters=params)

    if ids is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), []

    image_pts = []
    bev_pts = []
    used_ids: list[int] = []
    for marker_corners, marker_id_arr in zip(corners, ids.flatten()):
        marker_id = int(marker_id_arr)
        if marker_id not in marker_world_by_id:
            continue
        image_pts.extend(marker_corners.reshape(4, 2))
        bev_pts.extend(world_to_bev_points(marker_world_by_id[marker_id], scale_px_per_meter))
        used_ids.append(marker_id)

    return (
        np.asarray(image_pts, dtype=np.float32),
        np.asarray(bev_pts, dtype=np.float32),
        used_ids,
    )


def _create_charuco_board(board_cfg: dict[str, Any]) -> Any:
    'CharucoBoard first_marker_id ArUco ID'
    squares_x = int(board_cfg["squares_x"])
    squares_y = int(board_cfg["squares_y"])
    sq_len = float(board_cfg["square_length"])
    mk_len = float(board_cfg["marker_length"])
    dict_name = board_cfg.get("aruco_dictionary", "DICT_4X4_50")
    if dict_name not in ARUCO_DICTS:
        raise ValueError(f"Unknown ArUco dictionary: {dict_name}")

    dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
    marker_ids: np.ndarray | None = None
    if "marker_ids" in board_cfg:
        marker_ids = np.asarray(board_cfg["marker_ids"], dtype=np.int32).reshape(-1, 1)
    elif "first_marker_id" in board_cfg:
        first_id = int(board_cfg["first_marker_id"])
        default_board = cv.aruco.CharucoBoard((squares_x, squares_y), sq_len, mk_len, dictionary)
        count = int(default_board.getIds().size)  # type: ignore[union-attr]
        marker_ids = np.arange(first_id, first_id + count, dtype=np.int32).reshape(-1, 1)

    try:
        if marker_ids is not None:
            board = cv.aruco.CharucoBoard(
                (squares_x, squares_y), sq_len, mk_len, dictionary, marker_ids
            )
        else:
            board = cv.aruco.CharucoBoard((squares_x, squares_y), sq_len, mk_len, dictionary)
    except TypeError:
        if marker_ids is not None:
            raise RuntimeError(
                'OpenCV marker_ids opencv-contrib-python'
            ) from None
        board = cv.aruco.CharucoBoard_create(  # type: ignore[attr-defined]
            squares_x, squares_y, sq_len, mk_len, dictionary
        )
    return board


def charuco_marker_id_grid(board_cfg: dict[str, Any]) -> np.ndarray:
    'ChArUco marker ID shape=(squares_y, squares_x) marker -1\n\n OpenCV CharucoBoard (row+col)%2==1 markerID \n first_marker_id / marker_ids'
    sx = int(board_cfg["squares_x"])
    sy = int(board_cfg["squares_y"])
    board = _create_charuco_board(board_cfg)
    ids = board.getIds().flatten()
    grid = np.full((sy, sx), -1, dtype=np.int32)
    idx = 0
    for row in range(sy):
        for col in range(sx):
            if (row + col) % 2 == 1:
                grid[row, col] = int(ids[idx])
                idx += 1
    return grid


def _make_aruco_detector_params() -> cv.aruco.DetectorParameters:
    '/ ArUco \n\n \n · (3–53)\n · minMarkerPerimeterRate 0.01\n · (CORNER_REFINE_SUBPIX)\n · (0.6)/'
    p = cv.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 4
    p.adaptiveThreshConstant = 7
    p.minMarkerPerimeterRate = 0.01
    p.maxMarkerPerimeterRate = 4.0
    p.polygonalApproxAccuracyRate = 0.03
    p.errorCorrectionRate = 0.6
    p.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 30
    return p


def _detection_variants(image: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
    '(, , )\n > 1 \n _detect_charuco_corners_raw / detect_aruco_enhanced'
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image

    def to_bgr(g: np.ndarray) -> np.ndarray:
        return cv.cvtColor(g, cv.COLOR_GRAY2BGR) if image.ndim == 3 else g

    variants: list[tuple[str, np.ndarray, float]] = [
        ('See the HSG-SRL reproduction notes.', image, 1.0),
    ]

                                  
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g_clahe = clahe.apply(gray)
    variants.append(("CLAHE", to_bgr(g_clahe), 1.0))

                                  
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    g_sharp = np.clip(
        cv.filter2D(g_clahe.astype(np.float32), -1, kernel), 0, 255
    ).astype(np.uint8)
    variants.append(('CLAHE+', to_bgr(g_sharp), 1.0))

                                             
    g_blur_clahe = clahe.apply(cv.GaussianBlur(gray, (3, 3), 0))
    variants.append(('+CLAHE', to_bgr(g_blur_clahe), 1.0))

                                                 
                             
    up_scale = 2.0
    h, w = gray.shape[:2]
    g_up = cv.resize(gray, (int(w * up_scale), int(h * up_scale)), interpolation=cv.INTER_CUBIC)
    g_up_clahe = clahe.apply(g_up)
    variants.append(('2×+CLAHE', to_bgr(g_up_clahe), up_scale))

    return variants


def _detect_charuco_corners_raw(
    image: np.ndarray,
    board_cfg: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    'ChArUco (corners [N,2], ids [N]) \n\n → CLAHE → CLAHE+ → +CLAHE →\n 2×+CLAHE'
    board = _create_charuco_board(board_cfg)
    total_expected = charuco_inner_corner_count(
        int(board_cfg["squares_x"]), int(board_cfg["squares_y"])
    )
    det_params = _make_aruco_detector_params()

    best_corners: np.ndarray | None = None
    best_ids: np.ndarray | None = None
    best_n = 0

    for _variant_name, variant_img, coord_scale in _detection_variants(image):
        corners: np.ndarray | None = None
        ids: np.ndarray | None = None

        if hasattr(cv.aruco, "CharucoDetector"):
                                
            charuco_params = cv.aruco.CharucoParameters()
            charuco_params.minMarkers = 1                                  
            detector = cv.aruco.CharucoDetector(board, charuco_params, det_params)
            corners, ids, _, _ = detector.detectBoard(variant_img)
        else:
                         
            gray_v = (
                cv.cvtColor(variant_img, cv.COLOR_BGR2GRAY)
                if variant_img.ndim == 3
                else variant_img
            )
            dict_name = board_cfg.get("aruco_dictionary", "DICT_4X4_50")
            dictionary = cv.aruco.getPredefinedDictionary(ARUCO_DICTS[dict_name])
            mc, mi, _ = cv.aruco.detectMarkers(  # type: ignore[attr-defined]
                gray_v, dictionary, parameters=det_params
            )
            if mi is not None and len(mi) > 0:
                _, corners, ids = cv.aruco.interpolateCornersCharuco(  # type: ignore[attr-defined]
                    mc, mi, gray_v, board
                )

        if corners is None or ids is None or len(ids) == 0:
            continue

        n = len(ids)
                       
        if coord_scale != 1.0:
            corners = corners / coord_scale

        if n > best_n:
            best_n = n
            best_corners = corners
            best_ids = ids

        if best_n >= total_expected:
            break                     

    if best_corners is None or best_ids is None:
        return None, None
    return best_corners.reshape(-1, 2), best_ids.flatten()


def charuco_corner_local_xy(idx: int, squares_x: int, square_length: float) -> tuple[float, float]:
    'ChArUco'
    cols_per_row = int(squares_x) - 1
    row = int(idx) // cols_per_row
    col = int(idx) % cols_per_row
    return (col + 1) * square_length, (row + 1) * square_length


def extrapolate_charuco_board_world_corners(
    board_cfg: dict[str, Any],
    partial_corner_world: dict[int, list[float]],
) -> dict[int, list[float]]:
    '2D \n\n auto_locate \n ID 0 \n ≥4'
    sx = int(board_cfg["squares_x"])
    sy = int(board_cfg["squares_y"])
    sq_len = float(board_cfg["square_length"])
    total = charuco_inner_corner_count(sx, sy)
    if len(partial_corner_world) >= total:
        return dict(partial_corner_world)
    if len(partial_corner_world) < 3:
        return dict(partial_corner_world)

    src_pts: list[list[float]] = []
    dst_pts: list[list[float]] = []
    for idx, wxy in partial_corner_world.items():
        lx, ly = charuco_corner_local_xy(int(idx), sx, sq_len)
        src_pts.append([lx, ly])
        dst_pts.append(wxy)

    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    if len(src) == 3:
        affine = cv.getAffineTransform(src, dst)
    else:
        affine, _ = cv.estimateAffine2D(src, dst)
        if affine is None:
            return dict(partial_corner_world)

    full: dict[int, list[float]] = {}
    for idx in range(total):
        lx, ly = charuco_corner_local_xy(idx, sx, sq_len)
        wx, wy = affine @ np.array([lx, ly, 1.0], dtype=np.float64)
        full[idx] = [float(wx), float(wy)]
    return full


def describe_charuco_detection_gaps(
    image: np.ndarray,
    boards_cfg: dict[str, dict[str, Any]],
    camera_board_names: list[str],
    board_corner_world_override: dict[str, dict[int, list[float]]] | None,
) -> list[str]:
    'ChArUco'
    lines: list[str] = []
    for board_name in camera_board_names:
        if board_name not in boards_cfg:
            continue
        cfg = boards_cfg[board_name]
        corners, ids = _detect_charuco_corners_raw(image, cfg)
        detected = 0 if ids is None else len(ids)
        need = charuco_inner_corner_count(int(cfg["squares_x"]), int(cfg["squares_y"]))
        matched = detected
        if cfg.get("auto_locate", False) and board_corner_world_override:
            pre = board_corner_world_override.get(board_name, {})
            if ids is not None and pre:
                matched = sum(1 for i in ids if int(i) in pre)
        id_sample = ""
        if ids is not None and len(ids) > 0:
            flat = [int(x) for x in ids.flatten()[:6]]
            id_sample = f" id={flat}{'...' if len(ids) > 6 else ''}"
        lines.append(
            f" '{board_name}':  {detected}/{need}  {matched}{id_sample}"
        )
    return lines


def detect_charuco_points(
    image: np.ndarray,
    boards_cfg: dict[str, dict[str, Any]],
    camera_board_names: list[str],
    scale_px_per_meter: float,
    board_corner_world_override: dict[str, dict[int, list[float]]] | None = None,
    manual_charuco_corners: dict[str, dict[int, list[float]]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    'ChArUco BEV \n\n YAML \n\n ① world_reference: true\n = (0,0) = X \n world_origin / world_yaw_deg\n\n ② auto_locate: true\n \n auto_locate_charuco_boards()\n\n ③ \n world_origin: [x, y] world_yaw_deg \n \n\n manual_charuco_corners \n tools/mark_charuco_corners.py ArUco'
    all_image_pts: list[np.ndarray] = []
    all_world_pts: list[list[float]] = []

    for board_name in camera_board_names:
        if board_name not in boards_cfg:
            continue
        cfg = boards_cfg[board_name]

                                        
        if manual_charuco_corners and board_name in manual_charuco_corners:
            manual = manual_charuco_corners[board_name]
            corners = np.array([[v[0], v[1]] for v in manual.values()], dtype=np.float32)
            ids = np.array(list(manual.keys()), dtype=np.int32)
            print(f"  [manual]  '{board_name}' {len(corners)} ")
        else:
            corners, ids = _detect_charuco_corners_raw(image, cfg)
        if corners is None or ids is None:
            continue

        sq_len = float(cfg["square_length"])

        if cfg.get("world_reference", False):
                                                           
            for corner, idx in zip(corners, ids):
                wx, wy = charuco_corner_local_xy(int(idx), int(cfg["squares_x"]), sq_len)
                all_image_pts.append(corner)
                all_world_pts.append([wx, wy])

        elif cfg.get("auto_locate", False):
                                                              
            if board_corner_world_override is None or board_name not in board_corner_world_override:
                continue
            precomputed = board_corner_world_override[board_name]
            for corner, idx in zip(corners, ids):
                if int(idx) in precomputed:
                    all_image_pts.append(corner)
                    all_world_pts.append(precomputed[int(idx)])

        else:
                                                          
            ox = float(cfg["world_origin"][0])
            oy = float(cfg["world_origin"][1])
            yaw = math.radians(float(cfg.get("world_yaw_deg", 0.0)))
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            for corner, idx in zip(corners, ids):
                lx, ly = charuco_corner_local_xy(int(idx), int(cfg["squares_x"]), sq_len)
                wx = ox + lx * cos_y - ly * sin_y
                wy = oy + lx * sin_y + ly * cos_y
                all_image_pts.append(corner)
                all_world_pts.append([wx, wy])

    if not all_image_pts:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    image_arr = np.asarray(all_image_pts, dtype=np.float32)
    world_arr = np.asarray(all_world_pts, dtype=np.float32)
    return image_arr, world_to_bev_points(world_arr, scale_px_per_meter)


def auto_locate_charuco_boards(
    all_images: dict[str, np.ndarray],
    boards_cfg: dict[str, dict[str, Any]],
    cam_board_names_map: dict[str, list[str]],
    scale_px_per_meter: float,
    ransac_reproj_threshold: float,
    manual_corners_by_cam: dict[str, dict[str, dict[int, list[float]]]] | None = None,
) -> dict[str, dict[int, list[float]]]:
    'auto_locate=true BFS \n\n N \n cam0 ─[board01]─ cam1 ─[board12]─ cam2 ─[board23]─ cam3 …\n\n BFS\n 1. world_reference=true ""\n 2. auto_locate \n ""\n H\n H \n 3. \n 4. \n\n {board_name: {corner_id: [wx_meters, wy_meters]}}'
    ref_names: set[str] = {
        name for name, cfg in boards_cfg.items() if cfg.get("world_reference", False)
    }
    auto_names = [name for name, cfg in boards_cfg.items() if cfg.get("auto_locate", False)]

    if not ref_names or not auto_names:
        return {}

                                                
    known_overrides: dict[str, dict[int, list[float]]] = {}
    remaining = list(auto_names)

    for _iteration in range(len(auto_names) + 1):
        if not remaining:
            break
        progress = False
        failure_notes: list[str] = []

        for auto_name in remaining[:]:
                                            
            for cam_name, cam_boards in cam_board_names_map.items():
                if auto_name not in cam_boards:
                    continue
                known_visible = [
                    b for b in cam_boards
                    if b in ref_names or b in known_overrides
                ]
                if not known_visible:
                    continue                     

                bridge_img = all_images[cam_name]
                bridge_manual = (
                    manual_corners_by_cam.get(cam_name)
                    if manual_corners_by_cam else None
                )

                                            
                ref_img_pts, ref_bev_pts = detect_charuco_points(
                    bridge_img,
                    {b: boards_cfg[b] for b in known_visible},
                    known_visible,
                    scale_px_per_meter,
                    board_corner_world_override=known_overrides,
                    manual_charuco_corners=bridge_manual,
                )
                if len(ref_img_pts) < 4:
                    per_board: list[str] = []
                    for bname in known_visible:
                        corners, ids = _detect_charuco_corners_raw(
                            bridge_img, boards_cfg[bname]
                        )
                        n = 0 if ids is None else len(ids)
                        need = charuco_inner_corner_count(
                            boards_cfg[bname]["squares_x"],
                            boards_cfg[bname]["squares_y"],
                        )
                        per_board.append(f"{bname}  {n}/{need} ")
                    failure_notes.append(
                        f" '{cam_name}'  4 "
                        f" {len(ref_img_pts)}{'; '.join(per_board)}"
                    )
                    continue                

                try:
                    H_bridge, _ = estimate_image_to_bev_homography(
                        ref_img_pts, ref_bev_pts, ransac_reproj_threshold
                    )
                except RuntimeError:
                    failure_notes.append(
                        f" '{cam_name}'  {known_visible} "
                    )
                    continue

                                          
                if bridge_manual and auto_name in bridge_manual:
                    manual_data = bridge_manual[auto_name]
                    corners = np.array(
                        [[v[0], v[1]] for v in manual_data.values()], dtype=np.float32
                    )
                    ids = np.array(list(manual_data.keys()), dtype=np.int32)
                    print(
                        f"  [manual] auto_locate  '{auto_name}'  "
                        f"'{cam_name}' {len(corners)} "
                    )
                else:
                    corners, ids = _detect_charuco_corners_raw(bridge_img, boards_cfg[auto_name])
                if corners is None or ids is None:
                    failure_notes.append(
                        f" '{cam_name}'  '{auto_name}'  ChArUco "
                    )
                    continue

                                         
                corner_world: dict[int, list[float]] = {}
                for corner, idx in zip(corners, ids):
                    pt_h = np.array([corner[0], corner[1], 1.0], dtype=np.float64)
                    bev_h = H_bridge @ pt_h
                    bev_x = bev_h[0] / bev_h[2]
                    bev_y = bev_h[1] / bev_h[2]
                    corner_world[int(idx)] = [
                        bev_x / scale_px_per_meter,
                        bev_y / scale_px_per_meter,
                    ]

                known_overrides[auto_name] = extrapolate_charuco_board_world_corners(
                    boards_cfg[auto_name],
                    corner_world,
                )
                remaining.remove(auto_name)
                progress = True
                n_partial = len(corner_world)
                n_full = len(known_overrides[auto_name])
                print(
                    f"[auto_locate]  '{auto_name}'  '{cam_name}' "
                    f"{known_visible} {n_partial} "
                    f" {n_full} "
                )
                break                  

        if not progress and remaining:
                                                                
            boards_needing_ruler = [b for b in remaining if boards_cfg[b].get("ruler_measurements")]
            boards_must_fix = [b for b in remaining if b not in boards_needing_ruler]
            if boards_needing_ruler:
                print(
                    f"  [auto_locate]  ruler_measurements auto_locate "
                    f"{boards_needing_ruler}"
                )
                for b in boards_needing_ruler:
                    remaining.remove(b)
                if not remaining:
                    break
            if boards_must_fix:
                detail = "\n  ".join(dict.fromkeys(failure_notes)) if failure_notes else 'See the HSG-SRL reproduction notes.'
                board_hints = []
                for bname in boards_must_fix:
                    if bname in boards_cfg:
                        bc = boards_cfg[bname]
                        sx, sy = int(bc["squares_x"]), int(bc["squares_y"])
                        nc = charuco_inner_corner_count(sx, sy)
                        board_hints.append(f"{bname}  {sx}×{sy} {nc} ")
                hint_boards = 'See the HSG-SRL reproduction notes.'.join(board_hints) if board_hints else 'YAML charuco_boards'
                raise RuntimeError(
                    f"{boards_must_fix}\n"
                    f"\n"
                    f"  1)  ChArUco  ≥ 4{hint_boards}\n"
                    f"  2)  + \n"
                    f"  3) YAML  square_length / marker_length / first_marker_id  tools/print_ChArUco.py \n"
                    f"  4)  py tools/diagnose_charuco.py --config <yaml>  ArUco ID\n"
                    f"  5)  ruler_measurements \n"
                    f"\n  {detail}"
                )

    return known_overrides


# ──────────────────────────────────────────────────────────────────────────────
                                          
# ──────────────────────────────────────────────────────────────────────────────

                                           
# ──────────────────────────────────────────────────────────────────────────────
         
#
                     
                                                   
                           
                           
                           
#
               
                                                             
#
                                                        
#
#         outer_tl ┌───┬───┬───┬───┐ outer_tr
                                             
#                  │   3   4   5   │
#                  │   6   7   8   │
                                           
#         outer_bl └───┴───┴───┴───┘ outer_br
#
                                                                   
                                                
#
                                                      
                


_OUTER_CORNER_NAMES = {"outer_tl", "outer_tr", "outer_bl", "outer_br"}


def _resolve_corner_ref(
    corner_ref: int | str,
    board_cfg: dict[str, Any],
) -> tuple[float, float]:
    '= outer_tlX Y'
    sq_len = float(board_cfg["square_length"])
    sx = int(board_cfg["squares_x"])
    sy = int(board_cfg["squares_y"])

    if isinstance(corner_ref, str):
        ref = corner_ref.lower()
        if ref == "outer_tl":
            return 0.0, 0.0
        if ref == "outer_tr":
            return sx * sq_len, 0.0
        if ref == "outer_bl":
            return 0.0, sy * sq_len
        if ref == "outer_br":
            return sx * sq_len, sy * sq_len
        raise ValueError(
            f" '{corner_ref}': outer_tl / outer_tr / outer_bl / outer_br "
            f"0 ~ {(sx-1)*(sy-1)-1}"
        )
    return charuco_corner_local_xy(int(corner_ref), sx, sq_len)


def _board_local_to_world(
    lx: float,
    ly: float,
    board_name: str,
    boards_cfg: dict[str, dict[str, Any]],
    known_overrides: dict[str, dict[int, list[float]]],
) -> np.ndarray | None:
    '(lx, ly) \n\n \n · world_reference outer_tl\n · auto_locate / ruler_measurements overrides'
    cfg = boards_cfg.get(board_name)
    if cfg is None:
        return None

    if cfg.get("world_reference", False):
        return np.array([lx, ly], dtype=np.float64)

                                
    overrides = known_overrides.get(board_name, {})
    if len(overrides) < 2:
        return None

    sq_len = float(cfg["square_length"])
    sq_x = int(cfg["squares_x"])
    ids = sorted(overrides.keys())[:8]                             
    l_pts = np.array([charuco_corner_local_xy(i, sq_x, sq_len) for i in ids])
    w_pts = np.array([overrides[i] for i in ids])
    ox, oy, yaw_deg = _fit_board_pose_2d(l_pts, w_pts)

    yaw_rad = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    wx = ox + lx * cos_y - ly * sin_y
    wy = oy + lx * sin_y + ly * cos_y
    return np.array([wx, wy], dtype=np.float64)


def _get_board_corner_world(
    board_name: str,
    corner_ref: int | str,
    boards_cfg: dict[str, dict[str, Any]],
    known_overrides: dict[str, dict[int, list[float]]],
) -> np.ndarray | None:
    'corner_ref \n · 0 = \n · outer_tl / outer_tr / outer_bl / outer_br'
    cfg = boards_cfg.get(board_name)
    if cfg is None:
        return None
    try:
        lx, ly = _resolve_corner_ref(corner_ref, cfg)
    except (ValueError, TypeError) as e:
        print(f"  [ruler] {board_name}/{corner_ref}: {e}")
        return None
    return _board_local_to_world(lx, ly, board_name, boards_cfg, known_overrides)


def _trilaterate_2d(
    anchors: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    'N≥3 2D \n\n \n ────\n (i, 0) tx²+ty² \n A·t = b A b \n 2*(x_i - x_0)*tx + 2*(y_i - y_0)*ty\n = (x_i² - x_0²) + (y_i² - y_0²) - (d_i² - d_0²)\n N=3 N>3 np.linalg.lstsq \n\n σ_d ≤ 5mm σ_d / sqrt(N)'
    if len(anchors) < 3:
        raise ValueError(
            f" 3  {len(anchors)} "
            'to_corner ruler_measurements'
        )
    A = 2.0 * (anchors[1:] - anchors[0])         # (N-1, 2)
    b = (
        np.sum(anchors[1:] ** 2, axis=1)
        - np.sum(anchors[0] ** 2)
        - (distances[1:] ** 2 - distances[0] ** 2)
    )                                              # (N-1,)
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return result  # (2,) = [tx, ty]


def _fit_board_pose_2d(
    local_pts: np.ndarray,                        
    world_pts: np.ndarray,                          
) -> tuple[float, float, float]:
    'Procrustes ↔\n\n (world_origin_x, world_origin_y, world_yaw_deg)\n world_origin (0,0)'
    L_bar = local_pts.mean(axis=0)
    W_bar = world_pts.mean(axis=0)
    dL = local_pts - L_bar
    dW = world_pts - W_bar
    H = dL.T @ dW                                               
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:                              
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    T = W_bar - R @ L_bar                                           
    yaw_deg = math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0])))
    return float(T[0]), float(T[1]), yaw_deg


def locate_boards_from_ruler_measurements(
    boards_cfg: dict[str, dict[str, Any]],
    accumulated_overrides: dict[str, dict[int, list[float]]],
) -> dict[str, dict[int, list[float]]]:
    'ruler_measurements \n\n from_board / from_corner\n to_cornerdistance_m\n\n corner \n · "outer_tl" / "outer_tr" / "outer_bl" / "outer_br" —— \n · 012… —— \n\n \n · to_corner ≥3 from_corner3 \n · 2 to_corner yaw\n\n auto_locate: true ruler_measurements\n auto_locate'
    result: dict[str, dict[int, list[float]]] = {}

    for board_name, board_cfg in boards_cfg.items():
        measurements = board_cfg.get("ruler_measurements")
        if not measurements:
            continue

                                             
        all_known = {**accumulated_overrides, **result}

                                                    
        groups: dict[int | str, list[tuple[np.ndarray, float]]] = {}
        for m in measurements:
            from_board = str(m["from_board"])
            from_corner = m["from_corner"]                       
            to_corner = m["to_corner"]                           
            dist_m = float(m["distance_m"])

                         
            if isinstance(from_corner, str):
                from_corner = from_corner.lower()
            if isinstance(to_corner, str):
                to_corner = to_corner.lower()

            anchor = _get_board_corner_world(from_board, from_corner, boards_cfg, all_known)
            if anchor is None:
                print(
                    f"  [ruler] :  '{from_board}'  '{from_corner}' "
                    f" from_board  world_reference "
                )
                continue
            groups.setdefault(to_corner, []).append((anchor, dist_m))

                                   
        located_local: list[tuple[float, float]] = []
        located_world: list[np.ndarray] = []

        for to_corner, pairs in groups.items():
            anchors = np.array([p[0] for p in pairs])
            dists = np.array([p[1] for p in pairs])
            try:
                world_pos = _trilaterate_2d(anchors, dists)
                lx, ly = _resolve_corner_ref(to_corner, board_cfg)
                located_local.append((lx, ly))
                located_world.append(world_pos)
                print(
                    f"  [ruler]  '{board_name}'  '{to_corner}' → "
                    f" ({world_pos[0]:.4f}, {world_pos[1]:.4f}) m  "
                    f"{len(pairs)} "
                )
            except ValueError as e:
                print(f"  [ruler] :  '{board_name}'  '{to_corner}' — {e}")

        if len(located_local) < 2:
            print(
                f"  [ruler]  '{board_name}':  {len(located_local)} "
                f" 2  to_corner yaw"
            )
            continue

                          
        local_pts = np.array(located_local)
        world_pts = np.array(located_world)
        ox, oy, yaw_deg = _fit_board_pose_2d(local_pts, world_pts)
        print(
            f"  [ruler]  '{board_name}' : "
            f"world_origin=[{ox:.4f}, {oy:.4f}], world_yaw_deg={yaw_deg:.2f}°"
        )

                                                      
                                                                   
        sq_len = float(board_cfg["square_length"])
        sq_x = int(board_cfg["squares_x"])
        yaw_rad = math.radians(yaw_deg)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

                                       
        total = charuco_inner_corner_count(sq_x, int(board_cfg["squares_y"]))
        full_corners: dict[int, list[float]] = {}
        for idx in range(total):
            ilx, ily = charuco_corner_local_xy(idx, sq_x, sq_len)
            wx = ox + ilx * cos_y - ily * sin_y
            wy = oy + ilx * sin_y + ily * cos_y
            full_corners[idx] = [wx, wy]

        result[board_name] = full_corners
        print(f"  [ruler]  '{board_name}'  {total} ")

    return result


def estimate_image_to_bev_homography(
    image_pts: np.ndarray,
    bev_pts: np.ndarray,
    ransac_reproj_threshold: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    'BEV 3×3 \n\n > 4 RANSAC 4 \n (H_3x3, inliers_mask)inliers_mask 4 None'
    if len(image_pts) < 4:
        raise ValueError(f"Need at least 4 point pairs, got {len(image_pts)}")
    h, inliers = cv.findHomography(
        image_pts,
        bev_pts,
        method=cv.RANSAC if len(image_pts) > 4 else 0,
        ransacReprojThreshold=ransac_reproj_threshold,
    )
    if h is None:
        raise RuntimeError("findHomography failed")
    return h.astype(np.float64), inliers

# ──────────────────────────────────────────────────────────────────────────────
                                                                 
# ──────────────────────────────────────────────────────────────────────────────

def normalize_horizon_line(
    line: np.ndarray | list[float] | tuple[float, float, float],
    frame_wh: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return line ``ax+by+c=0`` with ``sqrt(a^2+b^2)=1`` and bottom side positive."""
    arr = np.asarray(line, dtype=np.float64).reshape(3)
    n = float(np.hypot(arr[0], arr[1]))
    if not np.isfinite(n) or n <= 1e-12:
        raise ValueError("horizon line must have a non-zero normal")
    arr = arr / n
    if frame_wh is not None:
        fw, fh = frame_wh
        probe = np.array([float(fw) * 0.5, max(float(fh) - 1.0, 0.0), 1.0])
        if float(arr @ probe) < 0.0:
            arr = -arr
    return arr


def horizon_line_from_points(
    p0: tuple[float, float] | list[float],
    p1: tuple[float, float] | list[float],
    frame_wh: tuple[int, int] | None = None,
) -> np.ndarray:
    """Build a normalized horizon line from two manually clicked image points."""
    a = np.array([float(p0[0]), float(p0[1]), 1.0], dtype=np.float64)
    b = np.array([float(p1[0]), float(p1[1]), 1.0], dtype=np.float64)
    if np.linalg.norm(a[:2] - b[:2]) < 2.0:
        raise ValueError("manual horizon points are too close")
    return normalize_horizon_line(np.cross(a, b), frame_wh)


def homography_horizon_line(
    homography: np.ndarray,
    frame_wh: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return the image horizon induced by an image->BEV homography."""
    h = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    return normalize_horizon_line(h[2, :], frame_wh)


def horizon_line_to_segment(
    line: np.ndarray | list[float],
    frame_wh: tuple[int, int],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip an infinite image line to the image rectangle for drawing."""
    a, b, c = normalize_horizon_line(line, frame_wh)
    fw, fh = frame_wh
    candidates: list[tuple[float, float]] = []
    if abs(b) > 1e-12:
        for x in (0.0, float(fw - 1)):
            y = -(a * x + c) / b
            if -1.0 <= y <= float(fh):
                candidates.append((x, y))
    if abs(a) > 1e-12:
        for y in (0.0, float(fh - 1)):
            x = -(b * y + c) / a
            if -1.0 <= x <= float(fw):
                candidates.append((x, y))
    unique: list[tuple[float, float]] = []
    for p in candidates:
        if not any((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < 1.0 for q in unique):
            unique.append(p)
    if len(unique) < 2:
        return None
    best = (unique[0], unique[1])
    best_d2 = -1.0
    for i, p in enumerate(unique):
        for q in unique[i + 1:]:
            d2 = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
            if d2 > best_d2:
                best = (p, q)
                best_d2 = d2
    return best


def load_manual_horizon_line(
    cam_cfg: dict[str, Any],
    frame_wh: tuple[int, int],
) -> dict[str, Any] | None:
    """Load camera-level ``manual_horizon_line`` and normalize it to bottom-positive sign."""
    raw = cam_cfg.get("manual_horizon_line")
    if not isinstance(raw, dict):
        return None
    points_raw = raw.get("points")
    points: list[list[float]] | None = None
    if isinstance(points_raw, list) and len(points_raw) >= 2:
        points = [
            [float(points_raw[0][0]), float(points_raw[0][1])],
            [float(points_raw[1][0]), float(points_raw[1][1])],
        ]
        line = horizon_line_from_points(points[0], points[1], frame_wh)
    elif raw.get("line") is not None:
        line = normalize_horizon_line(raw["line"], frame_wh)
    else:
        return None
    return {
        "line": [float(v) for v in line.tolist()],
        "points": points,
        "ground_side_positive": True,
    }


def _project_homography_points(homography: np.ndarray, image_pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    h = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    src = np.c_[pts, np.ones(len(pts), dtype=np.float64)]
    dst = (h @ src.T).T
    denom = dst[:, 2:3]
    out = np.full((len(pts), 2), np.nan, dtype=np.float64)
    valid = np.abs(denom[:, 0]) > 1e-12
    out[valid] = dst[valid, :2] / denom[valid]
    return out


def homography_reprojection_rms(
    homography: np.ndarray,
    image_pts: np.ndarray,
    bev_pts: np.ndarray,
) -> float:
    proj = _project_homography_points(homography, image_pts)
    target = np.asarray(bev_pts, dtype=np.float64).reshape(-1, 2)
    valid = np.all(np.isfinite(proj), axis=1)
    if not np.any(valid):
        return float("inf")
    err2 = np.sum((proj[valid] - target[valid]) ** 2, axis=1)
    return float(np.sqrt(np.mean(err2)))


def _sample_line_points(line: np.ndarray, frame_wh: tuple[int, int]) -> np.ndarray:
    a, b, c = normalize_horizon_line(line, frame_wh)
    fw, fh = frame_wh
    pts: list[tuple[float, float]] = []
    if abs(b) >= abs(a) and abs(b) > 1e-12:
        for x in (0.0, float(fw - 1) * 0.5, float(fw - 1)):
            pts.append((x, -(a * x + c) / b))
    elif abs(a) > 1e-12:
        for y in (0.0, float(fh - 1) * 0.5, float(fh - 1)):
            pts.append((-(b * y + c) / a, y))
    else:
        pts = [(0.0, 0.0), (float(fw - 1), 0.0), (float(fw - 1) * 0.5, float(fh - 1) * 0.5)]
    return np.asarray(pts, dtype=np.float64)


def horizon_line_delta_px(
    line_a: np.ndarray | list[float],
    line_b: np.ndarray | list[float],
    frame_wh: tuple[int, int],
) -> float:
    """Symmetric mean distance between two normalized image lines, measured in pixels."""
    la = normalize_horizon_line(line_a, frame_wh)
    lb = normalize_horizon_line(line_b, frame_wh)
    pts_a = _sample_line_points(la, frame_wh)
    pts_b = _sample_line_points(lb, frame_wh)
    da = np.abs(np.c_[pts_a, np.ones(len(pts_a))] @ lb)
    db = np.abs(np.c_[pts_b, np.ones(len(pts_b))] @ la)
    return float(0.5 * (np.mean(da) + np.mean(db)))


def _pack_homography(homography: np.ndarray) -> np.ndarray:
    h = np.asarray(homography, dtype=np.float64).reshape(3, 3).copy()
    if abs(float(h[2, 2])) <= 1e-12:
        raise ValueError("cannot refine homography with near-zero h22")
    h /= h[2, 2]
    return np.array([h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1]], dtype=np.float64)


def _unpack_homography(params: np.ndarray) -> np.ndarray:
    p = np.asarray(params, dtype=np.float64).reshape(8)
    return np.array(
        [[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], 1.0]],
        dtype=np.float64,
    )


def refine_homography(
    image_pts: np.ndarray,
    bev_pts: np.ndarray,
    initial_h: np.ndarray,
    *,
    frame_wh: tuple[int, int],
    manual_horizon_line: np.ndarray | list[float] | None = None,
    lambda_horizon: float = 1.0,
    max_iters: int = 40,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine one image->BEV homography by LM on reprojection and optional horizon residuals.

    This is intentionally self-contained (no SciPy dependency).  ``manual_horizon_line`` is a
    camera-level, independently clicked line; ``manual_homography_horizon`` must not be passed here.
    """
    image_arr = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    bev_arr = np.asarray(bev_pts, dtype=np.float64).reshape(-1, 2)
    if len(image_arr) < 4:
        raise ValueError("Need at least 4 point pairs to refine homography")

    manual_line = None
    if manual_horizon_line is not None and lambda_horizon > 0.0:
        manual_line = normalize_horizon_line(manual_horizon_line, frame_wh)

    def residual(params: np.ndarray) -> np.ndarray:
        h = _unpack_homography(params)
        proj = _project_homography_points(h, image_arr)
        r = (proj - bev_arr).reshape(-1)
        r[~np.isfinite(r)] = 1e6
        if manual_line is not None:
            try:
                h_line = homography_horizon_line(h, frame_wh)
                h_pts = _sample_line_points(h_line, frame_wh)
                h_dist = np.c_[h_pts, np.ones(len(h_pts), dtype=np.float64)] @ manual_line
                r = np.concatenate([r, math.sqrt(lambda_horizon) * h_dist])
            except Exception:
                r = np.concatenate([r, np.full(3, 1e6, dtype=np.float64)])
        return r.astype(np.float64)

    p = _pack_homography(initial_h)
    best_p = p.copy()
    best_r = residual(best_p)
    best_cost = 0.5 * float(best_r @ best_r)
    damping = 1e-3
    accepted_steps = 0

    for iteration in range(max(0, int(max_iters))):
        r0 = residual(p)
        cost0 = 0.5 * float(r0 @ r0)
        jac = np.zeros((len(r0), len(p)), dtype=np.float64)
        for j in range(len(p)):
            step = 1e-6 * max(1.0, abs(float(p[j])))
            pp = p.copy()
            pp[j] += step
            jac[:, j] = (residual(pp) - r0) / step
        diag = np.diag(jac.T @ jac)
        lhs = jac.T @ jac + damping * np.diag(np.maximum(diag, 1.0))
        rhs = -(jac.T @ r0)
        try:
            delta = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            damping *= 10.0
            continue
        if not np.all(np.isfinite(delta)):
            damping *= 10.0
            continue
        candidate = p + delta
        rc = residual(candidate)
        cost_c = 0.5 * float(rc @ rc)
        if cost_c + 1e-9 < cost0:
            p = candidate
            damping = max(damping * 0.3, 1e-9)
            accepted_steps += 1
            if cost_c < best_cost:
                best_cost = cost_c
                best_p = candidate.copy()
                best_r = rc
            if float(np.linalg.norm(delta)) < 1e-10 * (1.0 + float(np.linalg.norm(p))):
                break
        else:
            damping = min(damping * 10.0, 1e12)

    h_before = np.asarray(initial_h, dtype=np.float64).reshape(3, 3)
    h_after = _unpack_homography(best_p)
    diag: dict[str, Any] = {
        "point_count": int(len(image_arr)),
        "iterations": int(max(0, int(max_iters))),
        "accepted_steps": int(accepted_steps),
        "cost_before": float(0.5 * (residual(_pack_homography(h_before)) @ residual(_pack_homography(h_before)))),
        "cost_after": float(best_cost),
        "reproj_rms_before_px": homography_reprojection_rms(h_before, image_arr, bev_arr),
        "reproj_rms_after_px": homography_reprojection_rms(h_after, image_arr, bev_arr),
        "used_horizon_constraint": bool(manual_line is not None),
    }
    if manual_line is not None:
        diag["horizon_delta_before_px"] = horizon_line_delta_px(homography_horizon_line(h_before, frame_wh), manual_line, frame_wh)
        diag["horizon_delta_after_px"] = horizon_line_delta_px(homography_horizon_line(h_after, frame_wh), manual_line, frame_wh)
    diag["improved"] = bool(diag["cost_after"] <= diag["cost_before"] + 1e-9)
    diag["homography_before"] = [[float(v) for v in row] for row in h_before.tolist()]
    diag["homography_after"] = [[float(v) for v in row] for row in h_after.tolist()]
    return h_after, diag


def _backproject_bev_to_image(homography: np.ndarray, bev_pts: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(np.asarray(homography, dtype=np.float64).reshape(3, 3))
    dst = np.c_[np.asarray(bev_pts, dtype=np.float64).reshape(-1, 2), np.ones(len(bev_pts), dtype=np.float64)]
    src = (inv @ dst.T).T
    return src[:, :2] / src[:, 2:3]


def render_calibration_refine_overlay(
    image: np.ndarray,
    image_pts: np.ndarray,
    bev_pts: np.ndarray,
    h_before: np.ndarray,
    h_after: np.ndarray,
    *,
    frame_wh: tuple[int, int],
    manual_horizon_line: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Render before/after point residuals and horizon lines for visual QA."""
    vis = image.copy()
    pts = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    before_img = _backproject_bev_to_image(h_before, bev_pts)
    after_img = _backproject_bev_to_image(h_after, bev_pts)
    for p, b, a in zip(pts, before_img, after_img):
        pi = tuple(np.round(p).astype(int))
        bi = tuple(np.round(b).astype(int))
        ai = tuple(np.round(a).astype(int))
        cv.circle(vis, pi, 4, (0, 255, 255), -1, cv.LINE_AA)
        cv.arrowedLine(vis, pi, bi, (255, 80, 220), 1, cv.LINE_AA, tipLength=0.2)
        cv.arrowedLine(vis, pi, ai, (60, 220, 60), 1, cv.LINE_AA, tipLength=0.2)
    for line, color, label in (
        (homography_horizon_line(h_before, frame_wh), (255, 80, 220), "H before"),
        (homography_horizon_line(h_after, frame_wh), (60, 220, 60), "H refined"),
    ):
        seg = horizon_line_to_segment(line, frame_wh)
        if seg is not None:
            p0, p1 = (tuple(np.round(seg[0]).astype(int)), tuple(np.round(seg[1]).astype(int)))
            cv.line(vis, p0, p1, color, 2, cv.LINE_AA)
            cv.putText(vis, label, p0, cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv.LINE_AA)
    if manual_horizon_line is not None:
        seg = horizon_line_to_segment(manual_horizon_line, frame_wh)
        if seg is not None:
            p0, p1 = (tuple(np.round(seg[0]).astype(int)), tuple(np.round(seg[1]).astype(int)))
            cv.line(vis, p0, p1, (255, 255, 0), 2, cv.LINE_AA)
            cv.putText(vis, "manual horizon", p0, cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv.LINE_AA)
    return vis


def _collect_camera_calibration_points(
    image: np.ndarray,
    cam: dict[str, Any],
    *,
    marker_world_by_id: dict[int, np.ndarray],
    dictionary_name: str,
    charuco_boards_cfg: dict[str, dict[str, Any]],
    board_corner_world_override: dict[str, dict[int, list[float]]],
    manual_corners_by_cam: dict[str, dict[str, dict[int, list[float]]]],
    scale: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    image_pts, bev_pts = load_manual_point_pairs(cam, scale)
    source = "point_pairs"
    if len(image_pts) < 4:
        cam_board_names: list[str] = cam.get("charuco_boards", [])
        cam_manual = manual_corners_by_cam.get(cam["name"])
        if cam_board_names and charuco_boards_cfg:
            image_pts, bev_pts = detect_charuco_points(
                image,
                charuco_boards_cfg,
                cam_board_names,
                scale,
                board_corner_world_override=board_corner_world_override,
                manual_charuco_corners=cam_manual,
            )
            if len(image_pts) >= 4:
                has_manual = cam_manual and any(b in cam_manual for b in cam_board_names)
                source = f"charuco(manual) boards={cam_board_names}" if has_manual else f"charuco boards={cam_board_names}"
    if len(image_pts) < 4:
        aruco_img, aruco_bev, used_ids = detect_aruco_points(image, marker_world_by_id, dictionary_name, scale)
        if len(aruco_img) >= 4:
            image_pts, bev_pts = aruco_img, aruco_bev
            source = f"aruco markers ids={used_ids}"
    return image_pts, bev_pts, source


def run_bfs_joint_refine_preview(
    config_path: str,
    *,
    scene_id: str = "parking",
    diagnostics_dir: str | None = None,
    enable_horizon_refine: bool = False,
    lambda_horizon: float = 1.0,
    max_iters: int = 40,
) -> dict[str, Any]:
    """Dry-run BFS/RANSAC vs refined H and export console/JSON/overlay diagnostics."""
    config_path = str(resolve_project_path(config_path))
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    scale = float(cfg.get("scale_px_per_meter", 100.0))
    dictionary_name = cfg.get("aruco_dictionary", "DICT_4X4_50")
    ransac_threshold = float(cfg.get("ransac_reproj_threshold_px", 4.0))
    out_dir = resolve_project_path(diagnostics_dir or CALIBRATION_DIAGNOSTICS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded_images: dict[str, np.ndarray] = {}
    for cam in cfg["cameras"]:
        name = cam["name"]
        image_path = str(resolve_project_path(cam["image"]))
        img = cv.imread(image_path, cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image for camera {name}: {image_path}")
        camera_matrix, dist_coeffs = load_camera_matrix(cam.get("calibration"))
        if camera_matrix is not None:
            img = cv.undistort(img, camera_matrix, dist_coeffs)
        loaded_images[name] = img

    marker_world_by_id = {int(marker_id): marker_world_corners(marker_cfg) for marker_id, marker_cfg in cfg.get("markers", {}).items()}
    charuco_boards_cfg: dict[str, dict[str, Any]] = cfg.get("charuco_boards", {})
    for board_name, board_cfg in charuco_boards_cfg.items():
        validate_charuco_board_cfg(board_name, board_cfg)
    validate_charuco_board_topology(charuco_boards_cfg)
    cam_board_names_map = {cam["name"]: cam.get("charuco_boards", []) for cam in cfg["cameras"]}
    manual_corners_by_cam: dict[str, dict[str, dict[int, list[float]]]] = {}
    for cam in cfg["cameras"]:
        mc = load_manual_charuco_corners(cam)
        if mc:
            manual_corners_by_cam[cam["name"]] = mc
    board_corner_world_override = auto_locate_charuco_boards(
        loaded_images,
        charuco_boards_cfg,
        cam_board_names_map,
        scale,
        ransac_threshold,
        manual_corners_by_cam=manual_corners_by_cam or None,
    )
    ruler_overrides = locate_boards_from_ruler_measurements(charuco_boards_cfg, board_corner_world_override)
    if ruler_overrides:
        board_corner_world_override.update(ruler_overrides)

    diagnostics: dict[str, Any] = {
        "scene_id": scene_id,
        "config": config_path,
        "scale_px_per_meter": scale,
        "enable_horizon_refine": bool(enable_horizon_refine),
        "lambda_horizon": float(lambda_horizon),
        "cameras": {},
    }
    print("\n[refine-preview] camera       pts  inl  rms_before  rms_after   dH_rel    horizon_before  horizon_after")
    for cam in cfg["cameras"]:
        name = cam["name"]
        image = loaded_images[name]
        frame_wh = (image.shape[1], image.shape[0])
        image_pts, bev_pts, source = _collect_camera_calibration_points(
            image,
            cam,
            marker_world_by_id=marker_world_by_id,
            dictionary_name=dictionary_name,
            charuco_boards_cfg=charuco_boards_cfg,
            board_corner_world_override=board_corner_world_override,
            manual_corners_by_cam=manual_corners_by_cam,
            scale=scale,
        )
        if len(image_pts) < 4:
            raise RuntimeError(f"Camera {name}: not enough calibration points for refine preview")
        h0, inliers = estimate_image_to_bev_homography(image_pts, bev_pts, ransac_threshold)
        inlier_mask = inliers.ravel().astype(bool) if inliers is not None else np.ones(len(image_pts), dtype=bool)
        manual = load_manual_horizon_line(cam, frame_wh) if enable_horizon_refine else None
        h1, diag = refine_homography(
            image_pts[inlier_mask],
            bev_pts[inlier_mask],
            h0,
            frame_wh=frame_wh,
            manual_horizon_line=manual["line"] if manual else None,
            lambda_horizon=lambda_horizon,
            max_iters=max_iters,
        )
        d_h = float(np.linalg.norm((h1 / h1[2, 2]) - (h0 / h0[2, 2])) / max(np.linalg.norm(h0 / h0[2, 2]), 1e-12))
        diag.update({
            "source": source,
            "point_count_total": int(len(image_pts)),
            "inlier_count": int(inlier_mask.sum()),
            "delta_h_rel": d_h,
            "manual_horizon_line": manual,
        })
        diagnostics["cameras"][name] = diag
        overlay = render_calibration_refine_overlay(
            image,
            image_pts[inlier_mask],
            bev_pts[inlier_mask],
            h0,
            h1,
            frame_wh=frame_wh,
            manual_horizon_line=manual["line"] if manual else None,
        )
        imwrite(out_dir / f"{scene_id}_{name}_refine_overlay.jpg", overlay)
        hb = diag.get("horizon_delta_before_px")
        ha = diag.get("horizon_delta_after_px")
        print(
            f"[refine-preview] {name:<12} {len(image_pts):>4} {int(inlier_mask.sum()):>4} "
            f"{diag['reproj_rms_before_px']:>10.3f} {diag['reproj_rms_after_px']:>10.3f} "
            f"{d_h:>8.2e} "
            f"{hb if hb is not None else '-':>14} {ha if ha is not None else '-':>14}"
        )
    json_path = out_dir / f"{scene_id}_refine_preview.json"
    json_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[refine-preview] diagnostics written to {json_path}")
    return diagnostics

def camera_roi_points(image: np.ndarray, cam_cfg: dict[str, Any]) -> np.ndarray:
    '- image_roi ≥ 3\n - image_roi\n\n /FOV ≤ 90° image_roi\n FOV > 90°/'
    if "image_roi" in cam_cfg:
        roi = np.asarray(cam_cfg["image_roi"], dtype=np.float32)
        if roi.ndim != 2 or roi.shape[1] != 2 or len(roi) < 3:
            raise ValueError(f"{cam_cfg['name']}: image_roi must be a polygon of [x, y] points")
        return roi

    height, width = image.shape[:2]
    return np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )


def transform_camera_roi(image: np.ndarray, cam_cfg: dict[str, Any], h: np.ndarray) -> np.ndarray:
    'ROI H BEV'
    roi = camera_roi_points(image, cam_cfg).reshape(-1, 1, 2)
    return cv.perspectiveTransform(roi, h).reshape(-1, 2)


def draw_image_roi_overlay(
    image: np.ndarray,
    cam_cfg: dict[str, Any],
    *,
    line_color: tuple[int, int, int] = (0, 220, 80),
    fill_color: tuple[int, int, int] = (0, 180, 60),
    fill_alpha: float = 0.22,
    point_color: tuple[int, int, int] = (0, 60, 255),
) -> np.ndarray:
    'image_roi pipeline/select_image_roi.py \n\n YAML image_roi \n debug'
    canvas = image.copy()
    if "image_roi" not in cam_cfg:
        return canvas

    roi = camera_roi_points(image, cam_cfg)
    pts = np.round(roi).astype(np.int32).reshape(-1, 1, 2)

    overlay = canvas.copy()
    cv.fillPoly(overlay, [pts], fill_color)
    cv.addWeighted(overlay, fill_alpha, canvas, 1.0 - fill_alpha, 0, canvas)
    cv.polylines(canvas, [pts], True, line_color, 2, cv.LINE_AA)

    for i, (x, y) in enumerate(roi):
        px, py = int(round(x)), int(round(y))
        cv.circle(canvas, (px, py), 5, point_color, -1, cv.LINE_AA)
        cv.circle(canvas, (px, py), 5, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(
            canvas,
            str(i + 1),
            (px + 8, py - 6),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv.LINE_AA,
        )

    cv.putText(
        canvas,
        "image_roi",
        (pts[0, 0, 0] + 4, max(pts[0, 0, 1] - 10, 16)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        line_color,
        2,
        cv.LINE_AA,
    )
    return canvas


def draw_bev_roi_overlay(
    bev_image: np.ndarray,
    roi_bev_pts: np.ndarray,
    *,
    line_color: tuple[int, int, int] = (0, 220, 80),
    point_color: tuple[int, int, int] = (0, 60, 255),
) -> np.ndarray:
    'BEV ROI \n\n roi_bev_pts warpPerspective \n debug'
    canvas = bev_image.copy()
    if len(roi_bev_pts) < 3:
        return canvas

    pts = np.round(roi_bev_pts).astype(np.int32).reshape(-1, 1, 2)
    cv.polylines(canvas, [pts], True, line_color, 2, cv.LINE_AA)

    for i, (x, y) in enumerate(roi_bev_pts):
        px, py = int(round(x)), int(round(y))
        cv.circle(canvas, (px, py), 4, point_color, -1, cv.LINE_AA)
        cv.circle(canvas, (px, py), 4, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(
            canvas,
            str(i + 1),
            (px + 6, py - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv.LINE_AA,
        )

    label_x = int(np.clip(pts[:, 0, 0].min() + 4, 0, canvas.shape[1] - 1))
    label_y = int(np.clip(pts[:, 0, 1].min() - 8, 16, canvas.shape[0] - 1))
    cv.putText(
        canvas,
        "image_roi (BEV)",
        (label_x, label_y),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        line_color,
        2,
        cv.LINE_AA,
    )
    return canvas


                                  
DEFAULT_MAX_CANVAS_SIDE_PX = 50_000
DEFAULT_MAX_CANVAS_PIXELS = 500_000_000                       


def validate_projected_roi(
    camera_name: str,
    image: np.ndarray,
    cam_cfg: dict[str, Any],
    h: np.ndarray,
    *,
    inlier_count: int,
    total_points: int,
    max_side_px: float,
) -> np.ndarray:
    'ROI RuntimeError'
    corners = transform_camera_roi(image, cam_cfg, h)
    if not np.all(np.isfinite(corners)):
        raise RuntimeError(
            f" '{camera_name}'  ROI  NaN/Inf\n"
            f"   {inlier_count}/{total_points} square_length "
            f" tools/diagnose_charuco.py"
        )

    span = corners.max(axis=0) - corners.min(axis=0)
    max_abs = float(np.max(np.abs(corners)))
    if max_side_px > 0 and (span[0] > max_side_px or span[1] > max_side_px or max_abs > max_side_px * 2):
        h_img, w_img = image.shape[:2]
        raise RuntimeError(
            f" '{camera_name}'  ROI "
            f" {span[0]:.0f}px {span[1]:.0f}px {max_side_px:.0f}px\n"
            f"   BEV {np.round(corners).astype(int).tolist()}\n"
            f"   {inlier_count}/{total_points} {w_img}×{h_img}\n"
            f"\n"
            f"  1) YAML  square_length / marker_length  tools/print_ChArUco.py \n"
            f"  2) RANSAC  4 \n"
            f"  3) scale_px_per_meter  100–300\n"
            f" →  →  tools/diagnose_charuco.py  ≥6 "
        )
    return corners


def validate_canvas_size(
    canvas_width: int,
    canvas_height: int,
    *,
    max_side_px: int,
    max_pixels: int,
    camera_corners: list[tuple[str, np.ndarray]],
) -> None:
    pixels = int(canvas_width) * int(canvas_height)
    if canvas_width <= 0 or canvas_height <= 0:
        raise RuntimeError(f"BEV {canvas_width}×{canvas_height}")
    if canvas_width > max_side_px or canvas_height > max_side_px or pixels > max_pixels:
        detail = "\n".join(
            f"  {name}: ROI  {np.round(c.min(axis=0)).astype(int).tolist()}"
            f" ~ {np.round(c.max(axis=0)).astype(int).tolist()}"
            for name, c in camera_corners
        )
        raise RuntimeError(
            f"BEV {canvas_width}×{canvas_height} = {pixels / 1e6:.1f} MP "
            f" {max_side_px}px  / {max_pixels / 1e6:.0f} MP\n"
            f" warpPerspective  {pixels * 3 / 1e9:.1f} GB\n"
            f" ROI \n{detail}\n"
            f"scale_px_per_meter --save-cal"
        )


def feather_mask(mask: np.ndarray, feather_radius: int) -> np.ndarray:
    '[0,1] \n\n = min( / feather_radius, 1.0)\n feather_radius=0 0/1 \n build_blend_weights()'
    if feather_radius <= 0:
        return (mask > 0).astype(np.float32)
    binary = (mask > 0).astype(np.uint8)
    dist = cv.distanceTransform(binary, cv.DIST_L2, 3)
    alpha = np.clip(dist / float(feather_radius), 0.0, 1.0)
    return alpha.astype(np.float32)


def build_blend_weights(
    warped_masks: list[np.ndarray],
    seam_blend_px: int = 15,
) -> list[np.ndarray]:
    '""\n\n feather_mask \n - BEV \n ROI \n - seam_blend_px seam_blend_px \n 3D \n\n \n ① "" BEV \n ② distanceTransform \n ③ seam_signed = d_i - d_j\n |seam_signed| ≤ seam_blend_px \n \n ④ 1\n\n seam_blend_px=0 → 1px \n seam_blend_px=15 → 15px 30px'
    n = len(warped_masks)
    if n == 0:
        return []
    shape = warped_masks[0].shape[:2]

    coverages = [m > 0 for m in warped_masks]

                                         
    excl_masks: list[np.ndarray] = []
    for i, cov_i in enumerate(coverages):
        others = np.zeros(shape, dtype=bool)
        for j, cov_j in enumerate(coverages):
            if j != i:
                others |= cov_j
        excl_masks.append(cov_i & ~others)

                      
                                   
    dist_to_excl: list[np.ndarray] = []
    for i, excl in enumerate(excl_masks):
        ref = excl if excl.any() else coverages[i]
        src = (~ref).astype(np.uint8) * 255
        d = cv.distanceTransform(src, cv.DIST_L2, 5).astype(np.float32)
        dist_to_excl.append(d)

                                  
    weights: list[np.ndarray] = [cov.astype(np.float32) for cov in coverages]

                                           
                                       
    for i in range(n - 1):
        j = i + 1
        ov = coverages[i] & coverages[j]
        if not ov.any():
            continue

        d_i = dist_to_excl[i]
        d_j = dist_to_excl[j]
                                                      
        seam_signed = (d_i - d_j).astype(np.float32)

        if seam_blend_px <= 0:
                                        
            G_i = (d_i <= d_j).astype(np.float32)
        else:
                                                   
            t = np.clip(seam_signed / float(seam_blend_px), -1.0, 1.0)
            G_i = ((1.0 - t) / 2.0).astype(np.float32)

        weights[i] = np.where(ov, G_i, weights[i])
        weights[j] = np.where(ov, 1.0 - G_i, weights[j])

                                                         
                                           
                            
    total = np.zeros(shape, dtype=np.float32)
    for w in weights:
        total += w
    covered = total > 1e-6
    weights = [
        np.divide(w, total, out=np.zeros_like(w), where=covered).astype(np.float32)
        for w in weights
    ]
    return weights


def tune_gain(x: float) -> float:
    'surround-view-system-introduction\n\n > 1< 1'
    if x >= 1.0:
        return x * math.exp((1.0 - x) * 0.5)
    else:
        return x * math.exp((1.0 - x) * 0.8)


def compute_luminance_gains(
    camera_names: list[str],
    warped_images: list[np.ndarray],
    warped_masks: list[np.ndarray],
) -> dict[str, np.ndarray]:
    'BEV per-channel \n\n gain=[1,1,1]\n \n tune_gain() \n\n camera_names BEV \n {camera_name: np.array([gain_B, gain_G, gain_R], dtype=float32)}'
    n = len(camera_names)
    gains: dict[str, np.ndarray] = {camera_names[0]: np.ones(3, dtype=np.float32)}

    for i in range(1, n):
        prev_name = camera_names[i - 1]
        curr_name = camera_names[i]

        overlap = (warped_masks[i - 1] > 0) & (warped_masks[i] > 0)
        if not overlap.any():
            gains[curr_name] = np.ones(3, dtype=np.float32)
            print(f"  [] {prev_name}↔{curr_name}: ")
            continue

                           
        prev_gain = gains[prev_name]
        prev_adj = np.clip(
            warped_images[i - 1].astype(np.float32) * prev_gain.reshape(1, 1, 3),
            0, 255,
        )
        curr_img = warped_images[i].astype(np.float32)

        gain = np.ones(3, dtype=np.float32)
        for c in range(3):  # B, G, R
            med_ref = float(np.median(prev_adj[overlap, c]))
            med_cur = float(np.median(curr_img[overlap, c]))
            if med_cur > 2.0:                   
                raw = med_ref / med_cur
                gain[c] = float(tune_gain(raw))

        gains[curr_name] = gain

    return gains


def blend_images(
    canvas_size: tuple[int, int],
    warped_images: list[np.ndarray],
    warped_masks: list[np.ndarray],
    feather_radius: int = 0,
    *,
    blend_weights: list[np.ndarray] | None = None,
) -> np.ndarray:
    'blend_weights build_blend_weights() \n feather_mask feather_radius'
    width, height = canvas_size
    accum = np.zeros((height, width, 3), dtype=np.float32)

    if blend_weights is not None:
        covered = np.zeros((height, width), dtype=np.float32)
        for image, bw in zip(warped_images, blend_weights):
            accum += image.astype(np.float32) * bw[..., None]
            covered += bw
        result = np.where(covered[..., None] > 1e-6, accum, 0.0)
    else:
        weights_sum = np.zeros((height, width, 1), dtype=np.float32)
        for image, mask in zip(warped_images, warped_masks):
            alpha = feather_mask(mask, feather_radius)[..., None]
            accum += image.astype(np.float32) * alpha
            weights_sum += alpha
        result = accum / np.maximum(weights_sum, 1e-6)
        result[weights_sum[..., 0] <= 1e-6] = 0

    return np.clip(result, 0, 255).astype(np.uint8)


                                                                           

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS camera_calibrations (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_name         TEXT    NOT NULL,
    scene_id            TEXT    NOT NULL DEFAULT 'parking',
    scale_px_per_meter  REAL    NOT NULL,
    homography          BLOB    NOT NULL,
    image_pts           BLOB,
    world_pts           BLOB,
    source              TEXT,
    point_count         INTEGER,
    inlier_count        INTEGER,
    reproj_rms          REAL,
    geometry_version    TEXT,
    calibrated_at       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cam_scene
    ON camera_calibrations(camera_name, scene_id, calibrated_at DESC);

CREATE TABLE IF NOT EXISTS camera_gains (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_name  TEXT    NOT NULL,
    scene_id     TEXT    NOT NULL DEFAULT 'parking',
    gain_b       REAL    NOT NULL DEFAULT 1.0,
    gain_g       REAL    NOT NULL DEFAULT 1.0,
    gain_r       REAL    NOT NULL DEFAULT 1.0,
    computed_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gains_cam_scene
    ON camera_gains(camera_name, scene_id, computed_at DESC);
"""
                                   
                                
                                   
                                                    


def init_db(db_path: str) -> sqlite3.Connection:
    'SQLite'
    conn = sqlite3.connect(db_path)
    conn.executescript(_DB_SCHEMA)
    return conn


def save_calibration(
    conn: sqlite3.Connection,
    camera_name: str,
    scene_id: str,
    scale: float,
    H: np.ndarray,
    image_pts: np.ndarray,
    world_pts: np.ndarray,
    source: str,
    inlier_count: int,
    reproj_rms: float,
    *,
    geometry_version: str = GEOMETRY_VERSION,
    commit: bool = True,
) -> None:
    '``commit=False``'
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO camera_calibrations
            (camera_name, scene_id, scale_px_per_meter, homography,
             image_pts, world_pts, source, point_count, inlier_count,
             reproj_rms, geometry_version, calibrated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            camera_name,
            scene_id,
            scale,
            H.astype(np.float64).tobytes(),
            image_pts.astype(np.float32).tobytes() if image_pts is not None and len(image_pts) else None,
            world_pts.astype(np.float32).tobytes() if world_pts is not None and len(world_pts) else None,
            source,
            len(image_pts) if image_pts is not None else 0,
            inlier_count,
            reproj_rms,
            geometry_version,
            now,
        ),
    )
    if commit:
        conn.commit()
    print(f"  [DB]  '{camera_name}' '{scene_id}'")


def load_calibration(
    conn: sqlite3.Connection,
    camera_name: str,
    scene_id: str,
    *,
    require_geometry_version: str | None = None,
) -> tuple[np.ndarray, float] | None:
    '(H_3x3, scale_px_per_meter) None\n\n ``require_geometry_version``/ fab1 \n ``GEOMETRY_VERSION``\n cla1 fab1 update.md\n tracking/replay/'
    row = conn.execute(
        """
        SELECT homography, scale_px_per_meter, geometry_version
          FROM camera_calibrations
         WHERE camera_name = ? AND scene_id = ?
         ORDER BY calibrated_at DESC
         LIMIT 1
        """,
        (camera_name, scene_id),
    ).fetchone()
    if row is None:
        return None
    if require_geometry_version is not None and row[2] != require_geometry_version:
        raise RuntimeError(
            f" geometry_version : camera={camera_name!r} scene={scene_id!r} "
            f"={row[2]!r} ={require_geometry_version!r}"
            'cla1 fab1 /'
        )
    H = np.frombuffer(row[0], dtype=np.float64).reshape(3, 3).copy()
    scale = float(row[1])
    return H, scale


def save_gains(
    conn: sqlite3.Connection,
    gains: dict[str, np.ndarray],
    scene_id: str,
) -> None:
    'BGR'
    now = datetime.now(timezone.utc).isoformat()
    for camera_name, g in gains.items():
        conn.execute(
            """
            INSERT INTO camera_gains
                (camera_name, scene_id, gain_b, gain_g, gain_r, computed_at)
            VALUES (?,?,?,?,?,?)
            """,
            (camera_name, scene_id, float(g[0]), float(g[1]), float(g[2]), now),
        )
    conn.commit()
    print(f"  [DB] {len(gains)}  '{scene_id}'")


def load_gains(
    conn: sqlite3.Connection,
    camera_names: list[str],
    scene_id: str,
) -> dict[str, np.ndarray]:
    '[1,1,1]'
    result: dict[str, np.ndarray] = {}
    for name in camera_names:
        row = conn.execute(
            """
            SELECT gain_b, gain_g, gain_r
              FROM camera_gains
             WHERE camera_name = ? AND scene_id = ?
             ORDER BY computed_at DESC
             LIMIT 1
            """,
            (name, scene_id),
        ).fetchone()
        if row is not None:
            result[name] = np.array([row[0], row[1], row[2]], dtype=np.float32)
        else:
            result[name] = np.ones(3, dtype=np.float32)
    return result


def list_calibrations(conn: sqlite3.Connection) -> None:
    'See the HSG-SRL reproduction notes.'
    rows = conn.execute(
        """
        SELECT camera_name, scene_id, scale_px_per_meter,
               inlier_count, reproj_rms, geometry_version, calibrated_at
          FROM camera_calibrations
         ORDER BY camera_name, scene_id, calibrated_at DESC
        """
    ).fetchall()
    if not rows:
        print('See the HSG-SRL reproduction notes.')
        return
    header = (
        f"{'See the HSG-SRL reproduction notes.':<12} {'See the HSG-SRL reproduction notes.':<12} {'(px/m)':<14} {'See the HSG-SRL reproduction notes.':<8} "
        f"{'RMS(px)':<14} {'geometry_version':<26} (UTC)"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        inliers = r[3] if r[3] is not None else "-"
        rms = f"{r[4]:.2f}" if r[4] is not None else "-"
        geom_ver = r[5] if r[5] is not None else "-"
        print(
            f"{r[0]:<12} {r[1]:<12} {r[2]:<14.1f} {str(inliers):<8} "
            f"{rms:<14} {geom_ver:<26} {r[6]}"
        )


def run(
    config_path: str,
    *,
    db_path: str | None = None,
    save_cal: bool = False,
    use_cal: bool = False,
    scene_id: str = "parking",
    enable_joint_refine: bool = False,
    enable_horizon_refine: bool = False,
    refine_lambda_horizon: float = 1.0,
    refine_max_iters: int = 40,
    refine_diagnostics_dir: str | None = None,
) -> None:
    'BEV \n\n ─ \n --save-cal ─ + SQLite \n --use-cal ─ +'
    config_path = str(resolve_project_path(config_path))
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    scale = float(cfg.get("scale_px_per_meter", 100.0))
    dictionary_name = cfg.get("aruco_dictionary", "DICT_4X4_50")
    ransac_threshold = float(cfg.get("ransac_reproj_threshold_px", 4.0))
                                                
                                        
    seam_blend_px = int(cfg.get("seam_blend_px", cfg.get("feather_radius_px", 15)))
    output_path = resolve_project_path(cfg.get("output", "bev_mosaic.png"))
    debug_dir = resolve_project_path(cfg.get("debug_dir", "debug_bev"))
    if db_path:
        db_path = str(resolve_project_path(db_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

                                                             
    loaded_images: dict[str, np.ndarray] = {}
    for cam in cfg["cameras"]:
        name = cam["name"]
        image_path = str(resolve_project_path(cam["image"]))
        img = cv.imread(image_path, cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image for camera {name}: {image_path}")
        camera_matrix, dist_coeffs = load_camera_matrix(cam.get("calibration"))
        if camera_matrix is not None:
            img = cv.undistort(img, camera_matrix, dist_coeffs)
        loaded_images[name] = img

    images: list[np.ndarray] = []
    homographies: list[np.ndarray] = []
    camera_cfgs: list[dict[str, Any]] = []
    camera_names: list[str] = []

    if use_cal:
                                                      
        if db_path is None:
            raise ValueError('--use-cal --db')
        conn = init_db(db_path)
        for cam in cfg["cameras"]:
            name = cam["name"]
            result = load_calibration(conn, name, scene_id)
            if result is None:
                raise RuntimeError(
                    f" '{name}' '{scene_id}'\n"
                    f" --save-cal "
                )
            H, db_scale = result
            if abs(db_scale - scale) > 1e-6:
                print(
                    f"  []  '{name}' scale={db_scale}  scale={scale} "
                    f""
                )
                scale = db_scale
            print(f"  [DB]  '{name}' ")
            camera_names.append(name)
            camera_cfgs.append(cam)
            images.append(loaded_images[name])
            homographies.append(H)
        conn.close()

    else:
                                                         
        marker_world_by_id = {
            int(marker_id): marker_world_corners(marker_cfg)
            for marker_id, marker_cfg in cfg.get("markers", {}).items()
        }
        charuco_boards_cfg: dict[str, dict[str, Any]] = cfg.get("charuco_boards", {})
        for board_name, board_cfg in charuco_boards_cfg.items():
            validate_charuco_board_cfg(board_name, board_cfg)
        validate_charuco_board_topology(charuco_boards_cfg)

                                                   
        cam_board_names_map: dict[str, list[str]] = {
            cam["name"]: cam.get("charuco_boards", [])
            for cam in cfg["cameras"]
        }
                                               
        manual_corners_by_cam: dict[str, dict[str, dict[int, list[float]]]] = {}
        for cam in cfg["cameras"]:
            mc = load_manual_charuco_corners(cam)
            if mc:
                manual_corners_by_cam[cam["name"]] = mc
        if manual_corners_by_cam:
            cams_with_manual = [n for n, mc in manual_corners_by_cam.items()]
            print(f"  [manual] {cams_with_manual}")

        board_corner_world_override = auto_locate_charuco_boards(
            loaded_images, charuco_boards_cfg, cam_board_names_map, scale, ransac_threshold,
            manual_corners_by_cam=manual_corners_by_cam or None,
        )

                                                                   
        ruler_overrides = locate_boards_from_ruler_measurements(
            charuco_boards_cfg, board_corner_world_override
        )
        if ruler_overrides:
            board_corner_world_override.update(ruler_overrides)
            print(
                f"[ruler]  {len(ruler_overrides)} "
                f"{list(ruler_overrides.keys())}"
            )

                        
        conn = init_db(db_path) if (save_cal and db_path) else None

        for cam in cfg["cameras"]:
            name = cam["name"]
            image = loaded_images[name]

                       
                                          
                                                                      
                                      
            image_pts, bev_pts = load_manual_point_pairs(cam, scale)
            source = "point_pairs"

            if len(image_pts) < 4:
                cam_board_names: list[str] = cam.get("charuco_boards", [])
                cam_manual = manual_corners_by_cam.get(name)
                if cam_board_names and charuco_boards_cfg:
                    image_pts, bev_pts = detect_charuco_points(
                        image, charuco_boards_cfg, cam_board_names, scale,
                        board_corner_world_override=board_corner_world_override,
                        manual_charuco_corners=cam_manual,
                    )
                    if len(image_pts) >= 4:
                        has_manual = cam_manual and any(
                            b in cam_manual for b in cam_board_names
                        )
                        source = (
                            f"charuco(manual) boards={cam_board_names}"
                            if has_manual
                            else f"charuco boards={cam_board_names}"
                        )

            if len(image_pts) < 4:
                aruco_img, aruco_bev, used_ids = detect_aruco_points(
                    image, marker_world_by_id, dictionary_name, scale
                )
                if len(aruco_img) >= 4:
                    image_pts, bev_pts = aruco_img, aruco_bev
                    source = f"aruco markers ids={used_ids}"

            if len(image_pts) < 4:
                hints = describe_charuco_detection_gaps(
                    image,
                    charuco_boards_cfg,
                    cam.get("charuco_boards", []),
                    board_corner_world_override,
                )
                detail = "\n  ".join(hints) if hints else 'charuco_boards'
                raise RuntimeError(
                    f" '{name}'  4  {len(image_pts)} ={source}\n"
                    f" py tools/diagnose_charuco.py --config {config_path}\n"
                    f" ArUco \n"
                    f"  py tools/mark_charuco_corners.py --config {config_path}\n"
                    f"ChArUco \n  {detail}"
                )

            h_img_to_bev, inliers = estimate_image_to_bev_homography(
                image_pts, bev_pts, ransac_threshold
            )
            inlier_count = int(inliers.sum()) if inliers is not None else len(image_pts)
            if inlier_count < 4:
                raise RuntimeError(f"Camera {name}: too few homography inliers: {inlier_count}")

            min_inliers = int(cfg.get("min_homography_inliers", 6))
            uses_full_frame = "image_roi" not in cam
            if uses_full_frame and inlier_count < min_inliers:
                raise RuntimeError(
                    f" '{name}'  {inlier_count}  ≥ {min_inliers}\n"
                    f"   {len(image_pts)} ={source}\n"
                    f" tools/diagnose_charuco.py "
                    f" image_roi "
                )

            max_roi_side = float(cfg.get("max_canvas_side_px", DEFAULT_MAX_CANVAS_SIDE_PX))
            validate_projected_roi(
                name,
                image,
                cam,
                h_img_to_bev,
                inlier_count=inlier_count,
                total_points=len(image_pts),
                max_side_px=max_roi_side,
            )
            if inlier_count <= 4 and len(image_pts) > 4:
                print(
                    f"  [] {name}: RANSAC  {inlier_count}/{len(image_pts)}"
                    f""
                )

            inlier_mask = (
                inliers.ravel().astype(bool)
                if inliers is not None
                else np.ones(len(image_pts), dtype=bool)
            )
            src_h = np.c_[image_pts[inlier_mask], np.ones(inlier_mask.sum(), dtype=np.float64)]
            proj = (h_img_to_bev @ src_h.T).T
            proj = proj[:, :2] / proj[:, 2:3]
            reproj_rms = float(
                np.sqrt(np.mean(np.sum((proj - bev_pts[inlier_mask]) ** 2, axis=1)))
            )

            refine_requested = bool(enable_joint_refine or enable_horizon_refine)
            if refine_requested:
                frame_wh = (image.shape[1], image.shape[0])
                manual_horizon = load_manual_horizon_line(cam, frame_wh) if enable_horizon_refine else None
                if enable_horizon_refine and manual_horizon is None:
                    print(
                        f"  [refine] {name}:  manual_horizon_line"
                        'reprojection joint refine'
                    )
                h_before_refine = h_img_to_bev.copy()
                h_refined, refine_diag = refine_homography(
                    image_pts[inlier_mask],
                    bev_pts[inlier_mask],
                    h_img_to_bev,
                    frame_wh=frame_wh,
                    manual_horizon_line=manual_horizon["line"] if manual_horizon else None,
                    lambda_horizon=refine_lambda_horizon,
                    max_iters=refine_max_iters,
                )
                h_img_to_bev = h_refined
                reproj_rms = float(refine_diag["reproj_rms_after_px"])
                source += "+joint_refine"
                if manual_horizon is not None:
                    source += "+horizon_refine"
                validate_projected_roi(
                    name,
                    image,
                    cam,
                    h_img_to_bev,
                    inlier_count=inlier_count,
                    total_points=len(image_pts),
                    max_side_px=max_roi_side,
                )
                overlay_dir = resolve_project_path(refine_diagnostics_dir) if refine_diagnostics_dir else debug_dir
                overlay_dir.mkdir(parents=True, exist_ok=True)
                overlay = render_calibration_refine_overlay(
                    image,
                    image_pts[inlier_mask],
                    bev_pts[inlier_mask],
                    h_before_refine,
                    h_img_to_bev,
                    frame_wh=frame_wh,
                    manual_horizon_line=manual_horizon["line"] if manual_horizon else None,
                )
                imwrite(overlay_dir / f"{name}_refine_overlay.jpg", overlay)
                msg = (
                    f"  [refine] {name}: RMS "
                    f"{refine_diag['reproj_rms_before_px']:.3f} -> "
                    f"{refine_diag['reproj_rms_after_px']:.3f}px, "
                    f"accepted_steps={refine_diag['accepted_steps']}"
                )
                if "horizon_delta_before_px" in refine_diag:
                    msg += (
                        f", horizon_delta "
                        f"{refine_diag['horizon_delta_before_px']:.2f} -> "
                        f"{refine_diag['horizon_delta_after_px']:.2f}px"
                    )
                print(msg)

            vis = draw_image_roi_overlay(image, cam)
            for pt in image_pts.reshape(-1, 2):
                cv.circle(vis, tuple(np.round(pt).astype(int)), 5, (0, 255, 255), -1)
            imwrite(debug_dir / f"{name}_detected_points.jpg", vis)

            print(
                f"{name}: source={source}, points={len(image_pts)}, "
                f"inliers={inlier_count}, reproj_rms={reproj_rms:.2f}px"
            )

                                   
            if conn is not None:
                world_pts_m = bev_pts / scale
                save_calibration(
                    conn, name, scene_id, scale, h_img_to_bev,
                    image_pts, world_pts_m, source, inlier_count, reproj_rms,
                )

            camera_names.append(name)
            camera_cfgs.append(cam)
            images.append(image)
            homographies.append(h_img_to_bev)

        if conn is not None:
            conn.close()

                                                      
    projected_corners: list[tuple[str, np.ndarray]] = []
    for image, cam, h in zip(images, camera_cfgs, homographies):
        projected_corners.append(
            (cam["name"], transform_camera_roi(image, cam, h))
        )
    all_corners = np.vstack([c for _, c in projected_corners])
    min_xy = np.floor(all_corners.min(axis=0)).astype(int)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(int)
    margin = int(cfg.get("canvas_margin_px", 50))
    min_xy -= margin
    max_xy += margin

    canvas_width = int(max_xy[0] - min_xy[0] + 1)
    canvas_height = int(max_xy[1] - min_xy[1] + 1)
    max_side = int(cfg.get("max_canvas_side_px", DEFAULT_MAX_CANVAS_SIDE_PX))
    max_pixels = int(cfg.get("max_canvas_pixels", DEFAULT_MAX_CANVAS_PIXELS))
    validate_canvas_size(
        canvas_width,
        canvas_height,
        max_side_px=max_side,
        max_pixels=max_pixels,
        camera_corners=projected_corners,
    )
    print(f"  [BEV]  {canvas_width}×{canvas_height} px "
          f"({canvas_width * canvas_height / 1e6:.1f} MP)")
    translate = np.asarray(
        [[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    warped_images: list[np.ndarray] = []
    warped_masks: list[np.ndarray] = []
    for name, image, cam, h in zip(camera_names, images, camera_cfgs, homographies):
        h_canvas = translate @ h
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        roi = np.round(camera_roi_points(image, cam)).astype(np.int32)
        cv.fillPoly(mask, [roi], 255)
        warped = cv.warpPerspective(
            image,
            h_canvas,
            (canvas_width, canvas_height),
            flags=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        warped_mask = cv.warpPerspective(
            mask,
            h_canvas,
            (canvas_width, canvas_height),
            flags=cv.INTER_NEAREST,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0,
        )
        roi_bev = transform_camera_roi(image, cam, h_canvas)
        warped_dbg = draw_bev_roi_overlay(warped, roi_bev)
        imwrite(debug_dir / f"{name}_bev.jpg", warped_dbg)
        warped_images.append(warped)
        warped_masks.append(warped_mask)

                                                                         
    gains: dict[str, np.ndarray]
    if use_cal and db_path:
                      
        _gc = init_db(db_path)
        gains = load_gains(_gc, camera_names, scene_id)
        _gc.close()
        any_nontrivial = any(not np.allclose(g, 1.0, atol=0.02) for g in gains.values())
        if any_nontrivial:
            for name, g in gains.items():
                print(f"  [] {name}: B={g[0]:.3f} G={g[1]:.3f} R={g[2]:.3f}")
        else:
            print('[] (1.0)')
    else:
                                       
        gains = compute_luminance_gains(camera_names, warped_images, warped_masks)
        for name, g in gains.items():
            if not np.allclose(g, 1.0, atol=0.02):
                print(f"  [] {name}: B={g[0]:.3f} G={g[1]:.3f} R={g[2]:.3f}")
        if save_cal and db_path:
            _gc = init_db(db_path)
            save_gains(_gc, gains, scene_id)
            _gc.close()

                      
    adjusted_images: list[np.ndarray] = []
    for name, img in zip(camera_names, warped_images):
        g = gains.get(name, np.ones(3, dtype=np.float32))
        if np.allclose(g, 1.0, atol=1e-4):
            adjusted_images.append(img)
        else:
            adj = np.clip(
                img.astype(np.float32) * g.reshape(1, 1, 3), 0, 255
            ).astype(np.uint8)
            adjusted_images.append(adj)

                                                           
    blend_weights = build_blend_weights(warped_masks, seam_blend_px)
    for name, bw in zip(camera_names, blend_weights):
        imwrite(
            debug_dir / f"{name}_blend_weight.png",
            (bw * 255).astype(np.uint8),
        )

    mosaic = blend_images(
        (canvas_width, canvas_height),
        adjusted_images,
        warped_masks,
        blend_weights=blend_weights,
    )
    imwrite(output_path, mosaic)
    print(f"Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='BEV N + SQLite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='────────\n \n --save-cal \n --use-cal \n --list-cal \n\n────\n # \n py pipeline/multi_camera_bev_stitch.py --config config/my_scene.yaml --save-cal --db cals.db\n\n # \n py pipeline/multi_camera_bev_stitch.py --config config/my_scene.yaml --use-cal --db cals.db\n\n # \n py pipeline/multi_camera_bev_stitch.py --list-cal --db cals.db\n\n # --scene \n py pipeline/multi_camera_bev_stitch.py --config config/parking_lot.yaml --save-cal --db cals.db --scene parking_lot\n py pipeline/multi_camera_bev_stitch.py --config config/warehouse.yaml --save-cal --db cals.db --scene warehouse',
    )
    parser.add_argument("--config", help='YAML')
    parser.add_argument(
        "--db",
        default="camera_calibrations.db",
        metavar="PATH",
        help='SQLite camera_calibrations.db',
    )
    parser.add_argument(
        "--scene",
        default="parking",
        metavar="ID",
        help='IDparking',
    )
    parser.add_argument(
        "--save-cal",
        action="store_true",
        help='See the HSG-SRL reproduction notes.',
    )
    parser.add_argument(
        "--use-cal",
        action="store_true",
        help='See the HSG-SRL reproduction notes.',
    )
    parser.add_argument(
        "--list-cal",
        action="store_true",
        help='See the HSG-SRL reproduction notes.',
    )
    parser.add_argument(
        "--preview-refine",
        action="store_true",
        help='BFS/RANSAC vs refined H dry-run JSON overlay DB',
    )
    parser.add_argument(
        "--enable-joint-refine",
        action="store_true",
        help='/ reprojection joint refine',
    )
    parser.add_argument(
        "--enable-horizon-refine",
        action="store_true",
        help='manual_horizon_line H joint refine',
    )
    parser.add_argument(
        "--refine-lambda-horizon",
        type=float,
        default=1.0,
        help='manual_horizon_line 1.0',
    )
    parser.add_argument(
        "--refine-max-iters",
        type=int,
        default=40,
        help='H LM 40',
    )
    parser.add_argument(
        "--refine-diagnostics-dir",
        default=None,
        help=f" JSON/overlay  {CALIBRATION_DIAGNOSTICS_DIR}run  debug_dir",
    )
    args = parser.parse_args()
    if args.list_cal:
        conn = init_db(args.db)
        list_calibrations(conn)
        conn.close()
        return

    if args.config is None:
        parser.error('--config --list-cal')

    if args.save_cal and args.use_cal:
        parser.error('--save-cal --use-cal')

    if args.preview_refine and args.use_cal:
        parser.error('--preview-refine / --use-cal')

    if args.preview_refine:
        run_bfs_joint_refine_preview(
            args.config,
            scene_id=args.scene,
            diagnostics_dir=args.refine_diagnostics_dir,
            enable_horizon_refine=args.enable_horizon_refine,
            lambda_horizon=args.refine_lambda_horizon,
            max_iters=args.refine_max_iters,
        )
        return

    run(
        args.config,
        db_path=args.db if (args.save_cal or args.use_cal) else None,
        save_cal=args.save_cal,
        use_cal=args.use_cal,
        scene_id=args.scene,
        enable_joint_refine=args.enable_joint_refine or args.enable_horizon_refine,
        enable_horizon_refine=args.enable_horizon_refine,
        refine_lambda_horizon=args.refine_lambda_horizon,
        refine_max_iters=args.refine_max_iters,
        refine_diagnostics_dir=args.refine_diagnostics_dir,
    )


if __name__ == "__main__":
    main()
