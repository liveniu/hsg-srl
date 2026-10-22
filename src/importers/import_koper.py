#!/usr/bin/env python3
"""Import Ko-PER Sequence1a-1d into one origin database.

The four files share one camera-clock epoch. Object ids are offset per file
and stitched when the same camera has a unique neighbor within about 0.16 s
and 2 m. The annotated point is the front-bumper bottom center. This importer
stores the footprint center

    (x, y) - 0.5 * length * (cos yaw, sin yaw)

in outside_reference_*. Cameras are SK1 and SK4. Classes map car to 2 and
truck to 7. Pedestrians and bicycles are omitted. Ko-PER class 5 is a truck,
not a bus.

    python src/importers/import_koper.py import --all --together --db DEST --batch l0hn_gt_001
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.io as sio

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pipeline.multi_camera_bev_stitch import save_calibration  # noqa: E402
from pipeline.multi_camera_trajectory_fusion import (  # noqa: E402
    DEFAULT_ANCHOR_SHAPE_ETA,
    box_anchor,
    image_point_to_world,
    init_trajectory_db,
    is_bbox_clipped,
    utc_now,
)
from utils.scene_geometry_db import (  # noqa: E402
    load_batch_camera_frame_wh,
    load_import_audit_metadata,
    load_scene_overlap_pairs,
    save_batch_camera_frame_wh,
    save_import_audit_metadata,
    save_scene_overlap_pairs,
)

try:
    from pipeline.geometry_guided_residual_mlp import GeometryDataView  # noqa: E402
except ImportError:
    GeometryDataView = None  # type: ignore[misc, assignment]


DEFAULT_DATA_ROOT = _ROOT / "data" / "external" / "ko_per"
DEFAULT_Z_GROUND = 0.0
DEFAULT_SCALE = 4.0
DEFAULT_S0_M = 60.0
ANCHOR_MODE = "multiplicative_gating_offsetted"
MIN_DEPTH = 0.5
DEFAULT_FRAME_WH = (656, 494)
# viewer  10000 µs
DEFAULT_MAX_SYNC_US = 10_000.0
#  4
FRONT_BOTTOM_REFERENCE_POINT = 4.0

#  GT  YOLO/COCO  id --classes / --gt-class-ids
YOLO_VEHICLE_CLASS_IDS = frozenset({2, 5, 7})
YOLO_VEHICLE_CLASS_NAMES = frozenset({"car", "bus", "truck"})

# Ko-PER FEAT_CLASSIFICATION → YOLO
# 0 1 3  → 25  Ko-PER viewer  Truck
#  YOLO 7 5  bus bus 
KO_PER_TO_YOLO: dict[int, tuple[int, str]] = {
    3: (2, "car"),
    5: (7, "truck"),
}

CAMERAS: tuple[tuple[str, str, str], ...] = (
    ("SK1", "KAB_SK_1_undist", "CALIB_UNDIST_SK_1"),
    ("SK4", "KAB_SK_4_undist", "CALIB_UNDIST_SK_4"),
)

_FEAT_NAMES = (
    "FEAT_X_POS",
    "FEAT_Y_POS",
    "FEAT_Z_POS",
    "FEAT_WIDTH",
    "FEAT_LENGTH",
    "FEAT_HEIGHT",
    "FEAT_ORIENTATION_ANGLE",
    "FEAT_REFERENCEPOINT",
    "FEAT_ID",
    "FEAT_CLASSIFICATION",
)


@dataclass
class CameraCalib:
    name: str
    folder: str
    K: np.ndarray
    dist: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    T_world_to_cam: np.ndarray
    T_cam_to_world: np.ndarray
    frame_wh: tuple[int, int]
    homography: np.ndarray
    reproj_rms: float
    camera_xyz: np.ndarray


@dataclass
class ObservationDraft:
    global_id: int
    camera_name: str
    frame_index: int
    timestamp_sec: float
    image_x: float
    image_y: float
    world_x: float
    world_y: float
    outside_reference_x: float
    outside_reference_y: float
    outside_reference_image_x: float
    outside_reference_image_y: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    confidence: float
    class_id: int
    class_name: str


@dataclass
class LabelFrame:
    time_us: float
    features: np.ndarray  # (N, 10) _FEAT_NAMES


def _configure_stdio_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def list_sequences(data_root: Path) -> list[str]:
    names: list[str] = []
    for path in sorted(data_root.glob("Sequence*")):
        if path.is_dir() and any(path.glob("1_*.mat")) and any(path.glob("3_*.mat")):
            names.append(path.name)
    if not names:
        raise FileNotFoundError(f" Ko-PER : {data_root}")
    return names


def scene_id_for(sequence_name: str) -> str:
    suffix = sequence_name
    if suffix.lower().startswith("sequence"):
        suffix = suffix[len("sequence"):]
    return f"ko_per_{suffix.lower()}"


def _only_mat(seq_dir: Path, prefix: str) -> Path:
    matches = sorted(seq_dir.glob(f"{prefix}_*.mat"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{seq_dir}  {prefix}_*.mat  {len(matches)}")
    return matches[0]


def _as_feature_matrix(features: Any) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, len(_FEAT_NAMES)), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != len(_FEAT_NAMES):
        raise ValueError(f" {arr.shape[1]} {len(_FEAT_NAMES)}")
    return arr


def load_labels(seq_dir: Path) -> list[LabelFrame]:
    mat = sio.loadmat(_only_mat(seq_dir, "3"), squeeze_me=True, struct_as_record=False)
    ref = mat["result"].BoxfittingLabels_REF
    names = [str(x) for x in np.asarray(ref.availableFeatures).ravel()]
    if tuple(names) != _FEAT_NAMES:
        raise ValueError(f" viewer : {names}")
    times = np.asarray(ref.time, dtype=np.float64).ravel()
    frames: list[LabelFrame] = []
    for time_us, item in zip(times, np.asarray(ref.data, dtype=object).ravel()):
        item_names = [str(x) for x in np.asarray(item.availableFeatures).ravel()]
        if tuple(item_names) != _FEAT_NAMES:
            raise ValueError(f": {item_names}")
        frames.append(LabelFrame(float(time_us), _as_feature_matrix(item.features)))
    return frames


def load_camera_times(seq_dir: Path) -> dict[str, np.ndarray]:
    mat = sio.loadmat(_only_mat(seq_dir, "1"), squeeze_me=True, struct_as_record=False)
    result = mat["result"]
    times: dict[str, np.ndarray] = {}
    for cam_name, folder, _calib_key in CAMERAS:
        stream = getattr(result, folder)
        arr = np.asarray(stream.time, dtype=np.float64).ravel()
        if arr.size < 2 or np.any(np.diff(arr) < 0):
            raise ValueError(f"{cam_name} ")
        # Sequence1b/1d  bmp
        times[cam_name] = arr
    return times


def match_frame_index(times_us: np.ndarray, time_us: float, max_sync_us: float) -> int | None:
    """ 0-based  viewer """
    pos = int(np.searchsorted(times_us, time_us))
    candidates = [i for i in (pos - 1, pos) if 0 <= i < len(times_us)]
    if not candidates:
        return None
    best = min(candidates, key=lambda i: abs(float(times_us[i]) - time_us))
    if abs(float(times_us[best]) - time_us) > max_sync_us:
        return None
    return best


def _k_from_fc_cc(fc: np.ndarray, cc: np.ndarray) -> np.ndarray:
    return np.array(
        [[float(fc[0]), 0.0, float(cc[0])],
         [0.0, float(fc[1]), float(cc[1])],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _detect_frame_wh(seq_dir: Path, folder: str) -> tuple[int, int]:
    image_dir = seq_dir / folder
    sample = next(image_dir.glob("*.bmp"), None) if image_dir.is_dir() else None
    if sample is not None:
        image = cv2.imread(str(sample), cv2.IMREAD_UNCHANGED)
        if image is not None and image.ndim >= 2:
            height, width = image.shape[:2]
            return int(width), int(height)
    return DEFAULT_FRAME_WH


def fit_ground_homography(
    T_cam_to_world: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    *,
    z_ground: float,
    grid: int = 9,
) -> tuple[np.ndarray, float]:
    """ z=z_ground image→world"""
    rotation = T_cam_to_world[:3, :3]
    origin = T_cam_to_world[:3, 3]
    image_pts: list[list[float]] = []
    world_pts: list[list[float]] = []
    for v in np.linspace(0.08 * height, 0.92 * height, grid):
        for u in np.linspace(0.08 * width, 0.92 * width, grid):
            ray_cam = np.array(
                [(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0],
                dtype=np.float64,
            )
            direction = rotation @ ray_cam
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9 or abs(float(direction[2])) < 1e-9:
                continue
            direction /= norm
            scale = (z_ground - float(origin[2])) / float(direction[2])
            if scale <= 0:
                continue
            point = origin + scale * direction
            image_pts.append([float(u), float(v)])
            world_pts.append([float(point[0]), float(point[1])])
    if len(image_pts) < 4:
        raise ValueError(" homography ")
    src = np.asarray(image_pts, dtype=np.float32)
    dst = np.asarray(world_pts, dtype=np.float32)
    homography, _mask = cv2.findHomography(src, dst, method=0)
    if homography is None:
        raise ValueError(" findHomography ")
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((projected - dst) ** 2, axis=1))))
    return homography.astype(np.float64), rms


def load_calibrations(data_root: Path, seq_dir: Path, *, z_ground: float) -> dict[str, CameraCalib]:
    calib_path = data_root / "DataSetViewer" / "calib" / "Calib_undist.mat"
    if not calib_path.is_file():
        raise FileNotFoundError(f": {calib_path}")
    mat = sio.loadmat(calib_path, squeeze_me=True, struct_as_record=False)
    calibs: dict[str, CameraCalib] = {}
    for cam_name, folder, key in CAMERAS:
        raw = mat[key]
        if str(raw.DEVICE) != folder.replace("_undist", ""):
            raise ValueError(f"{key}  {raw.DEVICE}  {folder} ")
        kc = np.asarray(raw.kc, dtype=np.float64).ravel()
        if kc.shape != (5,) or float(np.max(np.abs(kc))) != 0.0:
            raise ValueError(f"{cam_name}  *_undist ")
        T_world_to_cam = np.asarray(raw.Tc_homo, dtype=np.float64)
        if T_world_to_cam.shape != (4, 4):
            raise ValueError(f"{cam_name} Tc_homo  {T_world_to_cam.shape}")
        T_cam_to_world = np.linalg.inv(T_world_to_cam)
        K = _k_from_fc_cc(np.asarray(raw.fc, dtype=np.float64), np.asarray(raw.cc, dtype=np.float64))
        rvec, _ = cv2.Rodrigues(T_world_to_cam[:3, :3])
        tvec = T_world_to_cam[:3, 3].reshape(3, 1).astype(np.float64)
        _assert_projection_matches_viewer(K, rvec, tvec, T_world_to_cam, cam_name)
        frame_wh = _detect_frame_wh(seq_dir, folder)
        homography, rms = fit_ground_homography(
            T_cam_to_world, K, frame_wh[0], frame_wh[1], z_ground=z_ground,
        )
        calibs[cam_name] = CameraCalib(
            name=cam_name,
            folder=folder,
            K=K,
            dist=np.zeros(5, dtype=np.float64),
            rvec=rvec,
            tvec=tvec,
            T_world_to_cam=T_world_to_cam,
            T_cam_to_world=T_cam_to_world,
            frame_wh=frame_wh,
            homography=homography,
            reproj_rms=rms,
            camera_xyz=T_cam_to_world[:3, 3].copy(),
        )
    return calibs


def _assert_projection_matches_viewer(
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    T_world_to_cam: np.ndarray,
    cam_name: str,
) -> None:
    """cv2.projectPoints  viewer  world2Image/cam2Image kc=0"""
    world = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    cam = T_world_to_cam @ world
    if abs(float(cam[2])) < 1e-6:
        return
    manual_u = float(K[0, 0] * cam[0] / cam[2] + K[0, 2])
    manual_v = float(K[1, 1] * cam[1] / cam[2] + K[1, 2])
    projected, _ = cv2.projectPoints(
        world[:3].reshape(1, 1, 3), rvec, tvec, K, np.zeros(5, dtype=np.float64),
    )
    err = math.hypot(float(projected[0, 0, 0]) - manual_u, float(projected[0, 0, 1]) - manual_v)
    if err > 1e-3:
        raise RuntimeError(f"{cam_name} OpenCV  viewer  {err:.4f} px")


def bottom_center(x: float, y: float, z: float, length: float, yaw: float) -> np.ndarray:
    """"""
    return np.array(
        [
            x - 0.5 * length * math.cos(yaw),
            y - 0.5 * length * math.sin(yaw),
            z,
        ],
        dtype=np.float64,
    )


def box_corners(
    x: float, y: float, z: float,
    width: float, length: float, height: float, yaw: float,
) -> np.ndarray:
    """ Box3DViewer.calcBoxpoints  (8, 3)"""
    ex = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
    ey = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float64)
    front = np.array([x, y, z], dtype=np.float64)
    bottom = np.stack([
        front + ey * (0.5 * width),
        front - ex * length + ey * (0.5 * width),
        front - ex * length - ey * (0.5 * width),
        front - ey * (0.5 * width),
    ])
    top = bottom + np.array([0.0, 0.0, height], dtype=np.float64)
    return np.vstack([bottom, top])


def _depths(T_world_to_cam: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    cam = (T_world_to_cam[:3, :3] @ points_xyz.T + T_world_to_cam[:3, 3:4]).T
    return cam[:, 2]


def project_world_to_image(
    calib: CameraCalib,
    world_xyz: np.ndarray,
) -> tuple[float, float] | None:
    if float(_depths(calib.T_world_to_cam, world_xyz.reshape(1, 3))[0]) < MIN_DEPTH:
        return None
    projected, _ = cv2.projectPoints(
        world_xyz.reshape(1, 1, 3).astype(np.float64),
        calib.rvec, calib.tvec, calib.K, calib.dist,
    )
    u, v = float(projected[0, 0, 0]), float(projected[0, 0, 1])
    width, height = calib.frame_wh
    if 0.0 <= u < width and 0.0 <= v < height:
        return u, v
    return None


def pixel_bbox_from_box(
    calib: CameraCalib,
    x: float, y: float, z: float,
    width: float, length: float, height: float, yaw: float,
) -> tuple[float, float, float, float] | None:
    corners = box_corners(x, y, z, width, length, height, yaw)
    visible = _depths(calib.T_world_to_cam, corners) > MIN_DEPTH
    if not np.any(visible):
        return None
    projected, _ = cv2.projectPoints(
        corners[visible].reshape(-1, 1, 3),
        calib.rvec, calib.tvec, calib.K, calib.dist,
    )
    pixels = projected.reshape(-1, 2)
    frame_w, frame_h = calib.frame_wh
    x1 = max(0.0, float(pixels[:, 0].min()))
    y1 = max(0.0, float(pixels[:, 1].min()))
    x2 = min(float(frame_w), float(pixels[:, 0].max()))
    y2 = min(float(frame_h), float(pixels[:, 1].max()))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def fit_image_to_world_homographies(
    samples: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]],
    fallback: dict[str, CameraCalib],
    *,
    ransac_thresh_m: float = 2.0,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, int], dict[str, str]]:
    homographies: dict[str, np.ndarray] = {}
    rms: dict[str, float] = {}
    inliers: dict[str, int] = {}
    sources: dict[str, str] = {}
    for cam_name, calib in fallback.items():
        pairs = samples.get(cam_name, [])
        if len(pairs) >= 4:
            src = np.asarray([pair[0] for pair in pairs], dtype=np.float32)
            dst = np.asarray([pair[1] for pair in pairs], dtype=np.float32)
            method = cv2.RANSAC if len(pairs) > 4 else 0
            homography, mask = cv2.findHomography(
                src, dst, method=method, ransacReprojThreshold=ransac_thresh_m,
            )
            if homography is not None:
                pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
                if mask is None:
                    keep = np.ones(len(src), dtype=bool)
                else:
                    keep = mask.reshape(-1).astype(bool)
                    if not np.any(keep):
                        keep = np.ones(len(src), dtype=bool)
                err = np.linalg.norm(pred[keep] - dst[keep], axis=1)
                homographies[cam_name] = homography.astype(np.float64)
                rms[cam_name] = float(np.sqrt(np.mean(err ** 2))) if len(err) else 0.0
                inliers[cam_name] = int(keep.sum())
                sources[cam_name] = "ko_per_bottom_center_fit"
                continue
        homographies[cam_name] = calib.homography
        rms[cam_name] = calib.reproj_rms
        inliers[cam_name] = 0
        sources[cam_name] = "ko_per_ground_plane"
    return homographies, rms, inliers, sources


def localize_homographies(
    homographies: dict[str, np.ndarray],
    sample_world_xy: list[tuple[float, float]],
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    if sample_world_xy:
        origin = np.median(np.asarray(sample_world_xy, dtype=np.float64), axis=0)
    else:
        origin = np.zeros(2, dtype=np.float64)
    translate = np.array(
        [[1.0, 0.0, -origin[0]], [0.0, 1.0, -origin[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    scale_m = np.array(
        [[DEFAULT_SCALE, 0.0, 0.0], [0.0, DEFAULT_SCALE, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    local = {name: scale_m @ translate @ homography for name, homography in homographies.items()}
    return local, origin, DEFAULT_SCALE


def homography_residual_m(
    homography: np.ndarray,
    pairs: list[tuple[tuple[float, float], tuple[float, float]]],
) -> float | None:
    if len(pairs) < 4:
        return None
    src = np.asarray([pair[0] for pair in pairs], dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    pred = cv2.perspectiveTransform(src, homography.astype(np.float64)).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))


def compute_overlap_pairs(
    calibs: dict[str, CameraCalib],
    homography_local: dict[str, np.ndarray],
    *,
    scale: float,
) -> set[tuple[str, str]]:
    bounds: dict[str, tuple[float, float, float, float]] = {}
    for name, calib in calibs.items():
        width, height = calib.frame_wh
        corners = np.array([[[0, 0], [width, 0], [width, height], [0, height]]], dtype=np.float32)
        points = cv2.perspectiveTransform(corners, homography_local[name].astype(np.float64)).reshape(-1, 2)
        points_m = points / scale
        bounds[name] = (
            float(points_m[:, 0].min()), float(points_m[:, 1].min()),
            float(points_m[:, 0].max()), float(points_m[:, 1].max()),
        )
    names = sorted(bounds)
    pairs: set[tuple[str, str]] = set()
    for i, left in enumerate(names):
        ax0, ay0, ax1, ay1 = bounds[left]
        for right in names[i + 1:]:
            bx0, by0, bx1, by1 = bounds[right]
            if ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1:
                pairs.add((left, right) if left < right else (right, left))
    return pairs


def import_sequence(
    sequence_name: str,
    data_root: Path,
    *,
    db_path: Path,
    scene_id: str,
    batch_id: str,
    z_ground: float,
    overwrite: bool,
    anchor_shape_eta: float,
    max_bbox_clip_px: int,
    max_bbox_clip_ratio: float,
    max_sync_us: float,
    h_source_mode: str,
) -> dict[str, Any]:
    seq_dir = data_root / sequence_name
    calibs = load_calibrations(data_root, seq_dir, z_ground=z_ground)
    labels = load_labels(seq_dir)
    camera_times = load_camera_times(seq_dir)
    epoch_us = min(float(times[0]) for times in camera_times.values())

    drafts: list[ObservationDraft] = []
    samples: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]] = defaultdict(list)
    stats: Counter = Counter()

    for label in labels:
        seen_cameras: Counter = Counter()
        for row in label.features:
            stats["label_rows"] += 1
            class_id = int(row[9])
            mapped = KO_PER_TO_YOLO.get(class_id)
            if mapped is None:
                stats["skip_not_vehicle"] += 1
                if class_id == 0:
                    stats["skip_pedestrian"] += 1
                elif class_id == 1:
                    stats["skip_bicycle"] += 1
                else:
                    stats["skip_unknown_class"] += 1
                continue
            if float(row[7]) != FRONT_BOTTOM_REFERENCE_POINT:
                stats["skip_reference_point"] += 1
                continue
            x, y, z = float(row[0]), float(row[1]), float(row[2])
            width, length, height = float(row[3]), float(row[4]), float(row[5])
            yaw = float(row[6])
            global_id = int(row[8])
            if length <= 0.05 or width <= 0.05 or height <= 0.05:
                stats["skip_degenerate_box"] += 1
                continue
            center = bottom_center(x, y, z, length, yaw)
            for cam_name, calib in calibs.items():
                frame_i = match_frame_index(camera_times[cam_name], label.time_us, max_sync_us)
                if frame_i is None:
                    stats["skip_sync"] += 1
                    continue
                ref_img = project_world_to_image(calib, center)
                if ref_img is None:
                    stats["skip_not_visible"] += 1
                    continue
                samples[cam_name].append((ref_img, (float(center[0]), float(center[1]))))
                bbox = pixel_bbox_from_box(calib, x, y, z, width, length, height, yaw)
                if bbox is None:
                    stats["skip_no_bbox"] += 1
                    continue
                bx, by, bw, bh = bbox
                xyxy = np.array([bx, by, bx + bw, by + bh], dtype=np.float64)
                if is_bbox_clipped(xyxy, calib.frame_wh, max_bbox_clip_px, max_bbox_clip_ratio):
                    stats["skip_clipped"] += 1
                    continue
                seen_cameras[global_id] += 1
                yolo_id, class_name = mapped
                if yolo_id not in YOLO_VEHICLE_CLASS_IDS:
                    raise RuntimeError(f" YOLO class_id={yolo_id}")
                drafts.append(ObservationDraft(
                    global_id=global_id,
                    camera_name=cam_name,
                    frame_index=frame_i + 1,
                    timestamp_sec=(label.time_us - epoch_us) * 1e-6,
                    image_x=0.0,
                    image_y=0.0,
                    world_x=0.0,
                    world_y=0.0,
                    outside_reference_x=float(center[0]),
                    outside_reference_y=float(center[1]),
                    outside_reference_image_x=float(ref_img[0]),
                    outside_reference_image_y=float(ref_img[1]),
                    bbox_x=bx,
                    bbox_y=by,
                    bbox_w=bw,
                    bbox_h=bh,
                    confidence=1.0,
                    class_id=yolo_id,
                    class_name=class_name,
                ))
                stats["obs"] += 1
        stats["mvc_candidates"] += sum(1 for count in seen_cameras.values() if count >= 2)

    if not drafts:
        raise RuntimeError(f"{sequence_name}  stats={dict(stats)}")

    sample_world = [(draft.outside_reference_x, draft.outside_reference_y) for draft in drafts]
    if h_source_mode == "model":
        fit_samples: dict[str, list] = {}
    else:
        fit_samples = {cam: list(pairs) for cam, pairs in samples.items()}
    h_base, reproj_rms, h_inliers, h_source = fit_image_to_world_homographies(fit_samples, calibs)
    #  H  H
    gt_h_rms = {
        cam: homography_residual_m(calibs[cam].homography, pairs)
        for cam, pairs in samples.items()
    }
    h_local, origin, scale = localize_homographies(h_base, sample_world)
    for draft in drafts:
        draft.outside_reference_x -= float(origin[0])
        draft.outside_reference_y -= float(origin[1])

    overlap_pairs = compute_overlap_pairs(calibs, h_local, scale=scale)
    geo_err: list[float] = []
    for draft in drafts:
        calib = calibs[draft.camera_name]
        xyxy = np.array(
            [draft.bbox_x, draft.bbox_y, draft.bbox_x + draft.bbox_w, draft.bbox_y + draft.bbox_h],
            dtype=np.float64,
        )
        ax, ay = box_anchor(
            xyxy,
            ANCHOR_MODE,
            homography=h_local[draft.camera_name],
            frame_wh=calib.frame_wh,
            scale=scale,
            risk_scalar_mode="metric_jacobian",
            s0_m=DEFAULT_S0_M,
            shape_eta=anchor_shape_eta,
        )
        draft.image_x, draft.image_y = ax, ay
        world = image_point_to_world(h_local[draft.camera_name], (ax, ay), scale)
        draft.world_x, draft.world_y = float(world[0]), float(world[1])
        geo_err.append(math.hypot(
            draft.world_x - draft.outside_reference_x,
            draft.world_y - draft.outside_reference_y,
        ))

    _write_scene(
        db_path=db_path,
        scene_id=scene_id,
        batch_id=batch_id,
        sequence_name=sequence_name,
        overwrite=overwrite,
        calibs=calibs,
        drafts=drafts,
        overlap_pairs=overlap_pairs,
        h_local=h_local,
        origin=origin,
        scale=scale,
        z_ground=z_ground,
        epoch_us=epoch_us,
        reproj_rms=reproj_rms,
        h_inliers=h_inliers,
        h_source=h_source,
        gt_h_rms=gt_h_rms,
        geo_err=geo_err,
        stats=stats,
        max_sync_us=max_sync_us,
    )
    return {
        "scene_id": scene_id,
        "sequence": sequence_name,
        "observation_count": len(drafts),
        "global_id_count": len({draft.global_id for draft in drafts}),
        "mvc_candidates": int(stats["mvc_candidates"]),
        "overlap_pairs": [list(pair) for pair in sorted(overlap_pairs)],
        "gt_ground_h_rms_m": {
            cam: None if val is None else round(float(val), 6)
            for cam, val in gt_h_rms.items()
        },
        "p_geo_vs_gt_median_m": float(np.median(geo_err)) if geo_err else None,
        "import_stats": dict(stats),
        "camera_xyz": {cam: calib.camera_xyz.tolist() for cam, calib in calibs.items()},
    }


def stitch_file_boundary_tracks(drafts: list[ObservationDraft], *, id_stride: int = 100_000) -> int:
    """Join the same vehicle across Sequence1a–1d file cuts.

    Each label file restarts object ids. A successor is the same vehicle when
    it is the unique nearest bottom center on the same camera, the gap is the
    short file boundary, and the separation is at most 2 m.
    """
    tracks: dict[tuple[str, int], list[float]] = {}
    for draft in drafts:
        key = (draft.camera_name, draft.global_id)
        slot = tracks.get(key)
        if slot is None:
            tracks[key] = [
                draft.timestamp_sec, draft.timestamp_sec,
                draft.outside_reference_x, draft.outside_reference_y,
                draft.outside_reference_x, draft.outside_reference_y,
            ]
            continue
        if draft.timestamp_sec < slot[0]:
            slot[0] = draft.timestamp_sec
            slot[4] = draft.outside_reference_x
            slot[5] = draft.outside_reference_y
        if draft.timestamp_sec >= slot[1]:
            slot[1] = draft.timestamp_sec
            slot[2] = draft.outside_reference_x
            slot[3] = draft.outside_reference_y
    by_part: dict[tuple[str, int], list[tuple[int, list[float]]]] = defaultdict(list)
    for (camera, gid), slot in tracks.items():
        by_part[(camera, gid // id_stride)].append((gid, slot))
    parent = {gid: gid for (_cam, gid) in tracks}

    def find(gid: int) -> int:
        while parent[gid] != gid:
            parent[gid] = parent[parent[gid]]
            gid = parent[gid]
        return gid

    links = 0
    cameras = sorted({cam for cam, _gid in tracks})
    parts = sorted({gid // id_stride for _cam, gid in tracks})
    for camera in cameras:
        for part in parts[:-1]:
            ends = by_part.get((camera, part), [])
            starts = by_part.get((camera, part + 1), [])
            if not ends or not starts:
                continue
            end_cut = max(slot[1] for _gid, slot in ends)
            start_cut = min(slot[0] for _gid, slot in starts)
            end_set = [(gid, slot) for gid, slot in ends if end_cut - slot[1] <= 0.4]
            start_set = [(gid, slot) for gid, slot in starts if slot[0] - start_cut <= 0.4]
            used: set[int] = set()
            for gid, slot in end_set:
                ranked = []
                for gid2, slot2 in start_set:
                    if gid2 in used:
                        continue
                    gap = slot2[0] - slot[1]
                    if gap < 0.0 or gap > 0.5:
                        continue
                    dist = math.hypot(slot[2] - slot2[4], slot[3] - slot2[5])
                    ranked.append((dist, gid2))
                ranked.sort()
                if not ranked or ranked[0][0] > 2.0:
                    continue
                if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 0.5 and ranked[1][0] <= 2.0:
                    continue
                parent[find(ranked[0][1])] = find(gid)
                used.add(ranked[0][1])
                links += 1
    if links:
        for draft in drafts:
            draft.global_id = find(draft.global_id)
    return links


def import_recording(
    data_root: Path,
    *,
    db_path: Path,
    scene_id: str,
    batch_id: str,
    z_ground: float,
    overwrite: bool,
    anchor_shape_eta: float,
    max_bbox_clip_px: int,
    max_bbox_clip_ratio: float,
    max_sync_us: float,
    h_source_mode: str,
) -> dict[str, Any]:
    """Concatenate Sequence1a–1d into one recording and one world origin."""
    sequences = list_sequences(data_root)
    clocks = {name: load_camera_times(data_root / name) for name in sequences}
    epoch_us = min(float(times[0]) for clock in clocks.values() for times in clock.values())
    calibs = load_calibrations(data_root, data_root / sequences[0], z_ground=z_ground)
    drafts: list[ObservationDraft] = []
    samples: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]] = defaultdict(list)
    stats: Counter = Counter()
    frame_base = {cam_name: 0 for cam_name, _folder, _key in CAMERAS}
    for part, sequence_name in enumerate(sequences):
        labels = load_labels(data_root / sequence_name)
        camera_times = clocks[sequence_name]
        id_offset = part * 100_000
        for label in labels:
            seen_cameras: Counter = Counter()
            for row in label.features:
                stats["label_rows"] += 1
                class_id = int(row[9])
                mapped = KO_PER_TO_YOLO.get(class_id)
                if mapped is None:
                    stats["skip_not_vehicle"] += 1
                    if class_id == 0:
                        stats["skip_pedestrian"] += 1
                    elif class_id == 1:
                        stats["skip_bicycle"] += 1
                    else:
                        stats["skip_unknown_class"] += 1
                    continue
                if float(row[7]) != FRONT_BOTTOM_REFERENCE_POINT:
                    stats["skip_reference_point"] += 1
                    continue
                x, y, z = float(row[0]), float(row[1]), float(row[2])
                width, length, height = float(row[3]), float(row[4]), float(row[5])
                yaw = float(row[6])
                global_id = int(row[8]) + id_offset
                if length <= 0.05 or width <= 0.05 or height <= 0.05:
                    stats["skip_degenerate_box"] += 1
                    continue
                center = bottom_center(x, y, z, length, yaw)
                for cam_name, calib in calibs.items():
                    frame_i = match_frame_index(camera_times[cam_name], label.time_us, max_sync_us)
                    if frame_i is None:
                        stats["skip_sync"] += 1
                        continue
                    ref_img = project_world_to_image(calib, center)
                    if ref_img is None:
                        stats["skip_not_visible"] += 1
                        continue
                    samples[cam_name].append((ref_img, (float(center[0]), float(center[1]))))
                    bbox = pixel_bbox_from_box(calib, x, y, z, width, length, height, yaw)
                    if bbox is None:
                        stats["skip_no_bbox"] += 1
                        continue
                    bx, by, bw, bh = bbox
                    xyxy = np.array([bx, by, bx + bw, by + bh], dtype=np.float64)
                    if is_bbox_clipped(xyxy, calib.frame_wh, max_bbox_clip_px, max_bbox_clip_ratio):
                        stats["skip_clipped"] += 1
                        continue
                    seen_cameras[global_id] += 1
                    yolo_id, class_name = mapped
                    drafts.append(ObservationDraft(
                        global_id=global_id,
                        camera_name=cam_name,
                        frame_index=frame_base[cam_name] + frame_i + 1,
                        timestamp_sec=(label.time_us - epoch_us) * 1e-6,
                        image_x=0.0,
                        image_y=0.0,
                        world_x=0.0,
                        world_y=0.0,
                        outside_reference_x=float(center[0]),
                        outside_reference_y=float(center[1]),
                        outside_reference_image_x=float(ref_img[0]),
                        outside_reference_image_y=float(ref_img[1]),
                        bbox_x=bx,
                        bbox_y=by,
                        bbox_w=bw,
                        bbox_h=bh,
                        confidence=1.0,
                        class_id=yolo_id,
                        class_name=class_name,
                    ))
                    stats["obs"] += 1
            stats["mvc_candidates"] += sum(1 for count in seen_cameras.values() if count >= 2)
        for cam_name in frame_base:
            frame_base[cam_name] += int(camera_times[cam_name].shape[0])
    linked = stitch_file_boundary_tracks(drafts)
    stats["boundary_links"] = linked
    print(f"  : {linked}", flush=True)
    if not drafts:
        raise RuntimeError(f" stats={dict(stats)}")
    sample_world = [(draft.outside_reference_x, draft.outside_reference_y) for draft in drafts]
    fit_samples = {} if h_source_mode == "model" else {cam: list(pairs) for cam, pairs in samples.items()}
    h_base, reproj_rms, h_inliers, h_source = fit_image_to_world_homographies(fit_samples, calibs)
    gt_h_rms = {
        cam: homography_residual_m(calibs[cam].homography, pairs)
        for cam, pairs in samples.items()
    }
    h_local, origin, scale = localize_homographies(h_base, sample_world)
    for draft in drafts:
        draft.outside_reference_x -= float(origin[0])
        draft.outside_reference_y -= float(origin[1])
    overlap_pairs = compute_overlap_pairs(calibs, h_local, scale=scale)
    geo_err: list[float] = []
    for draft in drafts:
        calib = calibs[draft.camera_name]
        xyxy = np.array(
            [draft.bbox_x, draft.bbox_y, draft.bbox_x + draft.bbox_w, draft.bbox_y + draft.bbox_h],
            dtype=np.float64,
        )
        ax, ay = box_anchor(
            xyxy, ANCHOR_MODE,
            homography=h_local[draft.camera_name],
            frame_wh=calib.frame_wh,
            scale=scale,
            risk_scalar_mode="metric_jacobian",
            s0_m=DEFAULT_S0_M,
            shape_eta=anchor_shape_eta,
        )
        draft.image_x, draft.image_y = ax, ay
        world = image_point_to_world(h_local[draft.camera_name], (ax, ay), scale)
        draft.world_x, draft.world_y = float(world[0]), float(world[1])
        geo_err.append(math.hypot(
            draft.world_x - draft.outside_reference_x,
            draft.world_y - draft.outside_reference_y,
        ))
    _write_scene(
        db_path=db_path,
        scene_id=scene_id,
        batch_id=batch_id,
        sequence_name="+".join(sequences),
        overwrite=overwrite,
        calibs=calibs,
        drafts=drafts,
        overlap_pairs=overlap_pairs,
        h_local=h_local,
        origin=origin,
        scale=scale,
        z_ground=z_ground,
        epoch_us=epoch_us,
        reproj_rms=reproj_rms,
        h_inliers=h_inliers,
        h_source=h_source,
        gt_h_rms=gt_h_rms,
        geo_err=geo_err,
        stats=stats,
        max_sync_us=max_sync_us,
    )
    return {
        "scene_id": scene_id,
        "sequence": "+".join(sequences),
        "observation_count": len(drafts),
        "global_id_count": len({draft.global_id for draft in drafts}),
        "mvc_candidates": int(stats["mvc_candidates"]),
        "overlap_pairs": [list(pair) for pair in sorted(overlap_pairs)],
        "time_start_s": min(draft.timestamp_sec for draft in drafts),
        "time_end_s": max(draft.timestamp_sec for draft in drafts),
        "p_geo_vs_gt_median_m": float(np.median(geo_err)) if geo_err else None,
        "import_stats": dict(stats),
    }


def _write_scene(
    *,
    db_path: Path,
    scene_id: str,
    batch_id: str,
    sequence_name: str,
    overwrite: bool,
    calibs: dict[str, CameraCalib],
    drafts: list[ObservationDraft],
    overlap_pairs: set[tuple[str, str]],
    h_local: dict[str, np.ndarray],
    origin: np.ndarray,
    scale: float,
    z_ground: float,
    epoch_us: float,
    reproj_rms: dict[str, float],
    h_inliers: dict[str, int],
    h_source: dict[str, str],
    gt_h_rms: dict[str, float | None],
    geo_err: list[float],
    stats: Counter,
    max_sync_us: float,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = init_trajectory_db(str(db_path), batch_id)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(trajectory_observations)").fetchall()}
    required = {"outside_reference_image_x", "outside_reference_image_y"}
    if not required.issubset(columns):
        conn.close()
        raise RuntimeError(f"{db_path}  {sorted(required)} import")

    if overwrite:
        for table in (
            "trajectory_observations", "trajectory_global_tracks",
            "trajectory_merges", "trajectory_runtime_stats", "trajectory_camera_config",
        ):
            conn.execute(f"DELETE FROM {table} WHERE scene_id=? AND batch_id=?", (scene_id, batch_id))
        for table in ("camera_calibrations", "scene_canvas", "scene_overlap_pairs"):
            conn.execute(f"DELETE FROM {table} WHERE scene_id=?", (scene_id,))
    else:
        existing = conn.execute(
            "SELECT COUNT(*) FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
            (scene_id, batch_id),
        ).fetchone()
        if existing and int(existing[0]) > 0:
            conn.close()
            raise RuntimeError(
                f"scene_id={scene_id} batch_id={batch_id}  {int(existing[0])} "
                " --no-overwrite "
            )

    for cam_name, calib in calibs.items():
        save_calibration(
            conn, cam_name, scene_id, scale, h_local[cam_name],
            image_pts=np.zeros((0, 2), dtype=np.float32),
            world_pts=np.zeros((0, 2), dtype=np.float32),
            source=f"{h_source.get(cam_name, 'ko_per')}:{sequence_name}",
            inlier_count=h_inliers.get(cam_name, 0),
            reproj_rms=reproj_rms.get(cam_name, calib.reproj_rms),
            commit=False,
        )

    now = utc_now()
    rows: list[tuple[Any, ...]] = []
    local_ids: dict[tuple[str, int], int] = {}
    next_local: dict[str, int] = defaultdict(int)
    global_ids: set[int] = set()
    world_pts: list[tuple[float, float]] = []
    for draft in drafts:
        key = (draft.camera_name, draft.global_id)
        if key not in local_ids:
            next_local[draft.camera_name] += 1
            local_ids[key] = next_local[draft.camera_name]
        global_ids.add(draft.global_id)
        world_pts.append((draft.world_x, draft.world_y))
        rows.append((
            scene_id, batch_id, draft.camera_name, local_ids[key], draft.global_id,
            draft.frame_index, draft.timestamp_sec,
            draft.image_x, draft.image_y, draft.world_x, draft.world_y,
            draft.outside_reference_x, draft.outside_reference_y,
            draft.outside_reference_image_x, draft.outside_reference_image_y,
            draft.bbox_x, draft.bbox_y, draft.bbox_w, draft.bbox_h,
            draft.confidence, draft.class_id, draft.class_name, now,
        ))
    conn.executemany(
        """
        INSERT INTO trajectory_observations
            (scene_id, batch_id, camera_name, local_track_id, global_id, frame_index,
             timestamp_sec, image_x, image_y, world_x, world_y,
             outside_reference_x, outside_reference_y,
             outside_reference_image_x, outside_reference_image_y,
             bbox_x, bbox_y, bbox_w, bbox_h, confidence,
             class_id, class_name, recorded_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        rows,
    )
    conn.executemany(
        """
        INSERT OR IGNORE INTO trajectory_global_tracks
            (scene_id, batch_id, global_id, created_at, source)
        VALUES (?,?,?,?,?)
        """,
        [(scene_id, batch_id, gid, now, f"ko_per:{sequence_name}") for gid in sorted(global_ids)],
    )
    if world_pts:
        pts = np.asarray(world_pts, dtype=np.float64)
        margin_m = 5.0
        min_xy = pts.min(axis=0) - margin_m
        max_xy = pts.max(axis=0) + margin_m
        canvas_w = int(np.ceil((max_xy[0] - min_xy[0]) * scale)) + 1
        canvas_h = int(np.ceil((max_xy[1] - min_xy[1]) * scale)) + 1
        conn.execute(
            """
            INSERT OR REPLACE INTO scene_canvas
                (scene_id, scale, min_x, min_y, canvas_w, canvas_h, saved_at)
            VALUES (?,?,?,?,?,?,?)
            """,
            (scene_id, scale, float(min_xy[0]), float(min_xy[1]), canvas_w, canvas_h, now),
        )
    save_scene_overlap_pairs(conn, scene_id, overlap_pairs, replace=True)
    save_batch_camera_frame_wh(
        conn, scene_id, batch_id,
        {cam: calib.frame_wh for cam, calib in calibs.items()},
    )
    geo = np.asarray(geo_err, dtype=np.float64)
    save_import_audit_metadata(conn, scene_id, batch_id, {
        "scene_id": scene_id,
        "sequence": sequence_name,
        "cameras": sorted(calibs),
        "observation_count": len(drafts),
        "global_id_count": len(global_ids),
        "mvc_candidates": int(stats["mvc_candidates"]),
        "overlap_pairs": [list(pair) for pair in sorted(overlap_pairs)],
        "origin_world": origin.tolist(),
        "scale": scale,
        "z_ground": z_ground,
        "epoch_us": epoch_us,
        "timestamp_origin": "camera_stream_first_us",
        "max_sync_us": max_sync_us,
        "classes": "yolo:2=car,7=truck",
        "yolo_vehicle_class_ids": sorted(YOLO_VEHICLE_CLASS_IDS),
        "yolo_class_5_bus": "absent_in_ko_per_labels",
        "excluded_classes": ["pedestrian", "bicycle"],
        "coordinate_frame": "ko_per_world_minus_origin",
        "box_reference": "front_bottom_center_refpoint_4",
        "reference_source": "gt_3d_bottom_center",
        "reference_image_source": "undistorted_pinhole_bottom_center",
        "world_source": "p_geo",
        "frame_wh_by_camera": {cam: list(calib.frame_wh) for cam, calib in calibs.items()},
        "camera_xyz": {cam: calib.camera_xyz.tolist() for cam, calib in calibs.items()},
        "reproj_rms": reproj_rms,
        "homography_source": h_source,
        "homography_inliers": h_inliers,
        "gt_ground_h_rms_m": gt_h_rms,
        "p_geo_vs_gt_median_m": float(np.median(geo)) if geo.size else None,
        "p_geo_vs_gt_p95_m": float(np.percentile(geo, 95)) if geo.size else None,
        "import_stats": dict(stats),
    })
    conn.commit()
    conn.close()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def validate_import(
    db_path: Path,
    scene_id: str,
    batch_id: str,
    *,
    max_time_gap: float = 0.12,
    bad_dist_m: float = 80.0,
) -> dict[str, Any]:
    if not db_path.is_file() or db_path.stat().st_size == 0:
        raise FileNotFoundError(f": {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
            (scene_id, batch_id),
        ).fetchone()
        if count is None or int(count[0]) == 0:
            raise RuntimeError(f"scene_id={scene_id} batch_id={batch_id} ")
        classes = dict(conn.execute(
            """SELECT class_name, COUNT(*) FROM trajectory_observations
               WHERE scene_id=? AND batch_id=? GROUP BY class_name""",
            (scene_id, batch_id),
        ).fetchall())
        class_ids = dict(conn.execute(
            """SELECT class_id, COUNT(*) FROM trajectory_observations
               WHERE scene_id=? AND batch_id=? GROUP BY class_id""",
            (scene_id, batch_id),
        ).fetchall())
        forbidden = [
            f"id={cid} n={n}" for cid, n in class_ids.items()
            if int(cid) not in YOLO_VEHICLE_CLASS_IDS
        ]
        forbidden.extend(
            name for name in classes if name not in YOLO_VEHICLE_CLASS_NAMES
        )
        cameras = [
            row[0] for row in conn.execute(
                """SELECT DISTINCT camera_name FROM trajectory_observations
                   WHERE scene_id=? AND batch_id=? ORDER BY camera_name""",
                (scene_id, batch_id),
            )
        ]
        gid_count = conn.execute(
            "SELECT COUNT(DISTINCT global_id) FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
            (scene_id, batch_id),
        ).fetchone()[0]
        multi_gid = conn.execute(
            """SELECT COUNT(*) FROM (
                   SELECT global_id FROM trajectory_observations
                   WHERE scene_id=? AND batch_id=?
                   GROUP BY global_id HAVING COUNT(DISTINCT camera_name) >= 2
               )""",
            (scene_id, batch_id),
        ).fetchone()[0]
        geo_rows = conn.execute(
            """SELECT world_x, world_y, outside_reference_x, outside_reference_y
               FROM trajectory_observations
               WHERE scene_id=? AND batch_id=? AND outside_reference_x IS NOT NULL""",
            (scene_id, batch_id),
        ).fetchall()
        geo = [
            math.hypot(wx - rx, wy - ry)
            for wx, wy, rx, ry in geo_rows
        ]
        cross_rows = conn.execute(
            """SELECT a.outside_reference_x, a.outside_reference_y,
                      b.outside_reference_x, b.outside_reference_y
               FROM trajectory_observations a
               JOIN trajectory_observations b
                 ON a.scene_id=b.scene_id AND a.batch_id=b.batch_id
                AND a.global_id=b.global_id AND a.timestamp_sec=b.timestamp_sec
                AND a.camera_name < b.camera_name
               WHERE a.scene_id=? AND a.batch_id=?""",
            (scene_id, batch_id),
        ).fetchall()
        cross = [
            math.hypot(ax - bx, ay - by)
            for ax, ay, bx, by in cross_rows
        ]
        overlap_pairs = load_scene_overlap_pairs(conn, scene_id, cameras)
        audit = load_import_audit_metadata(conn, scene_id, batch_id)
        warnings: list[str] = []
        if forbidden:
            warnings.append(f": {forbidden}")
        if not overlap_pairs and len(cameras) >= 2:
            warnings.append("DB  scene_overlap_pairs")
        mvc_count = 0
        spatial_count = 0
        dists: list[float] = []
        if GeometryDataView is None:
            warnings.append(" GeometryDataView MVC ")
        else:
            view = GeometryDataView()
            calibrations = view.load_camera_calibrations(conn, scene_id, cameras)
            db_wh = load_batch_camera_frame_wh(conn, scene_id, batch_id, cameras)
            observations = list(view.iter_observations_from_db(
                conn, scene_id, batch_id, cameras, calibrations,
                frame_wh_by_camera=db_wh,
            ))
            pairs = view.mine_mvc_pairs(observations, overlap_pairs, max_time_gap_sec=max_time_gap)
            mvc_count = len(pairs)
            for pair in pairs:
                ax, ay = pair.obs_a.world_geo
                bx, by = pair.obs_b.world_geo
                dist = float(np.hypot(ax - bx, ay - by))
                dists.append(dist)
                if dist <= bad_dist_m:
                    spatial_count += 1
            if mvc_count == 0:
                warnings.append("MVC  0")
    finally:
        conn.close()
    return {
        "scene_id": scene_id,
        "observation_count": int(count[0]),
        "class_counts": classes,
        "class_id_counts": {str(int(cid)): int(n) for cid, n in class_ids.items()},
        "camera_count": len(cameras),
        "global_id_count": int(gid_count),
        "multi_camera_global_ids": int(multi_gid),
        "overlap_pairs": sorted(overlap_pairs),
        "cross_camera_gt_pairs": len(cross),
        "cross_camera_gt_max_m": max(cross) if cross else None,
        "p_geo_vs_gt_median_m": _percentile(geo, 50),
        "p_geo_vs_gt_p95_m": _percentile(geo, 95),
        "gt_ground_h_rms_m": audit.get("gt_ground_h_rms_m"),
        "mvc_pair_count": mvc_count,
        "spatial_mvc_pair_count": spatial_count,
        "mvc_median_dist_m": _percentile(dists, 50),
        "warnings": warnings,
    }


def print_validation_report(report: dict[str, Any]) -> None:
    print(f"\n=== Ko-PER : {report['scene_id']} ===")
    print(f"          : {report['observation_count']}")
    print(f"            : {report['class_counts']}")
    print(f"  YOLO id       : {report['class_id_counts']}")
    print(f"          : {report['camera_count']}")
    print(f"   ID     : {report['global_id_count']}")
    print(f"   ID   : {report['multi_camera_global_ids']}")
    print(f"      : {report['overlap_pairs']}")
    print(f"   GT   : {report['cross_camera_gt_pairs']}")
    if report.get("cross_camera_gt_max_m") is not None:
        print(f"   GT : {report['cross_camera_gt_max_m']:.6f} m")
    if report.get("gt_ground_h_rms_m"):
        print(f"   H  GT RMS: {report['gt_ground_h_rms_m']}")
    if report.get("p_geo_vs_gt_median_m") is not None:
        print(
            f"  P_geo−GT  : {report['p_geo_vs_gt_median_m']:.3f} m"
            f"  p95={report['p_geo_vs_gt_p95_m']:.3f} m"
        )
    print(f"  MVC     : {report['mvc_pair_count']}")
    if report.get("mvc_median_dist_m") is not None:
        print(f"  MVC median d  : {report['mvc_median_dist_m']:.3f} m")
    for warning in report.get("warnings", []):
        print(f"  [WARN] {warning}")


def cmd_import(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    if getattr(args, "together", False):
        scene_id = args.scene_id or "ko_per"
        print(f"\n>>>  -> scene_id={scene_id}", flush=True)
        meta = import_recording(
            data_root,
            db_path=Path(args.db),
            scene_id=scene_id,
            batch_id=args.batch,
            z_ground=args.z_ground,
            overwrite=not args.no_overwrite,
            anchor_shape_eta=args.anchor_shape_eta,
            max_bbox_clip_px=args.clip_px,
            max_bbox_clip_ratio=args.clip_ratio,
            max_sync_us=args.max_sync_us,
            h_source_mode=args.h_source,
        )
        print(
            f"  obs={meta['observation_count']}  global_id={meta['global_id_count']}  "
            f"t={meta['time_start_s']:.1f}..{meta['time_end_s']:.1f}s  "
            f"P_geo-GT={meta['p_geo_vs_gt_median_m']:.3f} m",
            flush=True,
        )
        return
    if args.all and args.scene_id:
        raise SystemExit("--all  --scene-id ")
    sequences = list_sequences(data_root) if args.all else [args.sequence]
    db_path = Path(args.db)
    started = time.perf_counter()
    failures: list[tuple[str, str]] = []
    for sequence_name in sequences:
        if sequence_name not in list_sequences(data_root):
            raise SystemExit(f" {sequence_name}: {list_sequences(data_root)}")
        scene_id = args.scene_id or scene_id_for(sequence_name)
        print(f"\n>>>  {sequence_name} -> scene_id={scene_id}")
        try:
            meta = import_sequence(
                sequence_name, data_root,
                db_path=db_path,
                scene_id=scene_id,
                batch_id=args.batch,
                z_ground=args.z_ground,
                overwrite=not args.no_overwrite,
                anchor_shape_eta=args.anchor_shape_eta,
                max_bbox_clip_px=args.clip_px,
                max_bbox_clip_ratio=args.clip_ratio,
                max_sync_us=args.max_sync_us,
                h_source_mode=args.h_source,
            )
            print(
                f"  obs={meta['observation_count']}  "
                f"global_id={meta['global_id_count']}  "
                f"mvc_candidates={meta['mvc_candidates']}  "
                f"overlap={meta['overlap_pairs']}"
            )
            print(f"  HGT RMS(m): {meta['gt_ground_h_rms_m']}")
            print(f"  P_geo−GT (m): {meta['p_geo_vs_gt_median_m']:.3f}")
            print(
                "  : "
                f"ped={meta['import_stats'].get('skip_pedestrian', 0)}  "
                f"bike={meta['import_stats'].get('skip_bicycle', 0)}  "
                f"clipped={meta['import_stats'].get('skip_clipped', 0)}  "
                f"sync={meta['import_stats'].get('skip_sync', 0)}"
            )
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            failures.append((sequence_name, str(exc)))
            if args.fail_fast:
                raise SystemExit(1) from exc
    print(f"\n ({time.perf_counter() - started:.1f}s)  DB={db_path}  fail={len(failures)}")
    if failures:
        for sequence_name, message in failures:
            print(f"  [FAIL] {sequence_name}: {message}")
        raise SystemExit(1)


def cmd_validate(args: argparse.Namespace) -> None:
    report = validate_import(
        Path(args.db), args.scene, args.batch,
        max_time_gap=args.max_time_gap,
        bad_dist_m=args.bad_dist_m,
    )
    print_validation_report(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  JSON -> {args.json}")


def cmd_inspect(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    conn = sqlite3.connect(str(db_path))
    print(f"=== inspect scene={args.scene} batch={args.batch} ===")
    for row in conn.execute(
        """SELECT camera_name, class_name, COUNT(*),
                  MIN(timestamp_sec), MAX(timestamp_sec)
           FROM trajectory_observations
           WHERE scene_id=? AND batch_id=?
           GROUP BY camera_name, class_name
           ORDER BY camera_name, class_name""",
        (args.scene, args.batch),
    ):
        print(f"  {row[0]} {row[1]}: n={row[2]} t=[{row[3]:.3f},{row[4]:.3f}]")
    audit = load_import_audit_metadata(conn, args.scene, args.batch)
    if audit:
        print(f"  gt_ground_h_rms_m: {audit.get('gt_ground_h_rms_m')}")
        print(f"  p_geo_vs_gt_median_m: {audit.get('p_geo_vs_gt_median_m')}")
        print(f"  import_stats: {audit.get('import_stats')}")
    conn.close()


def render_preview(
    sequence_name: str,
    data_root: Path,
    *,
    label_index: int,
    out_dir: Path,
) -> list[Path]:
    """/ 3D """
    seq_dir = data_root / sequence_name
    calibs = load_calibrations(data_root, seq_dir, z_ground=DEFAULT_Z_GROUND)
    labels = load_labels(seq_dir)
    if label_index < 0 or label_index >= len(labels):
        raise IndexError(f"label index {label_index}  0..{len(labels) - 1}")
    label = labels[label_index]
    camera_mat = sio.loadmat(_only_mat(seq_dir, "1"), squeeze_me=True, struct_as_record=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    for cam_name, folder, _key in CAMERAS:
        stream = getattr(camera_mat["result"], folder)
        times = np.asarray(stream.time, dtype=np.float64).ravel()
        frame_i = match_frame_index(times, label.time_us, DEFAULT_MAX_SYNC_US)
        if frame_i is None:
            continue
        image_name = str(stream.data[frame_i].image)
        image = cv2.imread(str(seq_dir / folder / image_name), cv2.IMREAD_COLOR)
        if image is None:
            continue
        calib = calibs[cam_name]
        for row in label.features:
            if int(row[9]) not in KO_PER_TO_YOLO:
                continue
            x, y, z = float(row[0]), float(row[1]), float(row[2])
            width, length, height, yaw = float(row[3]), float(row[4]), float(row[5]), float(row[6])
            corners = box_corners(x, y, z, width, length, height, yaw)
            depth = _depths(calib.T_world_to_cam, corners)
            projected, _ = cv2.projectPoints(
                corners.reshape(-1, 1, 3), calib.rvec, calib.tvec, calib.K, calib.dist,
            )
            pixels = projected.reshape(-1, 2)
            color = (0, 180, 255) if int(row[9]) == 3 else (0, 255, 255)
            for a, b in edges:
                if depth[a] > MIN_DEPTH and depth[b] > MIN_DEPTH:
                    cv2.line(
                        image,
                        (int(round(pixels[a, 0])), int(round(pixels[a, 1]))),
                        (int(round(pixels[b, 0])), int(round(pixels[b, 1]))),
                        color, 1, cv2.LINE_AA,
                    )
            center = bottom_center(x, y, z, length, yaw)
            ref = project_world_to_image(calib, center)
            if ref is not None:
                cv2.circle(image, (int(round(ref[0])), int(round(ref[1]))), 3, (0, 0, 255), -1)
        out_path = out_dir / f"preview_{sequence_name}_{cam_name}_f{label_index}.png"
        cv2.imwrite(str(out_path), image)
        written.append(out_path)
    return written


def cmd_preview(args: argparse.Namespace) -> None:
    paths = render_preview(
        args.sequence, Path(args.data_root),
        label_index=args.frame, out_dir=Path(args.out),
    )
    for path in paths:
        print(path)


def main() -> None:
    _configure_stdio_utf8()
    parser = argparse.ArgumentParser(description="Ko-PER  GT -> SQLite")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_import = sub.add_parser("import", help="")
    p_import.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    group = p_import.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence", help=" Sequence1a")
    group.add_argument("--all", action="store_true")
    group.add_argument("--together", action="store_true", help=" scene_id=ko_per")
    p_import.add_argument("--db", required=True)
    p_import.add_argument("--batch", required=True)
    p_import.add_argument("--scene-id", default=None)
    p_import.add_argument("--z-ground", type=float, default=DEFAULT_Z_GROUND)
    p_import.add_argument("--no-overwrite", action="store_true")
    p_import.add_argument("--anchor-shape-eta", type=float, default=DEFAULT_ANCHOR_SHAPE_ETA)
    p_import.add_argument("--clip-px", type=int, default=3)
    p_import.add_argument("--clip-ratio", type=float, default=0.01)
    p_import.add_argument("--max-sync-us", type=float, default=DEFAULT_MAX_SYNC_US)
    p_import.add_argument("--h-source", choices=("model", "data"), default="model")
    p_import.add_argument("--fail-fast", action="store_true")
    p_import.set_defaults(func=cmd_import)

    p_val = sub.add_parser("validate", help=" GT  MVC")
    p_val.add_argument("--db", required=True)
    p_val.add_argument("--scene", required=True)
    p_val.add_argument("--batch", required=True)
    p_val.add_argument("--max-time-gap", type=float, default=0.12)
    p_val.add_argument("--bad-dist-m", type=float, default=80.0)
    p_val.add_argument("--json", default=None)
    p_val.set_defaults(func=cmd_validate)

    p_ins = sub.add_parser("inspect", help="")
    p_ins.add_argument("--db", required=True)
    p_ins.add_argument("--scene", required=True)
    p_ins.add_argument("--batch", required=True)
    p_ins.set_defaults(func=cmd_inspect)

    p_prev = sub.add_parser("preview", help=" 3D ")
    p_prev.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p_prev.add_argument("--sequence", default="Sequence1a")
    p_prev.add_argument("--frame", type=int, default=51)
    p_prev.add_argument("--out", default=str(_ROOT / "experiments" / "ko_per" / "media"))
    p_prev.set_defaults(func=cmd_preview)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
