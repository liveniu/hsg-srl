#!/usr/bin/env python3
'Synthehicle overlapping SQLite\n MLP MVC \n\n────────\n- overlapping-O- Town 3–8 Cxx \n- train Town01–Town05 out_bbox/{frame:06d}.txt\n 2D bbox + 3D outside_reference\n out_bbox 2D 3D box 8 amodal\n ~50%\n gt/gt.txt \n vehicle_id --bbox-source gt_tight amodal \n 3D out_bboxgt.txt \n- test Town06Town07Town10HD gt/gt.txtMOT 2D\n 3D import 2D outside_reference NULL\n- calibration/overlapping/{Town}/camera_info/camera_{N}.txt\n (intrinsic_matrix 3×3 + extrinsic_matrix 4×4world→camera )\n- --z-ground autotrain 3D 4 z \n model H GT Town02 z=0\n out_bbox 0.0\n- vehicle_id global_id\n- FPS=10timestamp_sec = (frame_index - 1) / FPS\n- 1920×1080\n\n────\n # \n py experiments/synthehicle/import_synthehicle_mvc.py import \\\n --scene Town01-O-dawn \\\n --db experiments/synthehicle/synth_Town01_O_dawn.db \\\n --batch synth_import_001\n\n # overlapping \n py experiments/synthehicle/import_synthehicle_mvc.py import --all-train \\\n --db experiments/synthehicle/synth_train_all.db \\\n --batch synth_import_001\n\n # overlapping 2D GT\n py experiments/synthehicle/import_synthehicle_mvc.py import --all-test \\\n --db experiments/synthehicle/synth_test_all.db \\\n --batch synth_test_001\n\n # \n py experiments/synthehicle/import_synthehicle_mvc.py validate \\\n --db experiments/synthehicle/synth_Town01_O_dawn.db \\\n --scene synth_Town01_O_dawn --batch synth_import_001\n\n # \n py experiments/synthehicle/import_synthehicle_mvc.py inspect \\\n --db experiments/synthehicle/synth_Town01_O_dawn.db \\\n --scene synth_Town01_O_dawn --batch synth_import_001'

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

_ROOT = Path(__file__).resolve().parents[2]
_SYNTH_DIR = Path(__file__).resolve().parent
_SRC = Path(__file__).resolve().parents[1]
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

                                                                             

DEFAULT_DATA_ROOT = _ROOT / "data" / "external" / "synthehicle_core"
FRAME_RATE = 10.0
FRAME_W, FRAME_H = 1920, 1080
DEFAULT_Z_GROUND = 0.0
DEFAULT_SCALE = 4.0
DEFAULT_S0_M = 60.0
ANCHOR_MODE = "multiplicative_gating_offsetted"

                            
OVERLAP_MARKER = "-O-"

                                                      
                                
SYNTH_CLASS_MAP: dict[int, tuple[int, str]] = {
    0: (2, "car"),
    1: (7, "truck"),
    2: (2, "van"),
    3: (5, "bus"),
    4: (3, "motorcycle"),
    5: (2, "vehicle"),   # emergency
    6: (2, "vehicle"),   # misc
    7: (2, "vehicle"),   # police
    8: (2, "vehicle"),   # ambulance
    9: (2, "vehicle"),   # firetruck / other
}
DEFAULT_CLASS = (2, "vehicle")

                                                                             


@dataclass
class CameraCalib:
    name: str
    K: np.ndarray                      
    T_world_to_cam: np.ndarray                                     
    T_cam_to_world: np.ndarray                                 
    frame_wh: tuple[int, int]
    homography: np.ndarray                                      
    reproj_rms: float


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
    outside_reference_x: float | None                                 
    outside_reference_y: float | None              
    outside_reference_image_x: float | None              
    outside_reference_image_y: float | None              
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    confidence: float
    class_id: int
    class_name: str


                                                                            


def load_camera_calib(
    calib_dir: Path,
    camera_idx: int,
    *,
    z_ground: float = DEFAULT_Z_GROUND,
) -> CameraCalib:
    'JSON homography'
    cam_file = calib_dir / "camera_info" / f"camera_{camera_idx}.txt"
    if not cam_file.is_file():
        raise FileNotFoundError(f": {cam_file}")
    data = json.loads(cam_file.read_text(encoding="utf-8"))
    K = np.array(data["intrinsic_matrix"], dtype=np.float64)   # 3×3
    E = np.array(data["extrinsic_matrix"], dtype=np.float64)   # 4×4 world→cam
    T_cam_to_world = np.linalg.inv(E)

    name = f"C{camera_idx:02d}"
    try:
        h_mat, reproj = fit_ground_homography(T_cam_to_world, K, FRAME_W, FRAME_H, z_ground=z_ground)
    except ValueError:
        h_mat = np.eye(3, dtype=np.float64)
        reproj = float("inf")
    return CameraCalib(
        name=name,
        K=K,
        T_world_to_cam=E,
        T_cam_to_world=T_cam_to_world,
        frame_wh=(FRAME_W, FRAME_H),
        homography=h_mat,
        reproj_rms=reproj,
    )


def discover_scene_cameras(scene_dir: Path) -> list[str]:
    'C01, C02, …'
    cams = sorted(
        p.name
        for p in scene_dir.iterdir()
        if p.is_dir() and p.name.startswith("C") and p.name[1:].isdigit()
    )
    if not cams:
        raise FileNotFoundError(f" Cxx : {scene_dir}")
    return cams


def discover_town_camera_indices(calib_root: Path, town: str) -> list[int]:
    'calibration/overlapping/{Town}/camera_info'
    info_dir = calib_root / "overlapping" / town / "camera_info"
    if not info_dir.is_dir():
        raise FileNotFoundError(f": {info_dir}")
    indices: list[int] = []
    for p in info_dir.glob("camera_*.txt"):
        try:
            indices.append(int(p.stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    if not indices:
        raise FileNotFoundError(f" camera_*.txt: {info_dir}")
    return sorted(indices)


def load_all_calibs(
    calib_root: Path,
    town: str,
    *,
    camera_names: list[str] | None = None,
    z_ground: float = DEFAULT_Z_GROUND,
) -> dict[str, CameraCalib]:
    'Town {camera_name: CameraCalib}\n\n ``camera_names`` C01–C08\n Town camera_*.txt'
    calib_dir = calib_root / "overlapping" / town
    available = discover_town_camera_indices(calib_root, town)
    if camera_names is None:
        indices = available
    else:
        wanted = []
        for name in camera_names:
            idx = int(name[1:])
            if idx not in available:
                raise FileNotFoundError(
                    f" {name}  {town}  "
                    f"(available={[f'C{i:02d}' for i in available]})"
                )
            wanted.append(idx)
        indices = sorted(wanted)
    calibs: dict[str, CameraCalib] = {}
    for idx in indices:
        c = load_camera_calib(calib_dir, idx, z_ground=z_ground)
        calibs[c.name] = c
    return calibs


                                                                              


def _carla_ray_cam(K: np.ndarray, u: float, v: float) -> np.ndarray:
    'Synthehicle / CARLA camera \n\n CARLA X = Y = Z = \n u = fx * Y/X + cxv = fy * (-Z/X) + cy\n (u,v) camera [1, (u-cx)/fx, -(v-cy)/fy]'
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([1.0, (u - cx) / fx, -(v - cy) / fy], dtype=np.float64)


def fit_ground_homography(
    T_cam_to_world: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    *,
    z_ground: float = DEFAULT_Z_GROUND,
    grid: int = 9,
) -> tuple[np.ndarray, float]:
    'z=z_ground image→world homography\n\n CARLA X=Y=Z= pinhole'
    o_world = (T_cam_to_world @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
    img_pts: list[list[float]] = []
    world_pts: list[list[float]] = []
    for v in np.linspace(0.08 * height, 0.92 * height, grid):
        for u in np.linspace(0.08 * width, 0.92 * width, grid):
            ray_cam = _carla_ray_cam(K, u, v)
            d_world = (T_cam_to_world @ np.append(ray_cam, 0.0))[:3]
            norm = np.linalg.norm(d_world)
            if norm < 1e-9:
                continue
            d_world /= norm
            if abs(d_world[2]) < 1e-9:
                continue
            s = (z_ground - o_world[2]) / d_world[2]
            if s <= 0:
                continue
            p = o_world + s * d_world
            img_pts.append([u, v])
            world_pts.append([p[0], p[1]])
    if len(img_pts) < 4:
        raise ValueError('homography')
    src = np.array(img_pts, dtype=np.float32)
    dst = np.array(world_pts, dtype=np.float32)
    h_mat, _ = cv2.findHomography(src, dst, method=0)
    if h_mat is None:
        raise ValueError('findHomography')
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h_mat).reshape(-1, 2)
    reproj = float(np.sqrt(np.mean(np.linalg.norm(proj - dst, axis=1) ** 2)))
    return h_mat.astype(np.float64), reproj


def fit_image_to_world_homographies(
    samples: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]],
    fallback_calibs: dict[str, CameraCalib],
    *,
    ransac_thresh_m: float = 2.0,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, int], dict[str, str]]:
    'GT image→world H H'
    h_by_cam: dict[str, np.ndarray] = {}
    reproj_by_cam: dict[str, float] = {}
    inliers_by_cam: dict[str, int] = {}
    source_by_cam: dict[str, str] = {}
    for cam_name, calib in fallback_calibs.items():
        pairs = samples.get(cam_name, [])
        if len(pairs) >= 4:
            src = np.asarray([p[0] for p in pairs], dtype=np.float32)
            dst = np.asarray([p[1] for p in pairs], dtype=np.float32)
            method = cv2.RANSAC if len(pairs) > 4 else 0
            h_mat, mask = cv2.findHomography(src, dst, method=method, ransacReprojThreshold=ransac_thresh_m)
            if h_mat is not None:
                pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h_mat).reshape(-1, 2)
                inlier_mask = mask.reshape(-1).astype(bool) if mask is not None else np.ones(len(src), dtype=bool)
                if not np.any(inlier_mask):
                    inlier_mask = np.ones(len(src), dtype=bool)
                err = np.linalg.norm(pred[inlier_mask] - dst[inlier_mask], axis=1)
                h_by_cam[cam_name] = h_mat.astype(np.float64)
                reproj_by_cam[cam_name] = float(np.sqrt(np.mean(err**2))) if len(err) else 0.0
                inliers_by_cam[cam_name] = int(inlier_mask.sum())
                source_by_cam[cam_name] = "synth_bottom_center_fit"
                continue
        h_by_cam[cam_name] = calib.homography
        reproj_by_cam[cam_name] = calib.reproj_rms
        inliers_by_cam[cam_name] = 0
        source_by_cam[cam_name] = "synth_ground_plane_fallback"
    return h_by_cam, reproj_by_cam, inliers_by_cam, source_by_cam


def localize_homographies(
    h_base_by_cam: dict[str, np.ndarray],
    sample_world_xy: list[tuple[float, float]],
) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    'scale_px_per_meter'
    if sample_world_xy:
        origin = np.median(np.array(sample_world_xy, dtype=np.float64), axis=0)
    else:
        origin = np.zeros(2, dtype=np.float64)
    T_trans = np.array(
        [[1.0, 0.0, -origin[0]], [0.0, 1.0, -origin[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    T_scale = np.array(
        [[DEFAULT_SCALE, 0.0, 0.0], [0.0, DEFAULT_SCALE, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    h_local = {name: T_scale @ T_trans @ h for name, h in h_base_by_cam.items()}
    return h_local, origin, DEFAULT_SCALE


def compute_overlap_pairs(
    calibs: dict[str, CameraCalib],
    h_local: dict[str, np.ndarray],
    *,
    scale: float,
) -> set[tuple[str, str]]:
    'BEV AABB'
    world_bounds: dict[str, tuple[float, float, float, float]] = {}
    for name, calib in calibs.items():
        w, h = calib.frame_wh
        corners_img = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float32)
        pts = cv2.perspectiveTransform(corners_img, h_local[name].astype(np.float64)).reshape(-1, 2)
        pts_m = pts / scale
        world_bounds[name] = (
            float(pts_m[:, 0].min()),
            float(pts_m[:, 1].min()),
            float(pts_m[:, 0].max()),
            float(pts_m[:, 1].max()),
        )
    names = sorted(world_bounds)
    pairs: set[tuple[str, str]] = set()
    for i, a in enumerate(names):
        ax0, ay0, ax1, ay1 = world_bounds[a]
        for b in names[i + 1:]:
            bx0, by0, bx1, by1 = world_bounds[b]
            if ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1:
                pairs.add(tuple(sorted((a, b))))
    return pairs


                                                                            


def bottom_z_from_world_coords(wc: Any) -> float:
    '3D 8 4 zCARLA'
    z = np.asarray(wc[2][:8], dtype=np.float64)
    return float(np.mean(np.sort(z)[:4]))


def estimate_road_z_from_out_bbox(
    scene_dir: Path,
    camera_names: list[str],
) -> tuple[float, dict[str, Any]]:
    'out_bbox 3D z'
    zs: list[float] = []
    for cam in camera_names:
        bbox_dir = scene_dir / cam / "out_bbox"
        if not bbox_dir.is_dir():
            continue
        for fp in bbox_dir.glob("*.txt"):
            data = json.loads(fp.read_text(encoding="utf-8"))
            for wc in data.get("world_coords") or []:
                zs.append(bottom_z_from_world_coords(wc))
    if not zs:
        return DEFAULT_Z_GROUND, {"source": "default_no_3d", "n": 0}
    arr = np.asarray(zs, dtype=np.float64)
    z = float(np.median(arr))
    return z, {
        "source": "median_3d_bottom_corners",
        "n": int(len(arr)),
        "median": z,
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def parse_z_ground(value: str) -> float | str:
    if str(value).strip().lower() == "auto":
        return "auto"
    return float(value)


def parse_bbox_file(path: Path) -> dict[str, Any]:
    'out_bbox/{frame:06d}.txt 3D JSON \n\n \n bboxes: list of [x1, y1, x2, y2]\n world_coords: list of (bottom_cx, bottom_cy) # CARLA \n bottom_image: list of (u, v) | None # \n vehicle_ids: list of int\n class_ids: list of int'
    data = json.loads(path.read_text(encoding="utf-8"))
    bboxes: list[list[float]] = []
    world_coords: list[tuple[float, float]] = []
    vehicle_ids: list[int] = []
    class_ids: list[int] = []
    n = len(data["vehicle_id"])
    for i in range(n):
        bb = data["bboxes"][i]  # [[x1, y1], [x2, y2]]
        bboxes.append([bb[0][0], bb[0][1], bb[1][0], bb[1][1]])
        wc = data["world_coords"][i]  # [[x0..x7], [y0..y7], [z0..z7], [1..1]]
                                      
        bx = float(np.mean(wc[0][:4]))
        by = float(np.mean(wc[1][:4]))
        world_coords.append((bx, by))
        vehicle_ids.append(int(data["vehicle_id"][i]))
        class_ids.append(int(data["vehicle_class"][i]))
    return {
        "bboxes": bboxes,
        "world_coords": world_coords,
        "vehicle_ids": vehicle_ids,
        "class_ids": class_ids,
    }


def parse_gt_file(path: Path) -> dict[int, list[dict[str, Any]]]:
    'gt/gt.txtMOT frame,id,x,y,w,h,conf,class,-1,-1\n\n {frame_index: [{"id", "bbox_xyxy", "class_id"}]}'
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split(",")
        if len(cols) < 6:
            continue
        frame = int(cols[0])
        obj_id = int(cols[1])
        x = float(cols[2])
        y = float(cols[3])
        w = float(cols[4])
        h = float(cols[5])
        class_id = int(cols[7]) if len(cols) > 7 else 0
        result[frame].append({
            "id": obj_id,
            "bbox_xyxy": [x, y, x + w, y + h],
            "class_id": class_id,
        })
    return dict(result)


def project_world_to_image(
    T_world_to_cam: np.ndarray,
    K: np.ndarray,
    world_xyz: np.ndarray,
    frame_wh: tuple[int, int],
) -> tuple[float, float] | None:
    '3D None\n\n CARLA X=Y=Z=\n u = fx * p_cam[1] / p_cam[0] + cx\n v = fy * (-p_cam[2]) / p_cam[0] + cy'
    p_cam = T_world_to_cam @ np.append(world_xyz, 1.0)
    if p_cam[0] <= 0.5:                     
        return None
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * p_cam[1] / p_cam[0] + cx
    v = fy * (-p_cam[2]) / p_cam[0] + cy
    w, h = frame_wh
    if 0 <= u < w and 0 <= v < h:
        return float(u), float(v)
    return None


                                                                            


def discover_overlapping_scenes(data_root: Path, split: str) -> list[Path]:
    "split'train' 'test' overlapping"
    split_dir = data_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split : {split_dir}")
    scenes = sorted(p for p in split_dir.iterdir() if p.is_dir() and OVERLAP_MARKER in p.name)
    if not scenes:
        raise FileNotFoundError(f" overlapping : {split_dir}")
    return scenes


def scene_to_town(scene_name: str) -> str:
    'Town01-O-dawn Town Town01'
    return scene_name.split("-")[0]


def scene_to_id(scene_name: str) -> str:
    'scene_id synth_Town01_O_dawn'
    return "synth_" + scene_name.replace("-", "_")


                                                                            


def import_scene(
    scene_dir: Path,
    calib_root: Path,
    *,
    db_path: Path,
    scene_id: str,
    batch_id: str,
    z_ground: float | str = "auto",
    overwrite: bool = True,
    anchor_shape_eta: float = DEFAULT_ANCHOR_SHAPE_ETA,
    max_bbox_clip_px: int = 3,
    max_bbox_clip_ratio: float = 0.01,
    h_source_mode: str = "data",
    bbox_source: str = "out_bbox",
) -> dict[str, Any]:
    'overlapping SQLite\n\n train out_bbox/{frame}.txt 3D \n test out_bbox gt/gt.txt 2D outside_reference\n\n bbox_source bbox \n - "out_bbox" out_bbox 3D amodal\n - "gt_tight" (frame, vehicle_id) gt/gt.txt \n 3D out_bboxgt.txt'
    scene_name = scene_dir.name
    town = scene_to_town(scene_name)
    camera_names = discover_scene_cameras(scene_dir)
    has_bbox = any((scene_dir / cam / "out_bbox").is_dir() for cam in camera_names)
    if z_ground == "auto":
        if has_bbox:
            z_ground, z_ground_stats = estimate_road_z_from_out_bbox(
                scene_dir, camera_names
            )
        else:
            z_ground = DEFAULT_Z_GROUND
            z_ground_stats = {"source": "default_no_3d", "n": 0}
    else:
        z_ground = float(z_ground)
        z_ground_stats = {"source": "cli", "n": 0}
    print(
        f"  z_ground={z_ground:.4f} m  source={z_ground_stats.get('source')} "
        f"n={z_ground_stats.get('n', 0)}"
    )
    calibs = load_all_calibs(
        calib_root, town, camera_names=camera_names, z_ground=z_ground
    )
    camera_names = sorted(calibs.keys())

                                       

    drafts: list[ObservationDraft] = []
    homography_samples: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]] = defaultdict(list)
    stats = Counter()

    if has_bbox:
                                                                        
                                 
        ref_cam = next(
            (c for c in camera_names if (scene_dir / c / "out_bbox").is_dir()),
            None,
        )
        if ref_cam is None:
            raise FileNotFoundError(f" out_bbox : {scene_dir}")
        bbox_frames = sorted((scene_dir / ref_cam / "out_bbox").glob("*.txt"))
        if not bbox_frames:
            raise FileNotFoundError(f" out_bbox : {scene_dir / ref_cam}")

                                                       
        tight_boxes: dict[str, dict[int, dict[int, list[float]]]] = {}
        if bbox_source == "gt_tight":
            for cam_name in camera_names:
                gt_path = scene_dir / cam_name / "gt" / "gt.txt"
                if not gt_path.is_file():
                    raise FileNotFoundError(f"--bbox-source gt_tight  gt.txt: {gt_path}")
                tight_boxes[cam_name] = {
                    frame: {o["id"]: o["bbox_xyxy"] for o in objs}
                    for frame, objs in parse_gt_file(gt_path).items()
                }

        for frame_path in bbox_frames:
            frame_idx = int(frame_path.stem)   # 1-based
            ts = (frame_idx - 1) / FRAME_RATE  # 0-based timestamp

                                          
            cam_data: dict[str, dict[str, Any]] = {}
            for cam_name in camera_names:
                fp = scene_dir / cam_name / "out_bbox" / frame_path.name
                if not fp.is_file():
                    continue
                parsed = parse_bbox_file(fp)
                cam_data[cam_name] = parsed

                                     
            all_vids: set[int] = set()
            for pd in cam_data.values():
                all_vids.update(pd["vehicle_ids"])

            for vid in all_vids:
                per_cam_obs: list[tuple[
                    str, list[float], tuple[float, float], tuple[float, float] | None, int, int
                ]] = []
                for cam_name, pd in cam_data.items():
                    if vid not in pd["vehicle_ids"]:
                        continue
                    i = pd["vehicle_ids"].index(vid)
                    bbox_xyxy = pd["bboxes"][i]
                    wxy = pd["world_coords"][i]
                    cls = pd["class_ids"][i]

                    if bbox_source == "gt_tight":
                        tb = tight_boxes[cam_name].get(frame_idx, {}).get(vid)
                        if tb is None:
                            stats["skip_no_tight_box"] += 1
                            continue
                        bbox_xyxy = tb

                          
                    xyxy_arr = np.array(bbox_xyxy, dtype=np.float64)
                    if is_bbox_clipped(xyxy_arr, calibs[cam_name].frame_wh,
                                       max_bbox_clip_px, max_bbox_clip_ratio):
                        stats["skip_clipped"] += 1
                        continue

                                   
                    calib = calibs[cam_name]
                    bottom_xyz = np.array([wxy[0], wxy[1], z_ground])
                    ref_img = project_world_to_image(
                        calib.T_world_to_cam, calib.K, bottom_xyz, calib.frame_wh
                    )

                               
                    if ref_img is not None:
                        homography_samples[cam_name].append((ref_img, wxy))

                    per_cam_obs.append((cam_name, bbox_xyxy, wxy, ref_img, cls, i))

                if not per_cam_obs:
                    continue

                for cam_name, bbox_xyxy, wxy, ref_img, cls, _ in per_cam_obs:
                    class_id, class_name = SYNTH_CLASS_MAP.get(cls, DEFAULT_CLASS)
                    xyxy_arr = np.array(bbox_xyxy, dtype=np.float64)
                    drafts.append(ObservationDraft(
                        global_id=vid,
                        camera_name=cam_name,
                        frame_index=frame_idx,
                        timestamp_sec=ts,
                        image_x=0.0,
                        image_y=0.0,
                        world_x=0.0,
                        world_y=0.0,
                        outside_reference_x=wxy[0],
                        outside_reference_y=wxy[1],
                        outside_reference_image_x=ref_img[0] if ref_img else None,
                        outside_reference_image_y=ref_img[1] if ref_img else None,
                        bbox_x=xyxy_arr[0],
                        bbox_y=xyxy_arr[1],
                        bbox_w=xyxy_arr[2] - xyxy_arr[0],
                        bbox_h=xyxy_arr[3] - xyxy_arr[1],
                        confidence=1.0,
                        class_id=class_id,
                        class_name=class_name,
                    ))
                    stats["obs"] += 1
                if len(per_cam_obs) >= 2:
                    stats["mvc_candidates"] += 1

    else:
                                                                     
        for cam_name in camera_names:
            gt_path = scene_dir / cam_name / "gt" / "gt.txt"
            if not gt_path.is_file():
                continue
            gt_by_frame = parse_gt_file(gt_path)
            for frame_idx, objs in sorted(gt_by_frame.items()):
                ts = (frame_idx - 1) / FRAME_RATE
                for obj in objs:
                    xyxy = obj["bbox_xyxy"]
                    xyxy_arr = np.array(xyxy, dtype=np.float64)
                    if is_bbox_clipped(xyxy_arr, calibs[cam_name].frame_wh,
                                       max_bbox_clip_px, max_bbox_clip_ratio):
                        stats["skip_clipped"] += 1
                        continue
                    cls = obj["class_id"]
                    class_id, class_name = SYNTH_CLASS_MAP.get(cls, DEFAULT_CLASS)
                    drafts.append(ObservationDraft(
                        global_id=obj["id"],
                        camera_name=cam_name,
                        frame_index=frame_idx,
                        timestamp_sec=ts,
                        image_x=0.0,
                        image_y=0.0,
                        world_x=0.0,
                        world_y=0.0,
                        outside_reference_x=None,
                        outside_reference_y=None,
                        outside_reference_image_x=None,
                        outside_reference_image_y=None,
                        bbox_x=xyxy_arr[0],
                        bbox_y=xyxy_arr[1],
                        bbox_w=xyxy_arr[2] - xyxy_arr[0],
                        bbox_h=xyxy_arr[3] - xyxy_arr[1],
                        confidence=1.0,
                        class_id=class_id,
                        class_name=class_name,
                    ))
                    stats["obs"] += 1
                                 

    if not drafts:
        raise RuntimeError(f": {scene_dir}  stats={dict(stats)}")

                                                                           
    sample_world_xy = [
        (d.outside_reference_x, d.outside_reference_y)
        for d in drafts
        if d.outside_reference_x is not None
    ]
    if has_bbox and sample_world_xy and h_source_mode != "model":
        h_base, reproj_rms, h_inliers, h_source = fit_image_to_world_homographies(
            {cam: [(ip, wxy) for ip, wxy in pts] for cam, pts in homography_samples.items()},
            calibs,
        )
    else:
                                                     
                                                                   
                                
        h_base = {cam: c.homography for cam, c in calibs.items()}
        reproj_rms = {cam: c.reproj_rms for cam, c in calibs.items()}
        h_inliers = {cam: 0 for cam in calibs}
        h_source = {cam: "synth_ground_plane_fallback" for cam in calibs}

    h_local, origin, scale = localize_homographies(h_base, sample_world_xy or [])

                                    
    for d in drafts:
        if d.outside_reference_x is not None:
            d.outside_reference_x -= float(origin[0])
            d.outside_reference_y -= float(origin[1])

    overlap_pairs = compute_overlap_pairs(calibs, h_local, scale=scale)

                                                                       
    for d in drafts:
        calib = calibs[d.camera_name]
        xyxy = np.array([d.bbox_x, d.bbox_y, d.bbox_x + d.bbox_w, d.bbox_y + d.bbox_h], dtype=np.float64)
        ax, ay = box_anchor(
            xyxy,
            ANCHOR_MODE,
            homography=h_local[d.camera_name],
            frame_wh=calib.frame_wh,
            scale=scale,
            risk_scalar_mode="metric_jacobian",
            s0_m=DEFAULT_S0_M,
            shape_eta=anchor_shape_eta,
        )
        d.image_x, d.image_y = ax, ay
        p_geo = image_point_to_world(h_local[d.camera_name], (ax, ay), scale)
        d.world_x, d.world_y = float(p_geo[0]), float(p_geo[1])

                                                                           
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = init_trajectory_db(str(db_path), batch_id)
    obs_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(trajectory_observations)").fetchall()
    }
    required_ref_img = {"outside_reference_image_x", "outside_reference_image_y"}
    if not required_ref_img.issubset(obs_cols):
        conn.close()
        raise RuntimeError(
            f"{db_path}  trajectory_observations  {sorted(required_ref_img)}"
            'DB import'
        )

    if overwrite:
        conn.execute("DELETE FROM trajectory_observations WHERE scene_id=? AND batch_id=?", (scene_id, batch_id))
        conn.execute("DELETE FROM trajectory_global_tracks WHERE scene_id=? AND batch_id=?", (scene_id, batch_id))
        conn.execute("DELETE FROM trajectory_merges WHERE scene_id=? AND batch_id=?", (scene_id, batch_id))
        conn.execute("DELETE FROM trajectory_runtime_stats WHERE scene_id=? AND batch_id=?", (scene_id, batch_id))
        conn.execute(
            "DELETE FROM trajectory_camera_config WHERE scene_id=? AND batch_id=?",
            (scene_id, batch_id),
        )
        conn.execute("DELETE FROM camera_calibrations WHERE scene_id=?", (scene_id,))
        conn.execute("DELETE FROM scene_canvas WHERE scene_id=?", (scene_id,))
        conn.execute("DELETE FROM scene_overlap_pairs WHERE scene_id=?", (scene_id,))

    for cam_name, calib in calibs.items():
        save_calibration(
            conn,
            cam_name,
            scene_id,
            scale,
            h_local[cam_name],
            image_pts=np.zeros((0, 2), dtype=np.float32),
            world_pts=np.zeros((0, 2), dtype=np.float32),
            source=f"{h_source.get(cam_name, 'synth')}:{scene_name}",
            inlier_count=h_inliers.get(cam_name, 0),
            reproj_rms=reproj_rms.get(cam_name, calib.reproj_rms),
            commit=False,
        )

    now = utc_now()
    insert_sql = """
        INSERT INTO trajectory_observations
            (scene_id, batch_id, camera_name, local_track_id, global_id, frame_index,
             timestamp_sec, image_x, image_y, world_x, world_y,
             outside_reference_x, outside_reference_y,
             outside_reference_image_x, outside_reference_image_y,
             bbox_x, bbox_y, bbox_w, bbox_h, confidence,
             class_id, class_name, recorded_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
                                                                  
    local_track_counter: dict[str, int] = defaultdict(int)
    local_track_map: dict[tuple[str, int], int] = {}
    global_ids: set[int] = set()
    world_pts: list[tuple[float, float]] = []

    for d in drafts:
        key = (d.camera_name, d.global_id)
        if key not in local_track_map:
            local_track_counter[d.camera_name] += 1
            local_track_map[key] = local_track_counter[d.camera_name]
        global_ids.add(d.global_id)
        world_pts.append((d.world_x, d.world_y))
        conn.execute(insert_sql, (
            scene_id, batch_id,
            d.camera_name, local_track_map[key], d.global_id,
            d.frame_index, d.timestamp_sec,
            d.image_x, d.image_y, d.world_x, d.world_y,
            d.outside_reference_x, d.outside_reference_y,
            d.outside_reference_image_x, d.outside_reference_image_y,
            d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h,
            d.confidence, d.class_id, d.class_name, now,
        ))

    for gid in sorted(global_ids):
        conn.execute(
            """INSERT OR IGNORE INTO trajectory_global_tracks
               (scene_id, batch_id, global_id, created_at, source)
               VALUES (?,?,?,?,?)""",
            (scene_id, batch_id, gid, now, f"synth:{scene_name}"),
        )

                  
    if world_pts:
        wpts_arr = np.array(world_pts, dtype=np.float64)
        margin_m = 5.0
        min_xy = wpts_arr.min(axis=0) - margin_m
        max_xy = wpts_arr.max(axis=0) + margin_m
        cw = int(np.ceil((max_xy[0] - min_xy[0]) * scale)) + 1
        ch = int(np.ceil((max_xy[1] - min_xy[1]) * scale)) + 1
        conn.execute(
            """INSERT OR REPLACE INTO scene_canvas
               (scene_id, scale, min_x, min_y, canvas_w, canvas_h, saved_at)
               VALUES (?,?,?,?,?,?,?)""",
            (scene_id, scale, float(min_xy[0]), float(min_xy[1]), cw, ch, now),
        )

    frame_wh_by_camera = {cam: (FRAME_W, FRAME_H) for cam in camera_names}
    save_scene_overlap_pairs(conn, scene_id, overlap_pairs, replace=True)
    save_batch_camera_frame_wh(conn, scene_id, batch_id, frame_wh_by_camera)
    meta: dict[str, Any] = {
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "scene_name": scene_name,
        "town": town,
        "split": "train" if has_bbox else "test",
        "has_3d_annotations": has_bbox,
        "observation_count": len(drafts),
        "global_id_count": len(global_ids),
        "mvc_candidates": int(stats["mvc_candidates"]),
        "overlap_pairs": [list(p) for p in sorted(overlap_pairs)],
        "origin_carla_world": origin.tolist(),
        "scale": scale,
        "z_ground": z_ground,
        "z_ground_stats": z_ground_stats,
        "frame_rate": FRAME_RATE,
        "frame_wh_by_camera": {cam: [FRAME_W, FRAME_H] for cam in camera_names},
        "coordinate_frame": "carla_world_minus_origin",
        "world_source": "p_geo",
        "bbox_source": bbox_source if has_bbox else "gt_2d_only",
        "reference_source": "gt_3d_bottom_center" if has_bbox else "none",
        "reference_image_source": "pinhole_bottom_center" if has_bbox else "none",
        "reproj_rms": reproj_rms,
        "homography_source": h_source,
        "homography_inliers": h_inliers,
        "import_stats": dict(stats),
    }
    save_import_audit_metadata(conn, scene_id, batch_id, meta)
    conn.commit()
    conn.close()
    return meta


                                                                             


def validate_import(
    db_path: Path,
    scene_id: str,
    batch_id: str,
    *,
    max_time_gap: float = 0.35,
    bad_dist_m: float = 80.0,
) -> dict[str, Any]:
    if not db_path.is_file() or db_path.stat().st_size == 0:
        raise FileNotFoundError(f": {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
            (scene_id, batch_id),
        ).fetchone()
        if row is None or int(row[0]) == 0:
            raise RuntimeError(f"scene_id={scene_id} batch_id={batch_id}  import")
    except Exception:
        conn.close()
        raise

    obs_count = conn.execute(
        "SELECT COUNT(*) FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
        (scene_id, batch_id),
    ).fetchone()[0]
    cameras = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT camera_name FROM trajectory_observations WHERE scene_id=? AND batch_id=? ORDER BY camera_name",
            (scene_id, batch_id),
        )
    ]
    gid_count = conn.execute(
        "SELECT COUNT(DISTINCT global_id) FROM trajectory_observations WHERE scene_id=? AND batch_id=?",
        (scene_id, batch_id),
    ).fetchone()[0]
    multi_gid = conn.execute(
        """SELECT COUNT(*) FROM (
               SELECT global_id FROM trajectory_observations WHERE scene_id=? AND batch_id=?
               GROUP BY global_id HAVING COUNT(DISTINCT camera_name) >= 2
           )""",
        (scene_id, batch_id),
    ).fetchone()[0]

    overlap_pairs = load_scene_overlap_pairs(conn, scene_id, cameras)
    warnings: list[str] = []
    if not overlap_pairs and len(cameras) >= 2:
        overlap_pairs = {tuple(sorted((cameras[i], cameras[j])))
                         for i in range(len(cameras)) for j in range(i + 1, len(cameras))}
        warnings.append('DB scene_overlap_pairs')

    mvc_count = 0
    spatial_count = 0
    dists: list[float] = []
    if GeometryDataView is None:
        warnings.append('GeometryDataView MVC')
    else:
        view = GeometryDataView()
        calibrations = view.load_camera_calibrations(conn, scene_id, cameras)
        db_wh = load_batch_camera_frame_wh(conn, scene_id, batch_id, cameras)
        wh_map = {c: db_wh.get(c) or (FRAME_W, FRAME_H) for c in cameras}
        observations = list(
            view.iter_observations_from_db(
                conn, scene_id, batch_id, cameras, calibrations, frame_wh_by_camera=wh_map,
            )
        )
        mvc_pairs = view.mine_mvc_pairs(observations, overlap_pairs, max_time_gap_sec=max_time_gap)
        mvc_count = len(mvc_pairs)
        for pair in mvc_pairs:
            ax, ay = pair.obs_a.world_geo
            bx, by = pair.obs_b.world_geo
            d = float(np.hypot(ax - bx, ay - by))
            dists.append(d)
            if d <= bad_dist_m:
                spatial_count += 1
        if mvc_count == 0:
            warnings.append('MVC 0')
        elif spatial_count < 64:
            warnings.append(f" MVC (≤{bad_dist_m}m)  {spatial_count} ")

    conn.close()
    return {
        "scene_id": scene_id,
        "observation_count": int(obs_count),
        "camera_count": len(cameras),
        "global_id_count": int(gid_count),
        "multi_camera_global_ids": int(multi_gid),
        "overlap_pairs": sorted(overlap_pairs),
        "mvc_pair_count": mvc_count,
        "spatial_mvc_pair_count": spatial_count,
        "mvc_median_dist_m": float(np.median(dists)) if dists else None,
        "warnings": warnings,
    }


def print_validation_report(report: dict[str, Any]) -> None:
    print(f"\n=== Synthehicle : {report['scene_id']} ===")
    print(f"          : {report['observation_count']}")
    print(f"          : {report['camera_count']}")
    print(f"   ID     : {report['global_id_count']}")
    print(f"   ID   : {report['multi_camera_global_ids']}")
    print(f"      : {report['overlap_pairs']}")
    print(f"  MVC     : {report['mvc_pair_count']}")
    print(f"  MVC (≤80m)    : {report['spatial_mvc_pair_count']}")
    if report.get("mvc_median_dist_m") is not None:
        print(f"  MVC median d  : {report['mvc_median_dist_m']:.3f} m")
    for w in report.get("warnings", []):
        print(f"  [WARN] {w}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def cmd_import(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    calib_root = data_root / "calibration"

                
    if args.all_train or args.all_test:
        split = "test" if args.all_test else "train"
        scenes = discover_overlapping_scenes(data_root, split)
    elif args.scene:
                                
        found = None
        for split in ("train", "test"):
            p = data_root / split / args.scene
            if p.is_dir() and OVERLAP_MARKER in args.scene:
                found = p
                break
        if found is None:
            raise FileNotFoundError(f" {args.scene} '-O-'")
        scenes = [found]
    else:
        raise ValueError('--scene SCENE_NAME --all-train / --all-test')

    db_path = Path(args.db)
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []

    for scene_dir in scenes:
        scene_name = scene_dir.name
        sid = args.scene_id or scene_to_id(scene_name)
        print(f"\n>>>  {scene_name} -> scene_id={sid}")
        try:
            meta = import_scene(
                scene_dir,
                calib_root,
                db_path=db_path,
                scene_id=sid,
                batch_id=args.batch,
                z_ground=args.z_ground,
                overwrite=not args.no_overwrite,
                anchor_shape_eta=args.anchor_shape_eta,
                h_source_mode=args.h_source,
                bbox_source=args.bbox_source,
            )
            results.append(meta)
            print(
                f"  obs={meta['observation_count']}  global_id={meta['global_id_count']}  "
                f"mvc_candidates={meta['mvc_candidates']}  overlap={meta['overlap_pairs']}"
            )
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            if args.fail_fast:
                raise

    elapsed = time.perf_counter() - t0
    print(f"\n ({elapsed:.1f}s)  DB={db_path}")
    for meta in results:
        print(f"  {meta['scene_id']}: obs={meta['observation_count']} mvc={meta['mvc_candidates']}")
        sid = meta['scene_id']
        print(f"    validate: py experiments/synthehicle/import_synthehicle_mvc.py validate "
              f"--db {db_path} --scene {sid} --batch {args.batch}")


def cmd_validate(args: argparse.Namespace) -> None:
    report = validate_import(
        Path(args.db), args.scene, args.batch,
        max_time_gap=args.max_time_gap,
        bad_dist_m=args.bad_dist_m,
    )
    print_validation_report(report)
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  JSON  -> {args.json}")


def cmd_inspect(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    if not db_path.is_file():
        raise FileNotFoundError(f"DB : {db_path}")
    conn = sqlite3.connect(str(db_path))
    print(f"=== inspect (Synthehicle) scene={args.scene} batch={args.batch} ===")
    for r in conn.execute(
        """SELECT camera_name, COUNT(*), MIN(timestamp_sec), MAX(timestamp_sec),
                  MIN(world_x), MAX(world_x), MIN(world_y), MAX(world_y), AVG(confidence)
             FROM trajectory_observations
            WHERE scene_id=? AND batch_id=?
            GROUP BY camera_name ORDER BY camera_name""",
        (args.scene, args.batch),
    ):
        print(
            f"  {r[0]}: n={r[1]} t=[{r[2]:.1f},{r[3]:.1f}] "
            f"wx=[{r[4]:.1f},{r[5]:.1f}] wy=[{r[6]:.1f},{r[7]:.1f}] conf={r[8]:.3f}"
        )
    pairs = load_scene_overlap_pairs(conn, args.scene)
    audit = load_import_audit_metadata(conn, args.scene, args.batch)
    print(f"  overlap_pairs : {sorted(pairs)}")
    if audit:
        print(f"  mvc_candidates: {audit.get('mvc_candidates')}")
        print(f"  import_stats  : {audit.get('import_stats')}")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Synthehicle overlapping → SQLiteMVC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── import ──────────────────────────────────────────────────────────────
    p_import = sub.add_parser("import", help='overlapping')
    p_import.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                          help='Synthehicle')
    grp = p_import.add_mutually_exclusive_group()
    grp.add_argument("--scene", default=None,
                     help='Town01-O-dawn train/ test/')
    grp.add_argument("--all-train", action="store_true", help='train overlapping')
    grp.add_argument("--all-test", action="store_true", help='test overlapping')
    p_import.add_argument("--db", required=True, help='SQLite')
    p_import.add_argument("--batch", required=True, help='batch_id pipeline')
    p_import.add_argument("--scene-id", default=None,
                          help='DB scene_id synth_{scene_name}--all')
    p_import.add_argument("--z-ground", type=parse_z_ground, default="auto",
                          help='Zauto=train 3D z CARLA')
    p_import.add_argument("--no-overwrite", action="store_true",
                          help='scene_id+batch_id')
    p_import.add_argument("--anchor-shape-eta", type=float, default=DEFAULT_ANCHOR_SHAPE_ETA,
                          dest="anchor_shape_eta", metavar="ETA",
                          help='0=1=slimness')
    p_import.add_argument("--fail-fast", action="store_true",
                          help='See the HSG-SRL reproduction notes.')
    p_import.add_argument("--h-source", choices=("data", "model"), default="data",
                          help='image→world H data=GT'
                               'model=')
    p_import.add_argument("--bbox-source", choices=("out_bbox", "gt_tight"), default="out_bbox",
                          help='train 2D out_bbox=3D amodal'
                               'gt_tight=gt.txt frame+vehicle_id'
                               '3D out_bbox amodal')
    p_import.set_defaults(func=cmd_import)

    # ── validate ─────────────────────────────────────────────────────────────
    p_val = sub.add_parser("validate", help='See the HSG-SRL reproduction notes.')
    p_val.add_argument("--db", required=True)
    p_val.add_argument("--scene", required=True)
    p_val.add_argument("--batch", required=True)
    p_val.add_argument("--max-time-gap", type=float, default=0.35)
    p_val.add_argument("--bad-dist-m", type=float, default=80.0)
    p_val.add_argument("--json", default=None, help='JSON')
    p_val.set_defaults(func=cmd_validate)

    # ── inspect ──────────────────────────────────────────────────────────────
    p_ins = sub.add_parser("inspect", help='See the HSG-SRL reproduction notes.')
    p_ins.add_argument("--db", required=True)
    p_ins.add_argument("--scene", required=True)
    p_ins.add_argument("--batch", required=True)
    p_ins.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
