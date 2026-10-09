#!/usr/bin/env python3
"""Build BEVHeight pkl infos from the packed KITTI folders. Official schema."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from pyquaternion import Quaternion

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "tools"))
from official_hparams import DEFAULT_HW  # noqa: E402

try:
    import mmcv
except ImportError:
    mmcv = None


NAME2NUSC = {"car": "vehicle.car", "van": "vehicle.car", "truck": "vehicle.truck",
             "bus": "vehicle.bus.rigid"}

# KITTI Easy-style. Official Rope3D train keeps these; use --filter-kitti-easy
# on roadside packs so clipped / tiny 2D boxes are not full supervision.
MIN_BOX_H_PX = 40.0
EDGE_PX = 2.0


def _drop_label(box2d, W: int, H: int) -> str | None:
    xmin, ymin, xmax, ymax = box2d
    if (ymax - ymin) < MIN_BOX_H_PX:
        return "tiny"
    if xmin <= EDGE_PX or ymin <= EDGE_PX or xmax >= W - EDGE_PX or ymax >= H - EDGE_PX:
        return "edge"
    return None


def load_calib(path: Path) -> np.ndarray:
    for row in path.read_text(encoding="utf-8").splitlines():
        if row.startswith("P2:"):
            vals = [float(x) for x in row.split()[1:]]
            return np.array(vals, dtype=np.float32).reshape(3, 4)[:3, :3]
    raise ValueError(path)


def load_denorm(path: Path) -> np.ndarray:
    return np.array([float(x) for x in path.read_text().split()], dtype=np.float64)


def get_cam2lidar(denorm):
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    Rz = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    origin = np.array([0.0, 1.0, 0.0])
    target = -np.array(denorm[:3])
    target = target / (np.linalg.norm(target) + 1e-12)
    sita = math.acos(float(np.clip(np.inner(target, origin), -1.0, 1.0)))
    n_vec = np.cross(target, origin)
    n_n = np.linalg.norm(n_vec)
    if n_n < 1e-8:
        rot = np.eye(3, dtype=np.float32)
    else:
        rot, _ = cv2.Rodrigues((n_vec / n_n).astype(np.float32) * sita)
    cam2lidar = (Rz @ Rx @ rot).astype(np.float32)
    d = abs(float(denorm[3])) / (np.linalg.norm(denorm[:3]) + 1e-12)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = cam2lidar
    T[:3, 3] = [0.0, 0.0, d]
    return cam2lidar, [0.0, 0.0, float(d)], T, denorm


def get_annos(label_path: Path, Tr_cam2lidar: np.ndarray) -> list:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    fields = ["type", "truncated", "occluded", "alpha", "xmin", "ymin", "xmax",
              "ymax", "dh", "dw", "dl", "lx", "ly", "lz", "ry"]
    annos = []
    with label_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=" ", fieldnames=fields)
        for row in reader:
            if row["type"] not in NAME2NUSC:
                continue
            dim = [float(row["dl"]), float(row["dw"]), float(row["dh"])]
            if sum(dim) == 0:
                continue
            loc_cam = np.array([float(row["lx"]), float(row["ly"]),
                                float(row["lz"]), 1.0]).reshape(4, 1)
            loc_lidar = (Tr_cam2lidar @ loc_cam).squeeze(-1)[:3]
            loc_lidar[2] += 0.5 * float(row["dh"])
            ry = float(row["ry"])
            rotation = 0.5 * math.pi - ry
            annos.append({
                "dim": dim,
                "loc": loc_lidar.tolist(),
                "rotation": rotation,
                "name": NAME2NUSC[row["type"]],
                "box2d": [float(row["xmin"]), float(row["ymin"]),
                          float(row["xmax"]), float(row["ymax"])],
                "truncated_state": int(float(row["truncated"])),
                "occluded_state": int(float(row["occluded"])),
            })
    return annos


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mmcv is not None:
        mmcv.dump(obj, str(path))
    else:
        import pickle
        with path.open("wb") as f:
            pickle.dump(obj, f)


def build_split(site_root: Path, split: str, filter_kitti_easy: bool):
    idx_list = [x.strip() for x in
                (site_root / "ImageSets" / f"{split}.txt").read_text().splitlines()
                if x.strip()]
    fmap = json.loads((site_root / "frame_map.json").read_text(encoding="utf-8"))
    infos = []
    dropped = {"tiny": 0, "edge": 0, "kept": 0}
    for index in idx_list:
        rec = fmap.get(index, {})
        img_rel = f"images/{index}.jpg"
        if not (site_root / img_rel).is_file():
            continue
        H, W = DEFAULT_HW
        if rec.get("image_wh"):
            W, H = int(rec["image_wh"][0]), int(rec["image_wh"][1])
        K = load_calib(site_root / "calib" / f"{index}.txt")
        denorm = load_denorm(site_root / "denorm" / f"{index}.txt")
        cam2lidar, translation, Tr, den = get_cam2lidar(denorm)
        info = {
            "sample_token": index,
            "timestamp": 1000000,
            "scene_token": index,
            "cam_infos": {
                "CAM_FRONT": {
                    "sample_token": index,
                    "timestamp": 1000000,
                    "is_key_frame": True,
                    "height": H,
                    "width": W,
                    "filename": img_rel,
                    "ego_pose": {
                        "translation": [0.0, 0.0, 0.0],
                        "rotation": [1.0, 0.0, 0.0, 0.0],
                        "token": index,
                        "timestamp": 1000000,
                    },
                    "calibrated_sensor": {
                        "token": index,
                        "sensor_token": index,
                        "translation": translation,
                        "rotation_matrix": cam2lidar.tolist(),
                        "camera_intrinsic": K.tolist(),
                    },
                    "denorm": den.tolist(),
                }
            },
            "lidar_infos": {"LIDAR_TOP": {}},
            "sweeps": [],
        }
        ann_infos = []
        for anno in get_annos(site_root / "label_2" / f"{index}.txt", Tr):
            if filter_kitti_easy and split != "test":
                why = _drop_label(anno["box2d"], W, H)
                if why:
                    dropped[why] += 1
                    continue
            dropped["kept"] += 1
            yaw = anno["rotation"]
            rot_mat = np.array([
                [math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0],
                [0, 0, 1],
            ])
            ann_infos.append({
                "category_name": anno["name"],
                "translation": anno["loc"],
                "rotation": Quaternion(matrix=rot_mat),
                "size": anno["dim"],
                "prev": "",
                "next": "",
                "sample_token": index,
                "instance_token": index,
                "token": index,
                "visibility_token": str(anno["occluded_state"]),
                "num_lidar_pts": 3,
                "num_radar_pts": 0,
                "velocity": np.zeros(3),
            })
        info["ann_infos"] = ann_infos
        infos.append(info)
    return infos, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--data-root", default=str(HERE / "data"))
    ap.add_argument(
        "--filter-kitti-easy", action="store_true",
        help="Drop train/val boxes that touch the image border or have height < 40 px",
    )
    args = ap.parse_args()
    site_root = Path(args.data_root) / args.site
    if not (site_root / "frame_map.json").is_file():
        raise SystemExit(f"missing pack: {site_root}")
    for split, name in (("train", "infos_train.pkl"),
                        ("val", "infos_val.pkl"),
                        ("test", "infos_test.pkl")):
        infos, dropped = build_split(site_root, split, args.filter_kitti_easy)
        dump(infos, site_root / name)
        extra = ""
        if args.filter_kitti_easy and split != "test":
            extra = (f"  kept={dropped['kept']} drop_edge={dropped['edge']} "
                     f"drop_tiny={dropped['tiny']}")
        print(f"{args.site} {split}: {len(infos)} -> {name}{extra}")


if __name__ == "__main__":
    main()
