#!/usr/bin/env python3
"""Build a KITTI-style BEVHeight pack from frozen splits. Writes only step2_pack/work."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ARTIFACT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ARTIFACT / "src"))
sys.path.insert(0, str(ARTIFACT / "src" / "importers"))

from layout import (  # noqa: E402
    ALL_SITES,
    CARLA_TO_KITTI,
    PACK_OUT,
    PACK_WORK,
    SITE_SPECS,
    assert_write,
    label_time,
    load_split_frames,
    load_time_maps,
    parse_sites,
    require_freeze,
    video_path,
)

import import_lumpi as lumpi  # noqa: E402
import import_synthehicle as synth  # noqa: E402

X_BOUND = (0.0, 140.8)
Y_BOUND = (-70.4, 70.4)
JPEG_QUALITY = 95


def _kitti_P2(K: np.ndarray) -> str:
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    return f"P2: {fx:.6f} 0 {cx:.6f} 0 0 {fy:.6f} {cy:.6f} 0 0 0 1 0\n"


def _plane_from_points(pts: np.ndarray) -> np.ndarray:
    p0, p1, p2 = pts
    n = np.cross(p1 - p0, p2 - p0)
    n = n / (np.linalg.norm(n) + 1e-12)
    d = -float(n @ p0)
    return np.array([n[0], n[1], n[2], d], dtype=np.float64)


def _denorm_from_w2c(T_w2c: np.ndarray, z_ground: float) -> np.ndarray:
    world = np.array([
        [0.0, 0.0, z_ground, 1.0],
        [1.0, 0.0, z_ground, 1.0],
        [0.0, 1.0, z_ground, 1.0],
    ], dtype=np.float64)
    cam = (T_w2c @ world.T).T[:, :3]
    den = _plane_from_points(cam)
    if den[1] > 0:
        den = -den
    return den


def get_cam2lidar(denorm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Official BEVHeight get_cam2lidar (scripts/gen_info_rope3d.py)."""
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    Rz = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    origin = np.array([0.0, 1.0, 0.0])
    target = -np.array(denorm[:3], dtype=np.float64)
    target = target / (np.linalg.norm(target) + 1e-12)
    sita = math.acos(float(np.clip(np.inner(target, origin), -1.0, 1.0)))
    n_vec = np.cross(target, origin)
    n_n = np.linalg.norm(n_vec)
    if n_n < 1e-8:
        rot = np.eye(3)
    else:
        rot, _ = cv2.Rodrigues((n_vec / n_n) * sita)
    cam2lidar = Rz @ Rx @ rot
    d = abs(float(denorm[3])) / (np.linalg.norm(denorm[:3]) + 1e-12)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = cam2lidar
    T[:3, 3] = [0.0, 0.0, d]
    return T, np.array([0.0, 0.0, d], dtype=np.float64)


def in_official_bev(xyz_lidar: np.ndarray) -> bool:
    return X_BOUND[0] <= xyz_lidar[0] <= X_BOUND[1] and Y_BOUND[0] <= xyz_lidar[1] <= Y_BOUND[1]


def token_of(cam: str, frame_index: int) -> str:
    return f"{cam}_{frame_index:06d}"


def write_kitti_label(path: Path, boxes: list[dict]) -> None:
    lines = []
    for b in boxes:
        trunc = int(b.get("truncated", 0))
        lines.append(
            f"{b['type']} {trunc} 0 {b['alpha']:.6f} "
            f"{b['xmin']:.2f} {b['ymin']:.2f} {b['xmax']:.2f} {b['ymax']:.2f} "
            f"{b['h']:.4f} {b['w']:.4f} {b['l']:.4f} "
            f"{b['lx']:.4f} {b['ly']:.4f} {b['lz']:.4f} {b['ry']:.6f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def kitti_box(loc_cam: np.ndarray, dims: tuple[float, float, float],
              ry: float, box2d: tuple[float, float, float, float]) -> dict | None:
    if loc_cam[2] < 0.5:
        return None
    h, w, l = dims
    xmin, ymin, xmax, ymax = box2d
    alpha = ry - math.atan2(float(loc_cam[0]), float(loc_cam[2]))
    return {
        "type": "car",
        "alpha": alpha,
        "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
        "h": h, "w": w, "l": l,
        "lx": float(loc_cam[0]), "ly": float(loc_cam[1]), "lz": float(loc_cam[2]),
        "ry": ry,
        "truncated": 0,
    }


class FrameReader:
    """Keep one VideoCapture per file; sequential frames just read()."""

    def __init__(self) -> None:
        self._caps: dict[str, cv2.VideoCapture] = {}
        self._next: dict[str, int] = {}

    def get(self, video: Path, timestamp: float, frame_index: int) -> np.ndarray | None:
        key = str(video)
        cap = self._caps.get(key)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(key)
            self._caps[key] = cap
            self._next[key] = -1
        if not cap.isOpened():
            return None
        want = max(int(frame_index) - 1, 0)
        nxt = self._next.get(key, -1)
        if want != nxt:
            cap.set(cv2.CAP_PROP_POS_FRAMES, want)
        ok, img = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp * 1000.0))
            ok, img = cap.read()
            self._next[key] = -1
            return img if ok else None
        self._next[key] = want + 1
        return img

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()


def load_lumpi_calibs(spec):
    data_root = ARTIFACT / "data" / "external" / "LUMPI"
    meta = lumpi.load_meta(data_root)
    out = {}
    for cam in spec.cameras:
        out[cam] = lumpi.load_camera_calib(
            meta, spec.measurement, int(cam[1:]), data_root, z_ground=spec.z_ground)
    return out


def lumpi_boxes_at(index, calib, t_label: float, frame_wh: tuple[int, int]) -> list[dict]:
    obj = index.at_time(t_label)
    if obj is None:
        return []
    boxes = []
    for o in obj:
        bb = lumpi.compute_pixel_bbox_from_3d(
            calib.rvec, calib.tvec, calib.K, calib.dist, calib.T_world_to_cam,
            o["cx"], o["cy"], o["cz"], o["length"], o["width"], o["height"],
            o["heading"], frame_wh)
        if bb is None:
            continue
        x, y, w, h = bb
        bottom = np.array([o["cx"], o["cy"], o["cz"] - o["height"] / 2.0, 1.0])
        loc = calib.T_world_to_cam @ bottom
        direction = np.array([math.cos(o["heading"]), math.sin(o["heading"]), 0.0])
        dcam = calib.T_world_to_cam[:3, :3] @ direction
        ry = math.atan2(float(dcam[0]), float(dcam[2]))
        rec = kitti_box(loc[:3], (o["height"], o["width"], o["length"]), ry,
                        (x, y, x + w, y + h))
        if rec:
            rec["truncated"] = int(
                rec["xmin"] <= 2 or rec["ymin"] <= 2
                or rec["xmax"] >= frame_wh[0] - 2 or rec["ymax"] >= frame_wh[1] - 2)
            rec["world_xy"] = (float(o["cx"]), float(o["cy"]))
            boxes.append(rec)
    return boxes


class LumpiTimeIndex:
    def __init__(self, labels_by_t: dict) -> None:
        items = sorted(labels_by_t.items(), key=lambda kv: kv[0])
        self.times = np.array([t for t, _ in items], dtype=float)
        self.objs = [o for _, o in items]

    def at_time(self, t: float, max_gap: float = 0.15) -> list[dict] | None:
        if len(self.times) == 0:
            return None
        i = int(np.searchsorted(self.times, t))
        cands = []
        if i < len(self.times):
            cands.append(i)
        if i > 0:
            cands.append(i - 1)
        j = min(cands, key=lambda k: abs(self.times[k] - t))
        if abs(self.times[j] - t) > max_gap:
            return None
        return self.objs[j]


def synth_kitti_T(T_w2c_carla: np.ndarray) -> np.ndarray:
    C = np.asarray(CARLA_TO_KITTI, dtype=np.float64)
    return C @ T_w2c_carla


def synth_boxes_at(bbox_path: Path, T_w2c_kitti: np.ndarray, K: np.ndarray,
                   frame_wh: tuple[int, int]) -> list[dict]:
    if not bbox_path.is_file():
        return []
    data = json.loads(bbox_path.read_text(encoding="utf-8"))
    boxes = []
    n = len(data.get("vehicle_id", []))
    fw, fh = frame_wh
    for i in range(n):
        wc = np.asarray(data["world_coords"][i], dtype=np.float64)
        # wc: [[x0..x7],[y0..y7],[z0..z7],[1..1]]
        xyz = np.stack([wc[0], wc[1], wc[2]], axis=1)
        bottom = xyz[:4]
        center = bottom.mean(axis=0)
        loc = T_w2c_kitti @ np.append(center, 1.0)
        dx = xyz[0] - xyz[3]
        heading = math.atan2(float(dx[1]), float(dx[0]))
        direction = np.array([math.cos(heading), math.sin(heading), 0.0])
        dcam = T_w2c_kitti[:3, :3] @ direction
        ry = math.atan2(float(dcam[0]), float(dcam[2]))
        l = float(np.linalg.norm(xyz[0, :2] - xyz[3, :2]))
        w = float(np.linalg.norm(xyz[0, :2] - xyz[1, :2]))
        h = float(abs(xyz[4, 2] - xyz[0, 2])) if len(xyz) >= 5 else 1.5
        bb = data["bboxes"][i]
        xmin, ymin = float(bb[0][0]), float(bb[0][1])
        xmax, ymax = float(bb[1][0]), float(bb[1][1])
        xmin, xmax = max(0.0, xmin), min(float(fw), xmax)
        ymin, ymax = max(0.0, ymin), min(float(fh), ymax)
        rec = kitti_box(loc[:3], (h, w, l), ry, (xmin, ymin, xmax, ymax))
        if rec:
            rec["truncated"] = int(
                rec["xmin"] <= 2 or rec["ymin"] <= 2
                or rec["xmax"] >= fw - 2 or rec["ymax"] >= fh - 2)
            rec["world_xy"] = (float(center[0]), float(center[1]))
            boxes.append(rec)
    return boxes


def pack_site(site: str, max_frames_per_cam: int | None) -> dict:
    require_freeze()
    spec = SITE_SPECS[site]
    work = assert_write(PACK_WORK / site, (PACK_WORK,))
    out_dir = assert_write(PACK_OUT / site, (PACK_OUT,))
    for sub in ("images", "calib", "denorm", "label_2", "ImageSets"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_split_frames(site)
    tm = load_time_maps(site)
    frame_map = {}
    audit = {"site": site, "in_range": 0, "out_range": 0, "no_box": 0, "by_cam": {}}
    split_lists = {k: [] for k in ("train", "val", "test")}

    if spec.kind == "lumpi":
        calibs = load_lumpi_calibs(spec)
        data_root = ARTIFACT / "data" / "external" / "LUMPI"
        labels = lumpi.parse_label_csv(
            lumpi.label_path_for_measurement(data_root, spec.measurement))
        lidx = LumpiTimeIndex(labels)
        T_kitti = {cam: calibs[cam].T_world_to_cam for cam in spec.cameras}
        Ks = {cam: calibs[cam].K for cam in spec.cameras}
        whs = {cam: tuple(calibs[cam].frame_wh) for cam in spec.cameras}
    else:
        calib_dir = ARTIFACT / spec.calib_rel
        calibs = {}
        T_kitti = {}
        Ks = {}
        whs = {}
        for cam in spec.cameras:
            raw = synth.load_camera_calib(calib_dir, int(cam[1:]), z_ground=spec.z_ground)
            calibs[cam] = raw
            T_kitti[cam] = synth_kitti_T(raw.T_world_to_cam)
            Ks[cam] = raw.K
            whs[cam] = spec.image_wh or (1920, 1080)

    per_cam_count: dict[str, int] = defaultdict(int)
    items = []
    for frames in splits.values():
        items.extend(frames.values())
    items.sort(key=lambda r: (r["camera_id"], r["frame_index"]))
    reader = FrameReader()
    try:
        for rec in items:
            cam = rec["camera_id"]
            split = rec["split"]
            if cam not in spec.cameras:
                continue
            if max_frames_per_cam is not None and per_cam_count[f"{split}:{cam}"] >= max_frames_per_cam:
                continue
            token = token_of(cam, rec["frame_index"])
            video = video_path(spec, cam)
            img = reader.get(video, rec["timestamp"], rec["frame_index"])
            if img is None:
                continue
            wh = (int(img.shape[1]), int(img.shape[0]))
            jpg = work / "images" / f"{token}.jpg"
            cv2.imwrite(str(jpg), img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            K = Ks[cam]
            (work / "calib" / f"{token}.txt").write_text(_kitti_P2(K), encoding="utf-8")
            den = _denorm_from_w2c(T_kitti[cam], spec.z_ground)
            (work / "denorm" / f"{token}.txt").write_text(
                " ".join(f"{x:.8f}" for x in den) + "\n", encoding="utf-8")
            T_c2l, t_c2l = get_cam2lidar(den)
            T_l2c = np.linalg.inv(T_c2l)
            T_c2w = np.linalg.inv(T_kitti[cam])
            T_l2w = T_c2w @ T_l2c

            boxes = []
            if spec.kind == "lumpi":
                t_lab = label_time(rec["timestamp"], tm.get(cam, (0.0, 0.0)))
                boxes = lumpi_boxes_at(lidx, calibs[cam], t_lab, wh)
            else:
                bbox_path = ARTIFACT / spec.data_rel / cam / "out_bbox" / f"{rec['frame_index']:06d}.txt"
                boxes = synth_boxes_at(bbox_path, T_kitti[cam], K, wh)

            if split != "test":
                write_kitti_label(work / "label_2" / f"{token}.txt", boxes)
            else:
                write_kitti_label(work / "label_2" / f"{token}.txt", [])

            n_in = 0
            for b in boxes:
                loc = np.array([b["lx"], b["ly"], b["lz"], 1.0])
                lid = T_c2l @ loc
                if in_official_bev(lid[:3]):
                    n_in += 1
                    audit["in_range"] += 1
                else:
                    audit["out_range"] += 1
            if not boxes:
                audit["no_box"] += 1
            audit["by_cam"].setdefault(cam, {"in_range": 0, "out_range": 0, "frames": 0})
            audit["by_cam"][cam]["in_range"] += n_in
            audit["by_cam"][cam]["out_range"] += max(0, len(boxes) - n_in)
            audit["by_cam"][cam]["frames"] += 1

            frame_map[token] = {
                "site": site,
                "camera_id": cam,
                "frame_index": rec["frame_index"],
                "timestamp": rec["timestamp"],
                "split": split,
                "image_wh": list(wh),
                "K": K.tolist(),
                "T_world_to_kitti_cam": T_kitti[cam].tolist(),
                "T_kitti_cam_to_world": T_c2w.tolist(),
                "T_cam2lidar": T_c2l.tolist(),
                "T_lidar_to_world": T_l2w.tolist(),
                "denorm": den.tolist(),
                "cam2lidar_translation": t_c2l.tolist(),
            }
            split_lists[split].append(token)
            per_cam_count[f"{split}:{cam}"] += 1
            done = sum(per_cam_count.values())
            if done % 500 == 0:
                print(f"  {site} packed {done} frames...", flush=True)
    finally:
        reader.close()

    for name, toks in split_lists.items():
        (work / "ImageSets" / f"{name}.txt").write_text(
            "\n".join(toks) + ("\n" if toks else ""), encoding="utf-8")
    (work / "frame_map.json").write_text(
        json.dumps(frame_map, indent=2), encoding="utf-8")
    total_box = audit["in_range"] + audit["out_range"]
    audit["in_range_frac"] = (audit["in_range"] / total_box) if total_box else None
    audit["n_frames"] = len(frame_map)
    (out_dir / "range_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    print(f"{site}: frames={audit['n_frames']} in_range={audit['in_range']} "
          f"out_range={audit['out_range']} frac={audit['in_range_frac']}")
    return audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", default=",".join(ALL_SITES))
    ap.add_argument("--max-frames-per-cam", type=int, default=None)
    args = ap.parse_args()
    PACK_WORK.mkdir(parents=True, exist_ok=True)
    PACK_OUT.mkdir(parents=True, exist_ok=True)
    for site in parse_sites(args.sites):
        pack_site(site, args.max_frames_per_cam)


if __name__ == "__main__":
    main()
