#!/usr/bin/env python3
"""Pack KITTI-style BEVHeight sites from the released frozen split.

Frame membership and clocks come from data/work/sites/<recording>/.
3D supervision is the ``l0hn_gt_001`` set: an object is labeled only when
that origin db has it on the matching camera and label frame. Dimensions and
the raw-world center still come from the measurement the importer used.
Town05 uses the out_bbox file for that GT frame, not the detection frame index.
Ground height and origin are the values stored in the import metadata.
Images are not rewritten by ``--labels-only``.

Copy each step2_pack/work/<site>/ directory to the 4090 as data/<site>/.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ARTIFACT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parents[2]
PRIOR = ARTIFACT / "data" / "work" / "sites"
PACK_WORK = HERE / "step2_pack" / "work"
PACK_OUT = HERE / "step2_pack" / "outputs"
SITES = ("lumpi_M6", "lumpi_M3", "synth_Town05_O_day")

_delta_path = HERE / "kitti_pack_lib.py"
_spec = importlib.util.spec_from_file_location("delta_kitti_pack", _delta_path)
if _spec is None or _spec.loader is None:
    raise SystemExit(f"cannot load {_delta_path}")
_delta = importlib.util.module_from_spec(_spec)
sys.modules["delta_kitti_pack"] = _delta
_spec.loader.exec_module(_delta)

sys.path.insert(0, str(HERE))
from layout import SITE_SPECS, label_time, video_path  # noqa: E402

GT_BATCH = "l0hn_gt_001"
LABEL_FPS = 10.0


def _assert_write(path: Path) -> Path:
    dest = path.resolve()
    root = (HERE / "step2_pack").resolve()
    if dest != root and root not in dest.parents:
        raise SystemExit(f"refusing to write outside {root}: {dest}")
    return dest


def connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def load_split_frames(site: str) -> dict[str, dict[tuple[str, int], dict]]:
    path = PRIOR / site / "dataset_split_manifest.csv"
    if not path.is_file():
        raise SystemExit(f"missing split manifest: {path}")
    out = {"train": {}, "val": {}, "test": {}}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            split = row.get("split") or ""
            if split not in out:
                continue
            key = (row["camera_id"], int(row["frame_index"]))
            if key in out[split]:
                continue
            out[split][key] = {
                "timestamp": float(row["timestamp"]),
                "camera_id": row["camera_id"],
                "frame_index": int(row["frame_index"]),
                "split": split,
            }
    return out


def load_time_maps(site: str) -> dict[str, tuple[float, float]]:
    path = PRIOR / site / "det_gt_match_report.json"
    if not path.is_file():
        raise SystemExit(f"missing match report: {path}")
    blob = json.loads(path.read_text(encoding="utf-8"))
    tm = {}
    for cam, rec in blob.get("cameras", {}).items():
        if not isinstance(rec, dict) or "error" in rec:
            continue
        tm[cam] = (
            float(rec.get("dt_applied_sec", 0.0))
            + float(rec.get("drift_const_ms", 0.0)) / 1000.0,
            float(rec.get("drift_slope_ms_per_min", 0.0)) / 60000.0,
        )
    return tm


def released_import(site: str) -> tuple[list[float], float]:
    db = PRIOR / site / "origin.db"
    conn = connect_ro(db)
    row = conn.execute(
        "SELECT metadata_json FROM trajectory_batches WHERE scene_id=? AND batch_id=?",
        (site, GT_BATCH),
    ).fetchone()
    conn.close()
    if row is None or not row[0]:
        raise SystemExit(f"{site} has no {GT_BATCH} import metadata in {db}")
    meta = json.loads(row[0])
    origin = meta.get("origin_world", meta.get("origin_carla_world"))
    if origin is None or "z_ground" not in meta:
        raise SystemExit(f"{site} metadata missing origin or z_ground")
    return [float(origin[0]), float(origin[1])], float(meta["z_ground"])


def load_gt(site: str) -> tuple[dict[tuple[str, int], set[int]], dict[tuple[str, int, int], tuple]]:
    """Fab4 GT identities and the tight image box, keyed by label frame."""
    db = PRIOR / site / "origin.db"
    conn = connect_ro(db)
    rows = conn.execute(
        """SELECT camera_name, frame_index, global_id, bbox_x, bbox_y, bbox_w, bbox_h
           FROM trajectory_observations
           WHERE scene_id=? AND batch_id=? AND outside_reference_x IS NOT NULL""",
        (site, GT_BATCH),
    )
    ids: dict[tuple[str, int], set[int]] = defaultdict(set)
    boxes: dict[tuple[str, int, int], tuple] = {}
    for cam, fr, gid, x, y, w, h in rows:
        key = (cam, int(fr))
        ids[key].add(int(gid))
        if None not in (x, y, w, h) and float(w) > 0 and float(h) > 0:
            boxes[(cam, int(fr), int(gid))] = (
                float(x), float(y), float(x) + float(w), float(y) + float(h),
            )
    conn.close()
    return ids, boxes


def label_frame(t_label: float) -> int:
    return int(round(float(t_label) * LABEL_FPS)) + 1


def _apply_gt_box(rec: dict, xyxy: tuple, frame_wh: tuple[int, int]) -> None:
    x1, y1, x2, y2 = xyxy
    fw, fh = frame_wh
    rec["xmin"], rec["ymin"] = max(0.0, x1), max(0.0, y1)
    rec["xmax"], rec["ymax"] = min(float(fw), x2), min(float(fh), y2)
    rec["truncated"] = int(
        rec["xmin"] <= 2 or rec["ymin"] <= 2
        or rec["xmax"] >= fw - 2 or rec["ymax"] >= fh - 2
    )


def lumpi_boxes(index, calib, t_label: float, frame_wh, gt_ids, gt_boxes, cam: str):
    fr = label_frame(t_label)
    ts = (fr - 1) / LABEL_FPS
    keep = gt_ids.get((cam, fr), set())
    if abs(t_label - ts) > 0.5 / LABEL_FPS or not keep:
        return []
    objs = index.at_time(ts, max_gap=1e-3)
    if not objs:
        return []
    boxes = []
    for o in objs:
        if int(o["object_id"]) not in keep:
            continue
        xyxy = gt_boxes.get((cam, fr, int(o["object_id"])))
        if xyxy is None:
            continue
        bottom = np.array([o["cx"], o["cy"], o["cz"] - o["height"] / 2.0, 1.0])
        loc = calib.T_world_to_cam @ bottom
        direction = np.array([math.cos(o["heading"]), math.sin(o["heading"]), 0.0])
        dcam = calib.T_world_to_cam[:3, :3] @ direction
        ry = math.atan2(float(dcam[0]), float(dcam[2]))
        rec = _delta.kitti_box(
            loc[:3], (o["height"], o["width"], o["length"]), ry, xyxy,
        )
        if rec:
            _apply_gt_box(rec, xyxy, frame_wh)
            rec["gid"] = int(o["object_id"])
            boxes.append(rec)
    return boxes


def synth_boxes(bbox_path: Path, T_w2c, frame_wh, gt_ids, gt_boxes, cam: str, frame_index: int):
    keep = gt_ids.get((cam, frame_index), set())
    if not keep or not bbox_path.is_file():
        return []
    data = json.loads(bbox_path.read_text(encoding="utf-8"))
    boxes = []
    n = len(data.get("vehicle_id", []))
    for i in range(n):
        gid = int(data["vehicle_id"][i])
        if gid not in keep:
            continue
        xyxy = gt_boxes.get((cam, frame_index, gid))
        if xyxy is None:
            continue
        wc = np.asarray(data["world_coords"][i], dtype=np.float64)
        xyz = np.stack([wc[0], wc[1], wc[2]], axis=1)
        center = xyz[:4].mean(axis=0)
        loc = T_w2c @ np.append(center, 1.0)
        dx = xyz[0] - xyz[3]
        heading = math.atan2(float(dx[1]), float(dx[0]))
        direction = np.array([math.cos(heading), math.sin(heading), 0.0])
        dcam = T_w2c[:3, :3] @ direction
        ry = math.atan2(float(dcam[0]), float(dcam[2]))
        length = float(np.linalg.norm(xyz[0, :2] - xyz[3, :2]))
        width = float(np.linalg.norm(xyz[0, :2] - xyz[1, :2]))
        height = float(abs(xyz[4, 2] - xyz[0, 2])) if len(xyz) >= 5 else 1.5
        rec = _delta.kitti_box(loc[:3], (height, width, length), ry, xyxy)
        if rec:
            _apply_gt_box(rec, xyxy, frame_wh)
            rec["gid"] = gid
            boxes.append(rec)
    return boxes


def label_alignment(rec: dict, tm: dict, kind: str):
    """Return ((camera, label_frame), abs time error), or None when off the 10 Hz tick."""
    cam = rec["camera_id"]
    t = float(rec["timestamp"])
    if kind == "lumpi":
        a, b = tm.get(cam, (0.0, 0.0))
        t_label = t + a + b * t
    else:
        t_label = t
    fr = label_frame(t_label)
    ts = (fr - 1) / LABEL_FPS
    if abs(t_label - ts) > 0.5 / LABEL_FPS:
        return None
    return (cam, fr), abs(t_label - ts)


def nearest_train_tokens(frame_map: dict, tm: dict, kind: str) -> set[str]:
    """One train/val image per (camera, 10 Hz label frame): the closest timestamp."""
    groups: dict[tuple, list[tuple[float, str]]] = defaultdict(list)
    for token, rec in frame_map.items():
        if rec.get("split") == "test":
            continue
        hit = label_alignment(rec, tm, kind)
        if hit is None:
            continue
        key, err = hit
        groups[key].append((err, token))
    return {min(items)[1] for items in groups.values()}


def boxes_for_frame(site, spec, rec, tm, gt_ids, gt_boxes, calibs, lidx, T_kitti):
    cam = rec["camera_id"]
    wh = tuple(int(v) for v in rec["image_wh"])
    if spec.kind == "lumpi":
        t_lab = label_time(float(rec["timestamp"]), tm.get(cam, (0.0, 0.0)))
        return lumpi_boxes(lidx, calibs[cam], t_lab, wh, gt_ids, gt_boxes, cam)
    gt_fr = label_frame(float(rec["timestamp"]))
    bbox_path = ARTIFACT / spec.data_rel / cam / "out_bbox" / f"{gt_fr:06d}.txt"
    return synth_boxes(
        bbox_path, np.asarray(T_kitti[cam], dtype=np.float64), wh, gt_ids, gt_boxes, cam, gt_fr,
    )


def pack_site(site: str, max_frames_per_cam: int | None) -> dict:
    if site not in SITES:
        raise SystemExit(f"{site} is not in the BEVHeight subset {SITES}")
    spec = SITE_SPECS[site]
    work = _assert_write(PACK_WORK / site)
    out_dir = _assert_write(PACK_OUT / site)
    for sub in ("images", "calib", "denorm", "label_2", "ImageSets"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_split_frames(site)
    tm = load_time_maps(site)
    origin, z_ground = released_import(site)
    gt_ids, gt_boxes = load_gt(site)
    frame_map = {}
    audit = {
        "site": site,
        "split_manifest": str(PRIOR / site / "dataset_split_manifest.csv"),
        "match_report": str(PRIOR / site / "det_gt_match_report.json"),
        "gt_batch": GT_BATCH,
        "origin_xy": origin,
        "z_ground": z_ground,
        "label_rule": "origin-db identity and tight box; raw 3D center",
        "batch_id": "l0hn_yolo_001",
        "in_range": 0,
        "out_range": 0,
        "no_box": 0,
        "missing_image": 0,
        "by_cam": {},
        "by_split": {},
    }
    split_lists = {k: [] for k in ("train", "val", "test")}

    if spec.kind == "lumpi":
        calibs = _delta.load_lumpi_calibs(spec)
        data_root = ARTIFACT / "data" / "external" / "LUMPI"
        labels = _delta.lumpi.parse_label_csv(
            _delta.lumpi.label_path_for_measurement(data_root, spec.measurement))
        lidx = _delta.LumpiTimeIndex(labels)
        T_kitti = {cam: calibs[cam].T_world_to_cam for cam in spec.cameras}
        Ks = {cam: calibs[cam].K for cam in spec.cameras}
    else:
        calibs = {}
        lidx = None
        calib_dir = ARTIFACT / spec.calib_rel
        T_kitti = {}
        Ks = {}
        for cam in spec.cameras:
            raw = _delta.synth.load_camera_calib(calib_dir, int(cam[1:]), z_ground=z_ground)
            T_kitti[cam] = _delta.synth_kitti_T(raw.T_world_to_cam)
            Ks[cam] = raw.K

    per_cam_count: dict[str, int] = defaultdict(int)
    items = []
    for frames in splits.values():
        items.extend(frames.values())
    items.sort(key=lambda r: (r["camera_id"], r["frame_index"]))
    reader = _delta.FrameReader()
    try:
        for rec in items:
            cam = rec["camera_id"]
            split = rec["split"]
            if cam not in spec.cameras:
                continue
            if max_frames_per_cam is not None and per_cam_count[f"{split}:{cam}"] >= max_frames_per_cam:
                continue
            token = _delta.token_of(cam, rec["frame_index"])
            video = video_path(spec, cam)
            img = reader.get(video, rec["timestamp"], rec["frame_index"])
            if img is None:
                audit["missing_image"] += 1
                print(f"  missing image {site} {token}", flush=True)
                continue
            wh = (int(img.shape[1]), int(img.shape[0]))
            jpg = work / "images" / f"{token}.jpg"
            _delta.cv2.imwrite(str(jpg), img, [int(_delta.cv2.IMWRITE_JPEG_QUALITY), _delta.JPEG_QUALITY])
            K = Ks[cam]
            (work / "calib" / f"{token}.txt").write_text(_delta._kitti_P2(K), encoding="utf-8")
            den = _delta._denorm_from_w2c(T_kitti[cam], z_ground)
            (work / "denorm" / f"{token}.txt").write_text(
                " ".join(f"{x:.8f}" for x in den) + "\n", encoding="utf-8")
            T_c2l, t_c2l = _delta.get_cam2lidar(den)
            T_l2c = np.linalg.inv(T_c2l)
            T_c2w = np.linalg.inv(T_kitti[cam])
            T_l2w = T_c2w @ T_l2c

            held = {
                "camera_id": cam,
                "timestamp": rec["timestamp"],
                "image_wh": list(wh),
            }
            boxes = boxes_for_frame(
                site, spec, held, tm, gt_ids, gt_boxes, calibs, lidx, T_kitti,
            )

            label_boxes = boxes if split != "test" else []
            _delta.write_kitti_label(work / "label_2" / f"{token}.txt", label_boxes)

            n_in = 0
            for b in boxes:
                loc = np.array([b["lx"], b["ly"], b["lz"], 1.0])
                lid = T_c2l @ loc
                if _delta.in_official_bev(lid[:3]):
                    n_in += 1
                    audit["in_range"] += 1
                else:
                    audit["out_range"] += 1
            if not boxes:
                audit["no_box"] += 1
            cam_rec = audit["by_cam"].setdefault(cam, {"in_range": 0, "out_range": 0, "frames": 0})
            cam_rec["in_range"] += n_in
            cam_rec["out_range"] += max(0, len(boxes) - n_in)
            cam_rec["frames"] += 1
            audit["by_split"][split] = audit["by_split"].get(split, 0) + 1

            frame_map[token] = {
                "site": site,
                "camera_id": cam,
                "frame_index": rec["frame_index"],
                "timestamp": rec["timestamp"],
                "split": split,
                "image_wh": list(wh),
                "K": np.asarray(K).tolist(),
                "T_world_to_kitti_cam": np.asarray(T_kitti[cam]).tolist(),
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
    (work / "frame_map.json").write_text(json.dumps(frame_map), encoding="utf-8")
    total_box = audit["in_range"] + audit["out_range"]
    audit["in_range_frac"] = (audit["in_range"] / total_box) if total_box else None
    audit["n_frames"] = len(frame_map)
    audit["expected_frames"] = sum(len(v) for v in splits.values())
    text = json.dumps(audit, indent=2)
    (work / "range_audit.json").write_text(text, encoding="utf-8")
    (out_dir / "range_audit.json").write_text(text, encoding="utf-8")
    print(
        f"{site}: frames={audit['n_frames']}/{audit['expected_frames']} "
        f"missing={audit['missing_image']} in_range={audit['in_range']} "
        f"out_range={audit['out_range']} frac={audit['in_range_frac']}",
        flush=True,
    )
    if audit["missing_image"]:
        raise SystemExit(f"{site}: {audit['missing_image']} frames had no image")
    return audit


def rewrite_labels(site: str) -> dict:
    """Rewrite denorm, lidar transforms, and train/val labels. Keep the JPEGs."""
    spec = SITE_SPECS[site]
    work = _assert_write(PACK_WORK / site)
    fmap_path = work / "frame_map.json"
    if not fmap_path.is_file():
        raise SystemExit(f"missing {fmap_path}; pack images before --labels-only")
    origin, z_ground = released_import(site)
    gt_ids, gt_boxes = load_gt(site)
    tm = load_time_maps(site)
    frame_map = json.loads(fmap_path.read_text(encoding="utf-8"))
    if spec.kind == "lumpi":
        calibs = _delta.load_lumpi_calibs(spec)
        data_root = ARTIFACT / "data" / "external" / "LUMPI"
        labels = _delta.lumpi.parse_label_csv(
            _delta.lumpi.label_path_for_measurement(data_root, spec.measurement))
        lidx = _delta.LumpiTimeIndex(labels)
        T_kitti = {cam: calibs[cam].T_world_to_cam for cam in spec.cameras}
    else:
        calibs, lidx = {}, None
        calib_dir = ARTIFACT / spec.calib_rel
        T_kitti = {}
        for cam in spec.cameras:
            raw = _delta.synth.load_camera_calib(calib_dir, int(cam[1:]), z_ground=z_ground)
            T_kitti[cam] = _delta.synth_kitti_T(raw.T_world_to_cam)

    audit = {
        "site": site,
        "mode": "labels-only",
        "gt_batch": GT_BATCH,
        "origin_xy": origin,
        "z_ground": z_ground,
        "label_rule": "GT identity; train/val keeps the image nearest each 10 Hz label frame",
        "in_range": 0,
        "out_range": 0,
        "no_box": 0,
        "train_val_boxes": 0,
        "train_val_images_kept": 0,
        "train_val_images_total": 0,
        "by_cam": {},
        "by_split": {},
    }
    kept = nearest_train_tokens(frame_map, tm, spec.kind)
    audit["train_val_images_kept"] = len(kept)
    audit["train_val_images_total"] = sum(1 for rec in frame_map.values() if rec.get("split") != "test")
    for name in ("train", "val"):
        path = work / "ImageSets" / f"{name}.txt"
        if not path.is_file():
            continue
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        chosen = [ln for ln in lines if ln in kept]
        path.write_text("\n".join(chosen) + ("\n" if chosen else ""), encoding="utf-8")
    n = 0
    for token, rec in frame_map.items():
        cam = rec["camera_id"]
        split = rec["split"]
        wh = tuple(int(v) for v in rec["image_wh"])
        T = np.asarray(T_kitti[cam], dtype=np.float64)
        den = _delta._denorm_from_w2c(T, z_ground)
        (work / "denorm" / f"{token}.txt").write_text(
            " ".join(f"{x:.8f}" for x in den) + "\n", encoding="utf-8")
        T_c2l, t_c2l = _delta.get_cam2lidar(den)
        T_l2w = np.linalg.inv(T) @ np.linalg.inv(T_c2l)
        rec["T_world_to_kitti_cam"] = T.tolist()
        rec["T_kitti_cam_to_world"] = np.linalg.inv(T).tolist()
        rec["T_cam2lidar"] = T_c2l.tolist()
        rec["T_lidar_to_world"] = T_l2w.tolist()
        rec["denorm"] = den.tolist()
        rec["cam2lidar_translation"] = t_c2l.tolist()
        boxes = boxes_for_frame(site, spec, rec, tm, gt_ids, gt_boxes, calibs, lidx, T_kitti)
        label_boxes = boxes if token in kept else []
        _delta.write_kitti_label(work / "label_2" / f"{token}.txt", label_boxes)
        audit["train_val_boxes"] += len(label_boxes)
        n_in = 0
        for b in boxes:
            loc = np.array([b["lx"], b["ly"], b["lz"], 1.0])
            lid = T_c2l @ loc
            if _delta.in_official_bev(lid[:3]):
                n_in += 1
                audit["in_range"] += 1
            else:
                audit["out_range"] += 1
        if not boxes:
            audit["no_box"] += 1
        cam_rec = audit["by_cam"].setdefault(cam, {"in_range": 0, "out_range": 0, "frames": 0})
        cam_rec["in_range"] += n_in
        cam_rec["out_range"] += max(0, len(boxes) - n_in)
        cam_rec["frames"] += 1
        audit["by_split"][split] = audit["by_split"].get(split, 0) + 1
        n += 1
        if n % 2000 == 0:
            print(f"  {site} relabeled {n} frames...", flush=True)
    total = audit["in_range"] + audit["out_range"]
    audit["in_range_frac"] = (audit["in_range"] / total) if total else None
    audit["n_frames"] = n
    text = json.dumps(audit, indent=2)
    (work / "range_audit.json").write_text(text, encoding="utf-8")
    out_dir = _assert_write(PACK_OUT / site)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "range_audit.json").write_text(text, encoding="utf-8")
    fmap_path.write_text(json.dumps(frame_map), encoding="utf-8")
    print(
        f"{site}: frames={n} train_val_boxes={audit['train_val_boxes']} "
        f"in_range={audit['in_range']} out_range={audit['out_range']} "
        f"z_ground={z_ground}",
        flush=True,
    )
    return audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", default=",".join(SITES))
    ap.add_argument("--max-frames-per-cam", type=int, default=None)
    ap.add_argument("--labels-only", action="store_true",
                    help="rewrite denorm and GT labels; keep existing JPEGs")
    args = ap.parse_args()
    sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    unknown = [s for s in sites if s not in SITES]
    if unknown:
        raise SystemExit(f"unknown sites {unknown}; allowed {SITES}")
    PACK_WORK.mkdir(parents=True, exist_ok=True)
    PACK_OUT.mkdir(parents=True, exist_ok=True)
    for site in sites:
        if args.labels_only:
            rewrite_labels(site)
        else:
            pack_site(site, args.max_frames_per_cam)


if __name__ == "__main__":
    main()
