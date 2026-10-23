#!/usr/bin/env python3
"""Time detector, the localization path, and fusion as separate passes.

HSG is the published projection, evaluated for every box in a camera frame in
one vectorized call. Overlap and the residual are timed in that same frame.
Slice percentiles are not added into the path. Detector and fusion stay separate.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

BUNDLE = Path(__file__).resolve().parent
DET_BATCH = "l0hn_yolo_001"
SITES = ("lumpi_M6", "synth_Town05_O_day", "ko_per")


def percentile(values: list[float]) -> tuple[float, float, int]:
    if not values:
        return float("nan"), float("nan"), 0
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, 50)), float(np.percentile(array, 95)), int(array.size)


def enable_host_packages() -> None:
    import site

    paths = []
    user = site.getusersitepackages()
    if isinstance(user, str):
        paths.append(user)
    base = getattr(sys, "base_prefix", sys.prefix)
    ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    paths.extend([
        os.path.join(base, "lib", ver, "dist-packages"),
        os.path.join(base, "local", "lib", ver, "dist-packages"),
        os.path.join(base, "lib", ver, "site-packages"),
    ])
    for path in paths:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


def load_net(ckpt: Path):
    import torch
    from pipeline.geometry_guided_residual_mlp import GeometryGuidedResidualMLP

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    mlp = GeometryGuidedResidualMLP(
        in_dim=int(blob["in_dim"]), hidden=64, max_residual_m=float(blob["max_residual_m"]),
    )
    mlp.load_state_dict(blob["state_dict"])
    mlp.eval()
    return mlp


def project_jac(homography: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hom = np.column_stack([points, np.ones(len(points))]) @ homography.T
    den = hom[:, 2]
    xy = hom[:, :2] / den[:, None]
    jac = (homography[:2, :2][None] - xy[:, :, None] * homography[2, :2][None, None]) / den[:, None, None]
    return xy, jac


def geometry(box: np.ndarray, homography: np.ndarray, wh: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Published HSG anchor and 7-D feature for a whole camera frame."""
    width = np.maximum(box[:, 2], 1e-6)
    height = np.maximum(box[:, 3], 1e-6)
    center = box[:, :2] + box[:, 2:] / 2.0
    _xy, jac = project_jac(homography, center)
    sensitivity = np.linalg.norm(jac * np.asarray(wh, dtype=np.float64)[None, None], axis=(1, 2))
    alpha = sensitivity / (sensitivity + 60.0)
    if np.linalg.norm(homography[2, :2]) >= 1e-9:
        probe = homography[2] @ np.array([wh[0] / 2.0, wh[1] - 1.0, 1.0])
        ground = (np.column_stack([center, np.ones(len(center))]) @ homography[2]) * (1.0 if probe >= 0 else -1.0) > 0
        alpha = alpha.copy()
        alpha[~ground] = 1.0
    rho = np.clip(height / width, 0.5, 3.0)
    phi = rho ** 2 / (1.0 + rho ** 2)
    gate = 4.0 * alpha * (1.0 - alpha)
    beta = alpha + 0.75 * gate * (phi - alpha)
    anchor = center.copy()
    anchor[:, 1] += beta * height / 2.0
    xy, anchor_jac = project_jac(homography, anchor)
    wm = width * np.linalg.norm(anchor_jac[:, :, 0], axis=1)
    hm = height * np.linalg.norm(anchor_jac[:, :, 1], axis=1)
    feat = np.column_stack([
        center[:, 0] / wh[0], center[:, 1] / wh[1], wm / 10.0, hm / 10.0, 1.0 / rho, alpha, gate,
    ]).astype(np.float32)
    return xy, feat


def inside_mask(points: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray:
    """Paper membership: signed distance at least -1e-6 m, on the HSG point."""
    from geometry.overlap_gate import inside_mask as published_mask

    return published_mask(np.asarray(points, dtype=np.float64), polygons)


def load_frames(db: Path, site: str) -> dict[tuple[str, int], list[tuple[float, float, float, float]]]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute(
        """SELECT camera_name, frame_index, bbox_x, bbox_y, bbox_w, bbox_h
           FROM trajectory_observations
           WHERE scene_id=? AND batch_id=? AND bbox_w IS NOT NULL
           ORDER BY camera_name, frame_index""",
        (site, DET_BATCH),
    ).fetchall()
    conn.close()
    grouped: dict[tuple[str, int], list] = {}
    for camera, frame, *box in rows:
        if None in box:
            continue
        grouped.setdefault((str(camera), int(frame)), []).append(tuple(float(value) for value in box))
    return grouped


def time_localization(site: str, limit_frames: int) -> dict:
    import torch

    folder = BUNDLE / "data" / site
    support = json.loads((folder / "support.json").read_text(encoding="utf-8"))
    polygons = json.loads((folder / "polygons.json").read_text(encoding="utf-8"))
    calib = np.load(folder / "calib.npz")
    cameras = list(support["cameras"])
    mlp = load_net(folder / "checkpoint_seed0.pt")
    frames = load_frames(folder / "geometry.db", site)
    keys = [key for key in frames if f"hm_{key[0]}" in calib.files]
    if limit_frames > 0:
        keys = keys[:limit_frames]
    opened = set(support["S"])
    path_ms, hsg_ms, overlap_ms, mlp_ms = [], [], [], []
    n_box, n_active = [], []
    print(
        f"[runtime] {site} localization frames={len(keys)} edges={len(support['edges'])} S={support['S']}",
        flush=True,
    )
    for index, key in enumerate(keys):
        camera, _frame = key
        boxes = np.asarray(frames[key], dtype=np.float64)
        homography = np.asarray(calib[f"hm_{camera}"], dtype=np.float64)
        wh = tuple(int(value) for value in calib[f"wh_{camera}"])
        polys = [np.asarray(item, dtype=np.float32) for item in polygons.get(camera, [])]
        t0 = time.perf_counter()
        xy, feat = geometry(boxes, homography, wh)
        t1 = time.perf_counter()
        mask = inside_mask(xy, polys)
        t2 = time.perf_counter()
        active = 0
        if camera in opened and len(feat):
            batch = torch.from_numpy(feat)
            with torch.no_grad():
                residual = mlp(batch).numpy()
            _applied = residual * mask[:, None]
            active = int(mask.sum())
        t3 = time.perf_counter()
        hsg_ms.append((t1 - t0) * 1000.0)
        overlap_ms.append((t2 - t1) * 1000.0)
        mlp_ms.append((t3 - t2) * 1000.0)
        path_ms.append((t3 - t0) * 1000.0)
        n_box.append(len(boxes))
        n_active.append(active)
        if index and index % 2000 == 0:
            print(f"[runtime] {site} {index}/{len(keys)}", flush=True)
    if len(path_ms) > 20:
        path_ms, hsg_ms = path_ms[10:], hsg_ms[10:]
        overlap_ms, mlp_ms = overlap_ms[10:], mlp_ms[10:]
        n_box, n_active = n_box[10:], n_active[10:]
    p50, p95, n = percentile(path_ms)
    h50, h95, _n = percentile(hsg_ms)
    o50, o95, _n = percentile(overlap_ms)
    m50, m95, _n = percentile(mlp_ms)
    return {
        "site": site,
        "seed": 0,
        "hsg_call": "vectorized_frame",
        "edges": len(support["edges"]),
        "cameras": len(cameras),
        "S": support["S"],
        "frames": n,
        "boxes_per_frame_p50": percentile([float(v) for v in n_box])[0],
        "activation_fraction": float(np.sum(n_active) / max(np.sum(n_box), 1)),
        "path_ms_p50": p50,
        "path_ms_p95": p95,
        "hsg_ms_p50": h50,
        "hsg_ms_p95": h95,
        "overlap_ms_p50": o50,
        "overlap_ms_p95": o95,
        "mlp_ms_p50": m50,
        "mlp_ms_p95": m95,
        "sum_of_slice_p50_ms": h50 + o50 + m50,
    }


def time_fusion(site: str, edges: list[str]) -> dict:
    from utils.scene_geometry_db import save_scene_overlap_pairs

    db = BUNDLE / "data" / site / "geometry.db"
    work = BUNDLE / "work" / site
    work.mkdir(parents=True, exist_ok=True)
    dest = work / "remerge.db"
    if dest.exists():
        dest.unlink()
    shutil.copy2(db, dest)
    conn = sqlite3.connect(str(dest))
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute(
        """UPDATE trajectory_observations
           SET final_world_x=world_x, final_world_y=world_y
           WHERE scene_id=? AND batch_id=? AND world_x IS NOT NULL""",
        (site, DET_BATCH),
    )
    save_scene_overlap_pairs(
        conn, site, [tuple(sorted(edge.split(":"))) for edge in edges], replace=True,
    )
    conn.commit()
    conn.close()
    csv_path = work / "remerge_batches.csv"
    fusion = BUNDLE / "src" / "pipeline" / "multi_camera_trajectory_fusion.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BUNDLE / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, "-u", str(fusion),
        "--db", str(dest), "--scene", site, "--batch", DET_BATCH,
        "--remerge", "--remerge-from-db", "--remerge-dry-run", "--no-view",
        "--remerge-world-coords", "final",
        "--remerge-profile", str(csv_path),
        "--merge-score-threshold", "0.70", "--merge-max-distance", "2.5",
    ]
    log = work / "remerge.log"
    print(f"[runtime] {site} fusion remerge", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(cmd, cwd=str(BUNDLE), env=env, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"remerge {site} rc={completed.returncode} see {log}")
    add, merge = [], []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            add.append(float(row["add_ms"]))
            merge.append(float(row["merge_ms"]))
    both = [left + right for left, right in zip(add, merge)]
    p50, p95, n = percentile(both)
    return {"batches": n, "fusion_ms_p50": p50, "fusion_ms_p95": p95}


def time_detector(engine: Path, clips: list[Path], limit_frames: int) -> dict:
    from ultralytics import YOLO

    print(f"[runtime] detector {engine} clips={len(clips)}", flush=True)
    model = YOLO(str(engine), task="detect")
    samples = []
    for clip in clips:
        cap = cv2.VideoCapture(str(clip))
        n = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            model.track(
                frame, persist=True, imgsz=[384, 640], verbose=False,
                tracker=str(BUNDLE / "botsort_static.yaml"),
            )
            samples.append((time.perf_counter() - t0) * 1000.0)
            n += 1
            if limit_frames > 0 and n >= limit_frames:
                break
        cap.release()
    if len(samples) > 10:
        samples = samples[10:]
    p50, p95, n = percentile(samples)
    return {"detector_frames": n, "detector_ms_p50": p50, "detector_ms_p95": p95}


def fmt_pair(row: dict, p50_key: str, p95_key: str) -> str:
    p50, p95 = row.get(p50_key), row.get(p95_key)
    if p50 in (None, "") or p95 in (None, ""):
        return ""
    return f"{float(p50):.2f}/{float(p95):.2f}"


def render(rows: list[dict]) -> str:
    board = BUNDLE / "outputs" / "board_check.txt"
    lines = [
        "# Component latency",
        "",
        "Milliseconds. The localization path is one measured camera frame: vectorized HSG for every box, then the overlap test, then the seed-0 residual. HSG, overlap, and MLP columns are slices of that same call. Their percentiles are not added to produce the path. Detector and fusion are separate passes. Fusion is one synchronized batch. No stream fps is computed from these percentiles.",
        "",
        "Detector, when present, is YOLO26l TensorRT FP16, 384×640, BoT-SORT with gmc_method none, one clip at a time. HSG and the residual run on CPU. Fusion is a dry-run remerge, score 0.70, max distance 2.5 m. Support S is the retained validation-pair set. This file records that measurement and the board configuration. A later run with the same configuration is a new measurement and is not required to match these percentiles. This file does not claim power, memory, or station throughput. The detector column is empty: this package has no clip replay for that separate pass.",
        "",
    ]
    if board.exists():
        lines.extend(["## Board check", "", "```text", board.read_text(encoding="utf-8").rstrip(), "```", ""])
    lines.extend([
        "| site | edges | S | boxes/frame p50 | activation | detector p50/p95 | path p50/p95 | HSG p50/p95 | overlap p50/p95 | MLP p50/p95 | fusion p50/p95 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        lines.append(
            f"| {row['site']} | {row['edges']} | {','.join(row['S'])} | {row['boxes_per_frame_p50']:.2f} | "
            f"{row['activation_fraction']:.4f} | {fmt_pair(row, 'detector_ms_p50', 'detector_ms_p95')} | "
            f"{fmt_pair(row, 'path_ms_p50', 'path_ms_p95')} | {fmt_pair(row, 'hsg_ms_p50', 'hsg_ms_p95')} | "
            f"{fmt_pair(row, 'overlap_ms_p50', 'overlap_ms_p95')} | {fmt_pair(row, 'mlp_ms_p50', 'mlp_ms_p95')} | "
            f"{fmt_pair(row, 'fusion_ms_p50', 'fusion_ms_p95')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--skip-fusion", action="store_true")
    parser.add_argument("--engine", default="")
    parser.add_argument("--clips", default="")
    parser.add_argument("--sites", nargs="*", default=list(SITES))
    args = parser.parse_args()
    sys.path.insert(0, str(BUNDLE / "src"))
    enable_host_packages()
    rows = []
    for site in args.sites:
        print(f"[runtime] {site}", flush=True)
        support = json.loads((BUNDLE / "data" / site / "support.json").read_text(encoding="utf-8"))
        row = time_localization(site, args.limit_frames)
        if not args.skip_fusion and args.limit_frames <= 0:
            row.update(time_fusion(site, list(support["edges"])))
        else:
            row.update({"batches": 0, "fusion_ms_p50": "", "fusion_ms_p95": ""})
        clip_dir = Path(args.clips) / site if args.clips else BUNDLE / "data" / "clips" / site
        clips = sorted(clip_dir.glob("*.mp4")) if clip_dir.is_dir() else []
        if args.engine and clips:
            row.update(time_detector(Path(args.engine), clips, args.limit_frames or 0))
        else:
            row["detector_clips"] = len(clips)
            row["detector_ms_p50"] = ""
            row["detector_ms_p95"] = ""
        rows.append(row)
        print(json.dumps({key: row[key] for key in row if key != "S"}, default=str), flush=True)
    dest = BUNDLE / "outputs"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "runtime_raw.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (dest / "runtime.md").write_text(render(rows), encoding="utf-8")
    print(f"[runtime] -> {dest / 'runtime.md'}", flush=True)


if __name__ == "__main__":
    main()
