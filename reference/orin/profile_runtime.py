#!/usr/bin/env python3
"""Time HSG projection, the residual MLP, and fusion. Detector is optional."""
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

import numpy as np

BUNDLE = Path(__file__).resolve().parent
ARTIFACT = BUNDLE.parents[1]
SRC = ARTIFACT / "src"
SITES = {
    "lumpi_M6": "lumpi_M6",
    "synth_Town05_O_day": "synth_Town05_O_day",
}
DET_BATCH = "l0hn_yolo_001"


def percentile(xs: list[float]) -> tuple[float, float, int]:
    if not xs:
        return float("nan"), float("nan"), 0
    a = np.asarray(xs, dtype=np.float64)
    return float(np.percentile(a, 50)), float(np.percentile(a, 95)), int(a.size)


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


def load_adapted(db: Path, site: str, cameras: list[str]):
    from pipeline.geometry_guided_residual_mlp import GeometryDataView, _infer_and_calib_wh_from_db
    from pipeline.multi_camera_trajectory_fusion import adapt_homography_to_resolution

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    view = GeometryDataView()
    cal = view.load_camera_calibrations(conn, site, cameras)
    infer_wh, calib_wh = _infer_and_calib_wh_from_db(conn, site, DET_BATCH, cameras)
    adapted, sizes = {}, {}
    for cam, (h, scale) in cal.items():
        iw, cw = infer_wh.get(cam), calib_wh.get(cam)
        if iw and cw and iw != cw:
            h = adapt_homography_to_resolution(h, cw, iw)
        adapted[cam] = (h, scale)
        sizes[cam] = tuple(iw or cw or (1920, 1080))
    conn.close()
    return adapted, sizes


def time_boxes(db: Path, site: str, ckpt: Path, opened: set[str], limit_frames: int) -> dict:
    print(f"[runtime] {site} load torch and calibrations", flush=True)
    import torch
    from pipeline.geometry_guided_residual_mlp import compute_geometry_base, xywh_to_xyxy

    meta_cameras = json.loads((ARTIFACT / "results" / "recorded" / "deploy_support" / f"{site}_F0.json").read_text(encoding="utf-8"))
    cameras = list(meta_cameras["cameras"])
    adapted, sizes = load_adapted(db, site, cameras)
    mlp = load_net(ckpt)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    print(f"[runtime] {site} read boxes", flush=True)
    rows = conn.execute(
        """SELECT camera_name, frame_index, bbox_x, bbox_y, bbox_w, bbox_h
           FROM trajectory_observations
           WHERE scene_id=? AND batch_id=? AND bbox_w IS NOT NULL
           ORDER BY camera_name, frame_index""",
        (site, DET_BATCH),
    ).fetchall()
    conn.close()
    frames: dict[tuple[str, int], list] = {}
    for cam, fr, bx, by_, bw, bh in rows:
        if cam not in adapted or None in (bx, by_, bw, bh):
            continue
        frames.setdefault((cam, int(fr)), []).append((float(bx), float(by_), float(bw), float(bh)))
    keys = list(frames)
    if limit_frames > 0:
        keys = keys[:limit_frames]
    print(f"[runtime] {site} HSG+MLP frames={len(keys)}", flush=True)
    geo_ms, mlp_ms, n_box, n_mlp = [], [], [], []
    for i, key in enumerate(keys):
        cam, _fr = key
        boxes = frames[key]
        H, scale = adapted[cam]
        wh = sizes[cam]
        t0 = time.perf_counter()
        feats = []
        for box in boxes:
            _a, _w, feat, _r = compute_geometry_base(xywh_to_xyxy(box), H, scale, wh)
            feats.append(feat.astype(np.float32))
        dt_geo = (time.perf_counter() - t0) * 1000.0
        dt_mlp = 0.0
        used = 0
        if cam in opened and feats:
            batch = torch.from_numpy(np.stack(feats))
            t1 = time.perf_counter()
            with torch.no_grad():
                mlp(batch)
            dt_mlp = (time.perf_counter() - t1) * 1000.0
            used = len(feats)
        if i and i % 2000 == 0:
            print(f"[runtime] {site} frame {i}/{len(keys)}", flush=True)
        geo_ms.append(dt_geo)
        mlp_ms.append(dt_mlp)
        n_box.append(len(boxes))
        n_mlp.append(used)
    if len(geo_ms) > 20:
        geo_ms, mlp_ms = geo_ms[10:], mlp_ms[10:]
        n_box, n_mlp = n_box[10:], n_mlp[10:]
    g50, g95, gn = percentile(geo_ms)
    m50, m95, mn = percentile(mlp_ms)
    return {
        "frames": gn,
        "boxes_per_frame_p50": percentile([float(x) for x in n_box])[0],
        "hsg_ms_p50": g50,
        "hsg_ms_p95": g95,
        "mlp_frames": mn,
        "mlp_ms_p50": m50,
        "mlp_ms_p95": m95,
        "mlp_boxes_p50": percentile([float(x) for x in n_mlp])[0],
    }


def time_fusion(db: Path, site: str, edges: list[str]) -> dict:
    from utils.scene_geometry_db import save_scene_overlap_pairs

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
        conn, site, [tuple(sorted(e.split(":"))) for e in edges], replace=True,
    )
    conn.commit()
    conn.close()
    csv_path = work / "remerge_batches.csv"
    fusion = SRC / "pipeline" / "multi_camera_trajectory_fusion.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, "-u", str(fusion),
        "--db", str(dest), "--scene", site, "--batch", DET_BATCH,
        "--remerge", "--remerge-from-db", "--remerge-dry-run", "--no-view",
        "--remerge-world-coords", "final",
        "--remerge-profile", str(csv_path),
        "--merge-score-threshold", "0.70", "--merge-max-distance", "2.5",
    ]
    log = work / "remerge.log"
    print(f"[runtime] {site} fusion remerge, log {log}", flush=True)
    with log.open("w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, cwd=str(BUNDLE), env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"remerge {site} rc={rc} see {log}")
    add, merge = [], []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            add.append(float(row["add_ms"]))
            merge.append(float(row["merge_ms"]))
    both = [a + m for a, m in zip(add, merge)]
    a50, a95, n = percentile(both)
    m50, m95, _n = percentile(merge)
    return {"batches": n, "fusion_ms_p50": a50, "fusion_ms_p95": a95, "merge_only_ms_p50": m50, "merge_only_ms_p95": m95}


def time_detector(engine: Path, clips: list[Path], limit_frames: int) -> dict:
    from ultralytics import YOLO

    print(f"[runtime] detector {engine} clips={len(clips)}", flush=True)
    model = YOLO(str(engine), task="detect")
    samples = []
    for clip in clips:
        import cv2
        cap = cv2.VideoCapture(str(clip))
        n = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            model.track(
                frame, persist=True, imgsz=[384, 640], verbose=False,
                tracker=str(ARTIFACT / "configs" / "botsort_static.yaml"),
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


def enable_host_packages() -> None:
    """A venv hides ~/.local even when the host pip can see torch."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit-frames", type=int, default=0)
    ap.add_argument("--skip-fusion", action="store_true")
    ap.add_argument("--engine", default="")
    ap.add_argument("--clips", default="")
    args = ap.parse_args()
    sys.path.insert(0, str(SRC))
    enable_host_packages()
    out_rows = []
    for site in SITES:
        print(f"[runtime] {site}", flush=True)
        db = ARTIFACT / "data" / "work" / "sites" / site / "geometry.db"
        ckpt = ARTIFACT / "results" / "recorded" / "checkpoints" / f"{site}_F0_seed0.pt"
        dep = json.loads((ARTIFACT / "results" / "recorded" / "deploy_support" / f"{site}_F0.json").read_text(encoding="utf-8"))
        graph = json.loads((ARTIFACT / "configs" / "overlap" / f"{site}.json").read_text(encoding="utf-8"))
        edges = [e["edge"] for e in graph["edges"]]
        row = {"site": site, "S": ",".join(dep.get("S") or []), "cameras": len(dep["cameras"]), "edges": len(edges)}
        row.update(time_boxes(db, site, ckpt, set(dep.get("S") or []), args.limit_frames))
        if not args.skip_fusion and args.limit_frames <= 0:
            row.update(time_fusion(db, site, edges))
        else:
            row.update({"batches": 0, "fusion_ms_p50": "", "fusion_ms_p95": "", "merge_only_ms_p50": "", "merge_only_ms_p95": ""})
        clip_dir = Path(args.clips) / site if args.clips else BUNDLE / "data" / "clips" / site
        clips = sorted(clip_dir.glob("*.mp4")) if clip_dir.is_dir() else []
        if args.engine and clips:
            row.update(time_detector(Path(args.engine), clips, args.limit_frames or 0))
        else:
            row["detector_clips"] = len(clips)
        out_rows.append(row)
        print(json.dumps(row), flush=True)
    dest = BUNDLE / "outputs" / "runtime_raw.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out_rows, indent=2), encoding="utf-8")
    lines = [
        "# Runtime",
        "",
        "Milliseconds. Detector, HSG, and MLP are per camera frame. Fusion is per sync batch. Stream fps uses the sum of the three per-frame p50 times. Those stages were timed separately, so this is not one measured end-to-end loop. Fusion p95 is a batch tail and is not folded into fps.",
        "",
        "Detector is YOLO26l, TensorRT FP16, 384×640, BoT-SORT with gmc_method none, one clip at a time. HSG uses s0=60 m on CPU. The residual is 7→64→64→2 on CPU, on cameras listed in results/recorded/deploy_support. Fusion is a full-graph dry-run remerge, score 0.70, max distance 2.5 m. Board model, JetPack, and power mode are not in this file.",
        "",
        "| site | detector p50/p95 | HSG p50/p95 | MLP p50/p95 | frame sum p50 | stream fps | fusion p50/p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in out_rows:
        det = row.get("detector_ms_p50")
        if det in (None, ""):
            lines.append(
                f"| {row['site']} |  | {row['hsg_ms_p50']:.2f}/{row['hsg_ms_p95']:.2f} | "
                f"{row['mlp_ms_p50']:.2f}/{row['mlp_ms_p95']:.2f} |  |  | "
                f"{row['fusion_ms_p50']}/{row['fusion_ms_p95']} |"
            )
            continue
        frame = float(det) + float(row["hsg_ms_p50"]) + float(row["mlp_ms_p50"])
        fps = 1000.0 / frame if frame > 0 else float("nan")
        lines.append(
            f"| {row['site']} | {float(det):.2f}/{float(row['detector_ms_p95']):.2f} | "
            f"{row['hsg_ms_p50']:.2f}/{row['hsg_ms_p95']:.2f} | "
            f"{row['mlp_ms_p50']:.2f}/{row['mlp_ms_p95']:.2f} | {frame:.2f} | {fps:.1f} | "
            f"{float(row['fusion_ms_p50']):.2f}/{float(row['fusion_ms_p95']):.2f} |"
        )
    (BUNDLE / "outputs" / "runtime.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[runtime] -> {dest}", flush=True)


if __name__ == "__main__":
    main()
