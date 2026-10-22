#!/usr/bin/env python3
"""Replay cross-camera association on one coordinate field and score TrackEval.

Detections and image tracks stay frozen. final_world_x/y is overwritten with
the coordinate under test, the full overlap edge list is written into the
database, and fusion is rerun with the paper merge flags. Same-camera
stitching stays off.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys

import numpy as np

from bootstrap import ARTIFACT_ROOT, SRC_ROOT
from db_access import (
    camera_polygons,
    checkpoint_dir_for,
    connect_ro,
    deploy_file,
    load_adapted,
    load_bias,
    load_mlp,
    load_test_keys,
)
from protocol import (
    DET_BATCH,
    GT_BATCH,
    MAX_T,
    SITES,
    geometry_db,
    match_report,
    origin_db,
    overlap_path,
    paper_merge_args,
    site_dir,
)


def paper_edges(site: str) -> list[str]:
    """Published association edges. A working overlap file does not replace them."""
    path = ARTIFACT_ROOT / "configs" / "overlap" / f"{site}.json"
    if not path.exists():
        path = overlap_path(site)
    blob = json.loads(path.read_text(encoding="utf-8"))
    return [str(edge["edge"]) for edge in blob["edges"]]


def coordinate_updates(site: str, seed: int | None, *, spatial: bool = True) -> list[tuple]:
    """Write overlap-gated coordinates. ``spatial=False`` keeps camera support only."""
    from geometry.overlap_gate import inside_mask
    from pipeline.geometry_guided_residual_mlp import compute_geometry_base, xywh_to_xyxy

    adapted, sizes = load_adapted(site)
    polygons = camera_polygons(site) if spatial else {}
    deploy = None
    opened: set[str] = set()
    model = None
    if seed is not None:
        deploy = json.loads(deploy_file(site, seed).read_text(encoding="utf-8"))
        if deploy.get("deploy") == "model":
            opened = set(deploy.get("S") or [])
            model = load_mlp(site, seed, checkpoint_dir_for(site, seed))
    conn = connect_ro(geometry_db(site))
    raw = conn.execute(
        """SELECT id, camera_name, bbox_x, bbox_y, bbox_w, bbox_h, world_x, world_y
           FROM trajectory_observations
           WHERE scene_id=? AND batch_id=? AND world_x IS NOT NULL""",
        (site, DET_BATCH),
    ).fetchall()
    conn.close()
    packed, feats = [], []
    for oid, cam, bx, by_, bw, bh, wx, wy in raw:
        xy = (float(wx), float(wy))
        if model is None or cam not in opened or cam not in adapted or None in (bx, by_, bw, bh):
            packed.append((int(oid), xy, None, cam))
            continue
        homography, scale = adapted[cam]
        box = xywh_to_xyxy((float(bx), float(by_), float(bw), float(bh)))
        _anchor, _world, feature, _risk = compute_geometry_base(box, homography, scale, sizes[cam])
        packed.append((int(oid), xy, len(feats), cam))
        feats.append(feature.astype(np.float32))
    residual = np.zeros((0, 2))
    if feats:
        import torch
        with torch.no_grad():
            residual = model(torch.from_numpy(np.stack(feats))).cpu().numpy()
    allow = np.ones(len(packed), dtype=bool)
    if spatial and packed:
        by_camera: dict[str, list[int]] = {}
        for index, (_oid, _xy, feat_index, cam) in enumerate(packed):
            if feat_index is not None:
                by_camera.setdefault(cam, []).append(index)
        for camera, indexes in by_camera.items():
            chosen = np.asarray(indexes, dtype=int)
            points = np.asarray([packed[index][1] for index in chosen], dtype=np.float64)
            mask = inside_mask(points, polygons.get(camera, []))
            allow[chosen] = mask
    updates = []
    for index, (oid, xy, feat_index, _cam) in enumerate(packed):
        point = xy
        if feat_index is not None and allow[index]:
            point = (xy[0] + float(residual[feat_index, 0]), xy[1] + float(residual[feat_index, 1]))
        updates.append((point[0], point[1], oid))
    return updates


def score(site: str, db) -> tuple[list[dict], dict, dict]:
    from evaluation.bev_hota import load_gt_bins, load_predictions, load_time_maps_from_match_report
    from evaluation.hota_trackeval import hota_trackeval

    time_map = load_time_maps_from_match_report(match_report(site))
    pred, t0 = load_predictions(
        db, site, "final_world", time_map, load_test_keys(site),
        pred_batch=DET_BATCH, cameras=set(SITES[site]["cameras"]),
    )
    gt_bins = load_gt_bins(
        origin_db(site), site, load_bias(site), t0, gt_batch=GT_BATCH,
        cameras=set(SITES[site]["cameras"]), max_t=MAX_T.get(site),
    )
    rows = hota_trackeval(gt_bins, pred)
    bev = float(np.mean([row["hota"] for row in rows]))
    for row in rows:
        row["site"] = site
        row["bev_hota"] = bev
    return rows, gt_bins, pred


def save_bins(path, gt_bins: dict, pred_bins: dict) -> str:
    """Write the 10 Hz inputs TrackEval replays. The file hash is the published row id."""
    import hashlib

    def flat(bins: dict):
        times, identities, xy = [], [], []
        for timestamp in sorted(bins):
            for identity, point in bins[timestamp].items():
                times.append(int(timestamp))
                identities.append(int(identity))
                xy.append(point)
        return (
            np.asarray(times, dtype=np.int32),
            np.asarray(identities, dtype=np.int64),
            np.asarray(xy, dtype=np.float64).reshape(-1, 2),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    gt_t, gt_id, gt_xy = flat(gt_bins)
    pr_t, pr_id, pr_xy = flat(pred_bins)
    np.savez_compressed(
        path, gt_t=gt_t, gt_id=gt_id, gt_xy=gt_xy, pr_t=pr_t, pr_id=pr_id, pr_xy=pr_xy,
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, choices=sorted(SITES))
    parser.add_argument("--seed", type=int, default=None, help="omit for geometry only")
    parser.add_argument(
        "--no-spatial-gate",
        action="store_true",
        help="apply the residual on every supported-camera box; supplement ablation",
    )
    parser.add_argument(
        "--save-bins",
        action="store_true",
        help="write the 10 Hz TrackEval inputs under results/recorded/bins",
    )
    args = parser.parse_args()
    from utils.scene_geometry_db import save_scene_overlap_pairs

    tag = "geo" if args.seed is None else f"f0_s{args.seed}"
    dest = site_dir(args.site) / "track" / f"{tag}.db"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.copy2(geometry_db(args.site), dest)
    conn = sqlite3.connect(str(dest))
    conn.executemany(
        "UPDATE trajectory_observations SET final_world_x=?, final_world_y=? WHERE id=?",
        coordinate_updates(args.site, args.seed, spatial=not args.no_spatial_gate),
    )
    edges = [tuple(edge.split(":")) for edge in paper_edges(args.site)]
    save_scene_overlap_pairs(conn, args.site, edges, replace=True)
    conn.commit()
    conn.close()
    cmd = [
        sys.executable, "-u", str(SRC_ROOT / "pipeline" / "multi_camera_trajectory_fusion.py"),
        "--db", str(dest),
        "--scene", args.site,
        "--batch", DET_BATCH,
        "--remerge",
        "--remerge-from-db",
        "--remerge-quiet",
        "--no-view",
        "--remerge-world-coords", "final",
        *paper_merge_args(),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"[track] {args.site} {tag}", flush=True)
    completed = subprocess.run(cmd, cwd=str(ARTIFACT_ROOT), env=env)
    if completed.returncode != 0:
        raise SystemExit(f"remerge {args.site} failed")
    rows, gt_bins, pred_bins = score(args.site, dest)
    out = site_dir(args.site) / "track" / f"{tag}_hota.json"
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if args.save_bins:
        if args.seed is None:
            bin_name = f"{args.site}_geo25.npz"
        elif args.no_spatial_gate:
            bin_name = f"{args.site}_full_s{args.seed}.npz"
        else:
            bin_name = f"{args.site}_overlap_s{args.seed}.npz"
        bin_path = ARTIFACT_ROOT / "results" / "recorded" / "bins" / bin_name
        digest = save_bins(bin_path, gt_bins, pred_bins)
        manifest = {
            "file": bin_name,
            "sha256": digest,
            "hota": {f"{float(row['gate']):.1f}": float(row["hota"]) for row in rows},
        }
        bin_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[bins] {bin_name} {digest}")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
