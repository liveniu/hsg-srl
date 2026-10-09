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
from db_access import connect_ro, load_adapted, load_bias, load_mlp, load_test_keys
from protocol import (
    DET_BATCH,
    GT_BATCH,
    MAX_T,
    PAPER_MERGE_DISTANCE_M,
    PAPER_MERGE_SCORE,
    SITES,
    geometry_db,
    match_report,
    origin_db,
    overlap_path,
    site_dir,
)


def paper_edges(site: str) -> list[str]:
    path = overlap_path(site)
    if not path.exists():
        path = ARTIFACT_ROOT / "configs" / "overlap" / f"{site}.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    return [str(edge["edge"]) for edge in blob["edges"]]


def coordinate_updates(site: str, seed: int | None) -> list[tuple]:
    from pipeline.geometry_guided_residual_mlp import compute_geometry_base, xywh_to_xyxy

    adapted, sizes = load_adapted(site)
    deploy = None
    opened: set[str] = set()
    model = None
    if seed is not None:
        name = f"{site}_F0.json" if seed == 0 else f"{site}_F0_seed{seed}.json"
        deploy = json.loads((site_dir(site) / "deploy" / name).read_text(encoding="utf-8"))
        if deploy.get("deploy") == "model":
            opened = set(deploy.get("S") or [])
            model = load_mlp(site, seed)
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
            packed.append((int(oid), xy, None))
            continue
        homography, scale = adapted[cam]
        box = xywh_to_xyxy((float(bx), float(by_), float(bw), float(bh)))
        _anchor, _world, feature, _risk = compute_geometry_base(box, homography, scale, sizes[cam])
        packed.append((int(oid), xy, len(feats)))
        feats.append(feature.astype(np.float32))
    residual = np.zeros((0, 2))
    if feats:
        import torch
        with torch.no_grad():
            residual = model(torch.from_numpy(np.stack(feats))).cpu().numpy()
    updates = []
    for oid, xy, index in packed:
        if index is None:
            updates.append((xy[0], xy[1], oid))
        else:
            updates.append((xy[0] + float(residual[index, 0]), xy[1] + float(residual[index, 1]), oid))
    return updates


def score(site: str, db) -> list[dict]:
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
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, choices=sorted(SITES))
    parser.add_argument("--seed", type=int, default=None, help="omit for geometry only")
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
        coordinate_updates(args.site, args.seed),
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
        "--no-stitch",
        "--remerge-world-coords", "final",
        "--merge-score-threshold", str(PAPER_MERGE_SCORE),
        "--merge-max-distance", str(PAPER_MERGE_DISTANCE_M),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"[track] {args.site} {tag}", flush=True)
    completed = subprocess.run(cmd, cwd=str(ARTIFACT_ROOT), env=env)
    if completed.returncode != 0:
        raise SystemExit(f"remerge {args.site} failed")
    rows = score(args.site, dest)
    out = site_dir(args.site) / "track" / f"{tag}_hota.json"
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
