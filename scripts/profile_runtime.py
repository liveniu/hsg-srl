#!/usr/bin/env python3
"""Time HSG and the residual MLP on a geometry database.

Detector timing is optional and requires a TensorRT engine built on the
same board. Fusion timing is the paper remerge, not the geometry-database
remerge. M6 and Town05 are the two published operating points.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from bootstrap import ARTIFACT_ROOT, SRC_ROOT
from db_access import connect_ro, load_adapted
from protocol import DET_BATCH, ORIN_SITES, geometry_db, site_dir

sys.path.insert(0, str(SRC_ROOT))


def percentile(values: list[float]) -> tuple[float, float, int]:
    if not values:
        return float("nan"), float("nan"), 0
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, 50)), float(np.percentile(array, 95)), int(array.size)


def time_boxes(site: str, limit_frames: int) -> dict:
    import torch
    from pipeline.geometry_guided_residual_mlp import GeometryGuidedResidualMLP, compute_geometry_base, xywh_to_xyxy

    deploy_name = f"{site}_F0.json"
    deploy = json.loads((site_dir(site) / "deploy" / deploy_name).read_text(encoding="utf-8"))
    opened = set(deploy.get("S") or [])
    ckpt = site_dir(site) / "checkpoints" / f"{site}_F0_seed0.pt"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = GeometryGuidedResidualMLP(
        in_dim=int(blob["in_dim"]), hidden=64, max_residual_m=float(blob["max_residual_m"]),
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    adapted, sizes = load_adapted(site)
    conn = connect_ro(geometry_db(site))
    rows = conn.execute(
        """SELECT camera_name, frame_index, bbox_x, bbox_y, bbox_w, bbox_h
           FROM trajectory_observations
           WHERE scene_id=? AND batch_id=? AND bbox_w IS NOT NULL
           ORDER BY camera_name, frame_index""",
        (site, DET_BATCH),
    ).fetchall()
    conn.close()
    frames: dict[tuple[str, int], list] = {}
    for cam, frame, bx, by_, bw, bh in rows:
        if cam not in adapted or None in (bx, by_, bw, bh):
            continue
        frames.setdefault((cam, int(frame)), []).append((float(bx), float(by_), float(bw), float(bh)))
    keys = list(frames)
    if limit_frames > 0:
        keys = keys[:limit_frames]
    geo_ms, mlp_ms = [], []
    for cam, _frame in keys:
        boxes = frames[(cam, _frame)]
        homography, scale = adapted[cam]
        wh = sizes[cam]
        t0 = time.perf_counter()
        feats = []
        for box in boxes:
            _a, _w, feat, _r = compute_geometry_base(xywh_to_xyxy(box), homography, scale, wh)
            feats.append(feat.astype(np.float32))
        geo_ms.append((time.perf_counter() - t0) * 1000.0)
        if cam in opened and feats:
            batch = torch.from_numpy(np.stack(feats))
            t1 = time.perf_counter()
            with torch.no_grad():
                model(batch)
            mlp_ms.append((time.perf_counter() - t1) * 1000.0)
        else:
            mlp_ms.append(0.0)
    if len(geo_ms) > 20:
        geo_ms, mlp_ms = geo_ms[10:], mlp_ms[10:]
    g50, g95, gn = percentile(geo_ms)
    m50, m95, mn = percentile(mlp_ms)
    return {"frames": gn, "hsg_ms_p50": g50, "hsg_ms_p95": g95, "mlp_ms_p50": m50, "mlp_ms_p95": m95, "mlp_frames": mn}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="", choices=["", *ORIN_SITES])
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--engine", default=os.environ.get("HSG_ORIN_ENGINE", ""))
    args = parser.parse_args()
    sites = [args.site] if args.site else list(ORIN_SITES)
    rows = []
    for site in sites:
        row = {"site": site}
        row.update(time_boxes(site, args.limit_frames))
        rows.append(row)
        print(json.dumps(row), flush=True)
    dest = ARTIFACT_ROOT / "results" / "runs" / "runtime_raw.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if args.engine:
        print(f"Detector engine {args.engine} is timed by the board script in reference/orin/run_orin.sh")


if __name__ == "__main__":
    main()
