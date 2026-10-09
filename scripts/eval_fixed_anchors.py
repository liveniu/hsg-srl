#!/usr/bin/env python3
"""Project fixed center-to-bottom anchors on the matched test cohort.

beta = 0 is the box center, beta = 1 is the bottom center, beta = 0.50 is
the midpoint, and beta = 0.70 is the published STMC contact y_top + 0.85 h.
The stored geometry position is checked against a fresh HSG projection
before the fixed anchors are scored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from bootstrap import ARTIFACT_ROOT
from db_access import connect_ro, load_adapted, load_bias, load_test_keys, matched_rows
from protocol import DET_BATCH, GROUPS, MAX_T, SITES, geometry_db


def stats(values: np.ndarray) -> dict:
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
    }


def project_beta(site: str) -> dict[str, np.ndarray]:
    keys = load_test_keys(site)
    bias = load_bias(site)
    adapted, _sizes = load_adapted(site)
    cameras = set(SITES[site]["cameras"])
    max_t = MAX_T.get(site)
    conn = connect_ro(geometry_db(site))
    raw = conn.execute(
        """SELECT camera_name, local_track_id, frame_index, timestamp_sec,
                  bbox_x, bbox_y, bbox_w, bbox_h, world_x, world_y,
                  outside_reference_x, outside_reference_y
           FROM trajectory_observations
           WHERE scene_id=? AND batch_id=?
             AND world_x IS NOT NULL AND outside_reference_x IS NOT NULL""",
        (site, DET_BATCH),
    ).fetchall()
    conn.close()
    kept = []
    for cam, tid, frame, timestamp, *values in raw:
        if cam not in cameras or cam not in adapted or (cam, int(tid), int(frame)) not in keys:
            continue
        if max_t is not None and float(timestamp) > max_t:
            continue
        if any(value is None for value in values):
            continue
        kept.append((cam, *values))
    coords = np.asarray([row[1:] for row in kept], dtype=float)
    gt = coords[:, 6:8] + bias
    stored = np.linalg.norm(coords[:, 4:6] - gt, axis=1)
    rows, _by = matched_rows(site)
    fresh = np.linalg.norm(
        np.stack([row["xy"] for row in rows]) - np.stack([row["gt"] for row in rows]),
        axis=1,
    )
    if len(fresh) != len(stored) or not np.allclose(np.sort(fresh), np.sort(stored)):
        raise SystemExit(f"{site}: stored geometry positions do not match a fresh HSG projection")
    cam_array = np.asarray([row[0] for row in kept])
    errors = {"hsg": stored}
    for name, beta in (("center", 0.0), ("b050", 0.5), ("b070", 0.7), ("bottom", 1.0)):
        projected = np.empty((len(kept), 2))
        for cam in sorted(set(cam_array.tolist())):
            select = cam_array == cam
            box = coords[select, :4]
            uv = np.column_stack((
                box[:, 0] + box[:, 2] / 2.0,
                box[:, 1] + (1.0 + beta) * box[:, 3] / 2.0,
                np.ones(int(select.sum())),
            ))
            homography, scale = adapted[cam]
            metric = np.diag([1.0 / scale, 1.0 / scale, 1.0]) @ homography
            projected_h = uv @ metric.T
            projected[select] = projected_h[:, :2] / projected_h[:, 2:3]
        errors[name] = np.linalg.norm(projected - gt, axis=1)
    return errors


def main() -> None:
    sites = sys.argv[1:] or list(SITES)
    per_site = {}
    raw = {}
    for site in sites:
        raw[site] = project_beta(site)
        per_site[site] = {name: stats(values) for name, values in raw[site].items()}
        print(f"[anchors] {site} n={per_site[site]['hsg']['n']}", flush=True)
    groups = dict(GROUPS)
    groups["LUMPI-real"] = tuple(site for site in SITES if site.startswith("lumpi_"))
    groups["All"] = tuple(SITES)
    pooled = {}
    for group, members in groups.items():
        present = [site for site in members if site in raw]
        if len(present) != len(members):
            continue
        pooled[group] = {
            name: stats(np.concatenate([raw[site][name] for site in present]))
            for name in ("hsg", "center", "bottom", "b050", "b070")
        }
    dest = ARTIFACT_ROOT / "results" / "runs" / "fixed_anchors.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"per_recording": per_site, "pooled": pooled}, indent=2) + "\n", encoding="utf-8")
    print(f"[anchors] -> {dest}")


if __name__ == "__main__":
    main()
