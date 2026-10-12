#!/usr/bin/env python3
"""Geometry-only tracking replay for fixed anchors and HSG.

Detections, image tracks, test keys, ground-truth clock, reference bias,
and the full overlap graph stay frozen. Each anchor is one deterministic
fusion replay: final_world is overwritten, then fusion is rerun with the
same flags as scripts/eval_tracking.py. No residual is applied.

HSG uses the stored world coordinates, which is what the published Geo
rows used. Fixed anchors are projected with the same center-to-bottom
formula as scripts/eval_fixed_anchors.py.
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
from pathlib import Path

import numpy as np

from bootstrap import ARTIFACT_ROOT, SRC_ROOT
from db_access import connect_ro, load_adapted, load_bias, load_test_keys
from eval_tracking import paper_edges
from protocol import (
    DET_BATCH,
    GROUPS,
    GT_BATCH,
    MAX_T,
    SITES,
    paper_merge_args,
    site_dir,
)

OUT_ROOT = ARTIFACT_ROOT / "results" / "anchor_tracking"
PUBLISHED = ARTIFACT_ROOT / "results" / "recorded" / "identity_hota25.csv"

ANCHORS = (
    ("beta0", 0.0),
    ("beta050", 0.50),
    ("beta070", 0.70),
    ("beta1", 1.0),
    ("hsg", None),
)


def recording_dir(site: str) -> Path:
    path = site_dir(site)
    if not path.is_dir():
        raise SystemExit(f"missing published recording directory: {path}")
    return path


def geometry_db(site: str) -> Path:
    return meta(site, "geometry.db")


def meta(site: str, name: str) -> Path:
    path = recording_dir(site) / name
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return path


def project_beta(coords: np.ndarray, homography: np.ndarray, scale: float, beta: float) -> np.ndarray:
    """Match scripts/eval_fixed_anchors.py, including the beta=0 and beta=1 ends."""
    box = coords[:, :4]
    uv = np.column_stack((
        box[:, 0] + box[:, 2] / 2.0,
        box[:, 1] + (1.0 + beta) * box[:, 3] / 2.0,
        np.ones(len(coords)),
    ))
    metric = np.diag([1.0 / scale, 1.0 / scale, 1.0]) @ homography
    projected = uv @ metric.T
    denom = projected[:, 2:3]
    if not np.isfinite(projected).all() or np.any(np.abs(denom) < 1e-12):
        raise SystemExit("non-finite fixed-anchor projection")
    return projected[:, :2] / denom


def coordinate_updates(site: str, beta: float | None) -> list[tuple]:
    adapted, _sizes = load_adapted(site, geometry_db(site))
    conn = connect_ro(geometry_db(site))
    raw = conn.execute(
        """SELECT id, camera_name, bbox_x, bbox_y, bbox_w, bbox_h, world_x, world_y
           FROM trajectory_observations
           WHERE scene_id=? AND batch_id=? AND world_x IS NOT NULL""",
        (site, DET_BATCH),
    ).fetchall()
    conn.close()
    if beta is None:
        return [(float(wx), float(wy), int(oid)) for oid, _cam, _bx, _by, _bw, _bh, wx, wy in raw]
    by_cam: dict[str, list] = {}
    for row in raw:
        by_cam.setdefault(row[1], []).append(row)
    located = {}
    for cam, rows in by_cam.items():
        if cam not in adapted:
            raise SystemExit(f"{site} {cam}: observation has no adapted homography")
        homography, scale = adapted[cam]
        coords = np.asarray([row[2:6] for row in rows], dtype=float)
        if not np.isfinite(coords).all():
            raise SystemExit(f"{site} {cam}: non-finite box")
        world = project_beta(coords, homography, scale, beta)
        for row, xy in zip(rows, world):
            located[int(row[0])] = (float(xy[0]), float(xy[1]))
    return [(located[int(row[0])][0], located[int(row[0])][1], int(row[0])) for row in raw]


def score(site: str, db: Path) -> list[dict]:
    from evaluation.bev_hota import load_gt_bins, load_predictions, load_time_maps_from_match_report
    from evaluation.hota_trackeval import hota_trackeval

    time_map = load_time_maps_from_match_report(meta(site, "det_gt_match_report.json"))
    pred, t0 = load_predictions(
        db, site, "final_world", time_map, load_test_keys(site, meta(site, "test_keys.csv")),
        pred_batch=DET_BATCH, cameras=set(SITES[site]["cameras"]),
    )
    gt_bins = load_gt_bins(
        meta(site, "origin.db"), site, load_bias(site, meta(site, "reference_verdict.txt")),
        t0, gt_batch=GT_BATCH, cameras=set(SITES[site]["cameras"]), max_t=MAX_T.get(site),
    )
    rows = hota_trackeval(gt_bins, pred)
    bev = float(np.mean([row["hota"] for row in rows]))
    for row in rows:
        row["site"] = site
        row["bev_hota"] = bev
    return rows


def published_geo(site: str) -> dict[float, dict]:
    with PUBLISHED.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["site"] == site and row["method"] == "Geo"
        ]
    if len(rows) != 4:
        raise SystemExit(f"{site}: expected 4 published Geo rows, found {len(rows)}")
    return {float(row["gate"]): row for row in rows}


def check_hsg(site: str, rows: list[dict]) -> float:
    archived = published_geo(site)
    worst = 0.0
    for row in rows:
        ref = archived[float(row["gate"])]
        for field in ("hota", "deta", "assa", "loca", "tp", "fp", "fn"):
            worst = max(worst, abs(float(row[field]) - float(ref[field])))
    return worst


def replay(site: str, name: str, beta: float | None) -> list[dict]:
    from utils.scene_geometry_db import save_scene_overlap_pairs

    dest_dir = OUT_ROOT / "tmp" / site
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{name}.db"
    if dest.exists():
        dest.unlink()
    shutil.copy2(geometry_db(site), dest)
    conn = sqlite3.connect(str(dest))
    conn.execute("PRAGMA synchronous=OFF")
    conn.executemany(
        "UPDATE trajectory_observations SET final_world_x=?, final_world_y=? WHERE id=?",
        coordinate_updates(site, beta),
    )
    edges = [tuple(edge.split(":")) for edge in paper_edges(site)]
    save_scene_overlap_pairs(conn, site, edges, replace=True)
    conn.commit()
    conn.close()
    cmd = [
        sys.executable, "-u", str(SRC_ROOT / "pipeline" / "multi_camera_trajectory_fusion.py"),
        "--db", str(dest),
        "--scene", site,
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
    log = OUT_ROOT / "logs" / f"{site}_{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[anchor-track] {site} {name}", flush=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(cmd, cwd=str(ARTIFACT_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise SystemExit(f"remerge {site} {name} failed; see {log}")
    rows = score(site, dest)
    dest.unlink()
    for row in rows:
        row["anchor"] = name
    out = OUT_ROOT / f"{site}_{name}.json"
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if name == "hsg":
        gap = check_hsg(site, rows)
        print(f"[anchor-track] {site} hsg vs published Geo max_abs={gap:.3e}", flush=True)
        if gap > 1e-9:
            raise SystemExit(f"{site} HSG replay does not match published Geo rows ({gap})")
    return rows


ANCHOR_LABELS = (
    ("hsg", "HSG (per-detection)"),
    ("beta0", "Box-center projection"),
    ("beta050", "Fixed interior midpoint"),
    ("beta070", "STMC-derived fixed anchor"),
    ("beta1", "Bottom-center projection (IPM)"),
)


def _macro(sites: tuple[str, ...], anchor: str) -> dict[float, float]:
    values = {gate: [] for gate in (0.5, 1.0, 1.5, 2.0)}
    for site in sites:
        path = OUT_ROOT / f"{site}_{anchor}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}. Replay that recording before --summarize.")
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            values[float(row["gate"])].append(100.0 * float(row["hota"]))
    return {gate: float(np.mean(samples)) for gate, samples in values.items()}


def summarize() -> None:
    """Write the geometry-only anchor-tracking summary. Does not read a geometry database."""
    groups = {
        "Real": tuple(site for site in SITES if site.startswith("lumpi_")),
        "Synthetic": GROUPS["Synth"],
        "All": tuple(SITES),
    }
    lines = [
        "# Geometry-only TrackEval BEV-HOTA",
        "",
        "Built from results/anchor_tracking/*.json. Same detections, test keys, "
        "ground-truth clock, reference bias, full overlap graph, and final-tracking "
        "flags as the HSG rows. No residual.",
        "",
        "Recording-macro means on a 0-100 scale. The mean column is the unweighted "
        "average of 0.5, 1, 1.5, and 2 m.",
        "",
        "| Data | Coordinate rule | 0.5 m | 1 m | 1.5 m | 2 m | Mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    cached = {}
    for label, members in groups.items():
        for anchor, name in ANCHOR_LABELS:
            cached[(label, anchor)] = _macro(members, anchor)
            row = cached[(label, anchor)]
            mean = float(np.mean([row[gate] for gate in (0.5, 1.0, 1.5, 2.0)]))
            cells = " | ".join(f"{row[gate]:.2f}" for gate in (0.5, 1.0, 1.5, 2.0))
            lines.append(f"| {label} | {name} | {cells} | {mean:.2f} |")
    lines += [
        "",
        "## Per recording, mean of four gates",
        "",
        "| Recording | Box center | Fixed interior midpoint | STMC-derived fixed anchor | Bottom center | HSG |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    order = ("beta0", "beta050", "beta070", "beta1", "hsg")
    for site in SITES:
        label = site.replace("lumpi_", "").replace("synth_", "").replace("_O_day", "")
        cells = []
        for anchor in order:
            row = _macro((site,), anchor)
            cells.append(f"{float(np.mean(list(row.values()))):.2f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    dest = OUT_ROOT / "summary.md"
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[anchor-track] -> {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="", choices=["", *sorted(SITES)])
    parser.add_argument("--anchors", default="beta0,beta050,beta070,beta1,hsg")
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Write the geometry-only anchor-tracking summary from stored JSON. Does not rerun fusion.",
    )
    args = parser.parse_args()
    if args.summarize and not args.site:
        summarize()
        return
    if not args.site:
        raise SystemExit("pass --site, or --summarize to rebuild the anchor-tracking summary")
    selected = [item.strip() for item in args.anchors.split(",") if item.strip()]
    known = {name: beta for name, beta in ANCHORS}
    for name in selected:
        if name not in known:
            raise SystemExit(f"unknown anchor {name}")
    for name in selected:
        replay(args.site, name, known[name])
    if args.summarize:
        summarize()


if __name__ == "__main__":
    main()
