#!/usr/bin/env python3
"""Remerge BEVHeight positions through the paper fusion backend.

Hung test observations use the BEVHeight point in the paper frame. Every
other observation keeps the stored HSG position. Fusion uses score 0.70,
maximum distance 2.5 m, the full overlap graph, and stitching off.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np

ARTIFACT = Path(__file__).resolve().parents[2]
BH = Path(__file__).resolve().parent
sys.path.insert(0, str(ARTIFACT / "scripts"))
sys.path.insert(0, str(ARTIFACT / "src"))
sys.path.insert(0, str(BH))

from db_access import load_bias, load_test_keys  # noqa: E402
from evaluation.bev_hota import load_gt_bins, load_predictions, load_time_maps_from_match_report  # noqa: E402
from evaluation.hota_trackeval import hota_trackeval  # noqa: E402
from protocol import (  # noqa: E402
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
from score_three import SITES as BH_SITES  # noqa: E402
from score_three import attach, observations  # noqa: E402
from utils.scene_geometry_db import save_scene_overlap_pairs  # noqa: E402


def updates(site: str, hung: dict) -> list[tuple]:
    conn = sqlite3.connect(f"{geometry_db(site).resolve().as_uri()}?mode=ro", uri=True)
    rows = conn.execute(
        """SELECT id, camera_name, local_track_id, frame_index, world_x, world_y
           FROM trajectory_observations
           WHERE scene_id=? AND batch_id=? AND world_x IS NOT NULL""",
        (site, DET_BATCH),
    ).fetchall()
    conn.close()
    out = []
    replaced = 0
    for oid, cam, tid, frame, wx, wy in rows:
        hit = hung.get((cam, int(tid), int(frame)))
        if hit is None:
            out.append((float(wx), float(wy), int(oid)))
        else:
            out.append((float(hit[0]), float(hit[1]), int(oid)))
            replaced += 1
    print(f"[bh-hota] {site} replaced {replaced} / {len(out)}", flush=True)
    return out


def score(site: str, db: Path) -> list[dict]:
    clocks = load_time_maps_from_match_report(match_report(site))
    pred, t0 = load_predictions(
        db, site, "final_world", clocks, load_test_keys(site),
        pred_batch=DET_BATCH, cameras=set(SITES[site]["cameras"]),
    )
    gt = load_gt_bins(
        origin_db(site), site, load_bias(site), t0, gt_batch=GT_BATCH,
        cameras=set(SITES[site]["cameras"]), max_t=MAX_T.get(site),
    )
    rows = hota_trackeval(gt, pred)
    bev = float(np.mean([row["hota"] for row in rows]))
    for row in rows:
        row["bev_hota"] = bev
    return rows


def remerge(site: str, updates_rows: list[tuple]) -> list[dict]:
    dest = site_dir(site) / "track" / "bevheight.db"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    shutil.copy2(geometry_db(site), dest)
    conn = sqlite3.connect(str(dest))
    conn.executemany(
        "UPDATE trajectory_observations SET final_world_x=?, final_world_y=? WHERE id=?",
        updates_rows,
    )
    graph_path = overlap_path(site)
    if not graph_path.exists():
        graph_path = ARTIFACT / "configs" / "overlap" / f"{site}.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    edges = [tuple(edge["edge"].split(":")) for edge in graph["edges"]]
    save_scene_overlap_pairs(conn, site, edges, replace=True)
    conn.commit()
    conn.close()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ARTIFACT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, "-u", str(ARTIFACT / "src" / "pipeline" / "multi_camera_trajectory_fusion.py"),
        "--db", str(dest), "--scene", site, "--batch", DET_BATCH,
        "--remerge", "--remerge-from-db", "--remerge-quiet", "--no-view", "--no-stitch",
        "--remerge-world-coords", "final",
        "--merge-score-threshold", str(PAPER_MERGE_SCORE),
        "--merge-max-distance", str(PAPER_MERGE_DISTANCE_M),
    ]
    completed = subprocess.run(cmd, cwd=str(ARTIFACT), env=env)
    if completed.returncode != 0:
        raise SystemExit(f"BEVHeight remerge failed for {site}")
    return score(site, dest)


def main() -> None:
    lines = [
        "# BEVHeight HOTA after the paper fusion replay",
        "",
        "Hung test boxes use BEVHeight. Other observations keep HSG.",
        "Fusion uses score 0.70, maximum distance 2.5 m, and the full overlap graph.",
        "",
    ]
    for site in BH_SITES:
        hung_rows, _info = attach(site, observations(site))
        hung = {(row["cam"], row["tid"], row["fr"]): row["bh"] for row in hung_rows}
        rows = remerge(site, updates(site, hung))
        lines.append(f"## {site}")
        lines.append(f"- BEV-HOTA {100 * float(rows[0]['bev_hota']):.1f}")
        for row in rows:
            lines.append(
                f"- gate {row['gate']:.1f}: HOTA={100 * row['hota']:.1f} "
                f"DetA={100 * row['deta']:.1f} AssA={100 * row['assa']:.1f}"
            )
        lines.append("")
    dest = BH / "outputs" / "hota_remerge.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[bh-hota] -> {dest}")


if __name__ == "__main__":
    main()
