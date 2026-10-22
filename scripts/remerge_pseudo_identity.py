#!/usr/bin/env python3
"""Rebuild Ko-PER cross-camera pseudo-identities on the frozen overlap edge.

Pair mining groups by global id. A fresh detector stores each camera's own
tracker id, so those ids are rebuilt with the real-intersection gates before
freeze_pairs.py. This remerge reads world_x/world_y and rewrites global_id.
It does not change boxes, HSG positions, or the attached ground truth.
The edge list is the published graph in configs/overlap.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys

from bootstrap import ARTIFACT_ROOT, SRC_ROOT
from protocol import DET_BATCH, SITES, geometry_db, geometry_merge_args


def edges_of(site: str) -> list[tuple[str, str]]:
    path = ARTIFACT_ROOT / "configs" / "overlap" / f"{site}.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    edges = []
    for edge in blob.get("edges") or []:
        left, right = str(edge["a"]), str(edge["b"])
        if right < left:
            left, right = right, left
        edges.append((left, right))
    if not edges:
        raise SystemExit(f"{site} has no retained overlap edge")
    return edges


def store_edges(site: str) -> None:
    sys.path.insert(0, str(SRC_ROOT))
    from utils.scene_geometry_db import save_scene_overlap_pairs

    edges = edges_of(site)
    conn = sqlite3.connect(str(geometry_db(site)))
    stored = save_scene_overlap_pairs(conn, site, edges, replace=True)
    conn.commit()
    conn.close()
    record = {
        "site": site,
        "role": "pseudo-identity edges copied from the published overlap graph",
        "edges": [{"a": left, "b": right, "edge": f"{left}:{right}"} for left, right in edges],
    }
    dest = ARTIFACT_ROOT / "configs" / "pseudo_identity_overlap" / f"{site}.json"
    dest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[pseudo-id] {site} edges={stored}", flush=True)


def remerge(site: str) -> None:
    cmd = [
        sys.executable, "-u", str(SRC_ROOT / "pipeline" / "multi_camera_trajectory_fusion.py"),
        "--db", str(geometry_db(site)),
        "--scene", site,
        "--batch", DET_BATCH,
        "--remerge",
        "--remerge-from-db",
        "--remerge-quiet",
        "--no-view",
        "--remerge-world-coords", "p_geo",
        *geometry_merge_args(SITES[site]["kind"]),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    print(f"[pseudo-id] {site} remerge", flush=True)
    completed = subprocess.run(cmd, cwd=str(ARTIFACT_ROOT), env=env)
    if completed.returncode != 0:
        raise SystemExit(f"pseudo-identity remerge {site} failed")


def main() -> None:
    sites = sys.argv[1:] or [name for name, spec in SITES.items() if spec["kind"] == "koper"]
    for site in sites:
        if SITES[site]["kind"] != "koper":
            raise SystemExit(f"{site} is not a Ko-PER recording")
        store_edges(site)
        remerge(site)


if __name__ == "__main__":
    main()
