#!/usr/bin/env python3
"""Detect one camera, then merge a recording and remerge it geometrically.

Detection is YOLO26l at 384x640 with BoT-SORT and camera-motion compensation
off. The association overlap is configs/pseudo_identity_overlap, the same
edges stored in the released geometry databases. Pair mining and tracking
use configs/overlap.

The geometry remerge passes the pseudo-identity gates in protocol.GEOMETRY_MERGE
on the command line. Those gates are also stored in the site YAML. They are
not the final-tracking settings. Final replay is scripts/eval_tracking.py,
which does not read the site YAML.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import yaml

from bootstrap import ARTIFACT_ROOT, SRC_ROOT
from protocol import (
    CLASSES,
    DET_BATCH,
    IMGSZ,
    MAX_FRAMES,
    MAX_T,
    OBS_TABLES,
    SITES,
    geometry_db,
    geometry_merge_args,
    site_dir,
)


def _fusion() -> Path:
    return SRC_ROOT / "pipeline" / "multi_camera_trajectory_fusion.py"


def _env() -> dict[str, str]:
    env = dict(os.environ)
    trt = os.environ.get("HSG_TRT_BIN", "").strip()
    if trt:
        env["PATH"] = trt + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def camera_yaml(site: str, camera: str) -> Path:
    source = yaml.safe_load(
        (ARTIFACT_ROOT / "configs" / "sites" / SITES[site]["yaml"]).read_text(encoding="utf-8")
    )
    cameras = [item for item in source["cameras"] if item["name"] == camera]
    if not cameras:
        raise SystemExit(f"{site} has no camera {camera}")
    payload = {key: value for key, value in source.items() if key != "cameras"}
    payload["cameras"] = cameras
    payload["output"] = str(site_dir(site) / f"{camera}_bev.png")
    payload["debug_dir"] = str(site_dir(site) / f"debug_{camera}")
    dest = site_dir(site) / "configs" / f"{camera}.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return dest


def prepare_camera_db(site: str, camera: str) -> Path:
    dest = site_dir(site) / "detect" / f"{camera}.db"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    shutil.copyfile(origin_db(site), dest)
    conn = sqlite3.connect(str(dest))
    for table in OBS_TABLES:
        try:
            conn.execute(f"DELETE FROM {table}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    return dest


def detect_camera(site: str, camera: str, model: str) -> None:
    db = prepare_camera_db(site, camera)
    cfg = camera_yaml(site, camera)
    cmd = [
        sys.executable, "-u", str(_fusion()),
        "--config", str(cfg),
        "--db", str(db),
        "--scene", site,
        "--batch", DET_BATCH,
        "--model", model,
        "--imgsz", IMGSZ,
        "--device", "0",
        "--classes", CLASSES,
        "--tracker", str(ARTIFACT_ROOT / "configs" / "botsort_static.yaml"),
        "--no-view",
    ]
    cap = MAX_FRAMES.get((site, camera))
    if cap is not None:
        cmd += ["--max-frames", str(cap)]
    print(f"[detect] {site} {camera}", flush=True)
    completed = subprocess.run(cmd, cwd=str(ARTIFACT_ROOT), env=_env())
    if completed.returncode != 0:
        raise SystemExit(f"detect {site} {camera} failed")


def _pseudo_identity_edges(site: str) -> list[tuple[str, str]]:
    path = ARTIFACT_ROOT / "configs" / "pseudo_identity_overlap" / f"{site}.json"
    if not path.exists():
        raise SystemExit(f"missing pseudo-identity overlap: {path}")
    blob = json.loads(path.read_text(encoding="utf-8"))
    edges = []
    for edge in blob["edges"]:
        left, right = str(edge["a"]), str(edge["b"])
        if right < left:
            left, right = right, left
        edges.append((left, right))
    return edges


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def merge_site(site: str) -> None:
    cameras = SITES[site]["cameras"]
    sources = [site_dir(site) / "detect" / f"{camera}.db" for camera in cameras]
    missing = [path.name for path in sources if not path.exists()]
    if missing:
        raise SystemExit(f"missing per-camera databases: {missing}")
    dest = geometry_db(site)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sources[0], dest)
    conn = sqlite3.connect(str(dest))
    for extra in sources[1:]:
        conn.execute("ATTACH DATABASE ? AS other", (str(extra),))
        for table in ("trajectory_observations", "trajectory_camera_config", "trajectory_runtime_stats"):
            cols = [name for name in _columns(conn, table) if name != "id"]
            if not cols:
                continue
            listed = ", ".join(cols)
            conn.execute(f"INSERT INTO {table} ({listed}) SELECT {listed} FROM other.{table}")
        conn.commit()
        conn.execute("DETACH DATABASE other")
    max_t = MAX_T.get(site)
    if max_t is not None:
        conn.execute(
            "DELETE FROM trajectory_observations WHERE scene_id=? AND timestamp_sec > ?",
            (site, max_t),
        )
        conn.commit()
    edges = _pseudo_identity_edges(site)
    conn.execute("DELETE FROM scene_overlap_pairs WHERE scene_id=?", (site,))
    conn.executemany(
        "INSERT INTO scene_overlap_pairs (scene_id, camera_a, camera_b) VALUES (?, ?, ?)",
        [(site, left, right) for left, right in edges],
    )
    conn.commit()
    conn.close()
    print(f"[geometry-remerge] {site} pseudo-identity edges={len(edges)}", flush=True)
    cfg = ARTIFACT_ROOT / "configs" / "sites" / SITES[site]["yaml"]
    cmd = [
        sys.executable, "-u", str(_fusion()),
        "--config", str(cfg),
        "--db", str(dest),
        "--scene", site,
        "--batch", DET_BATCH,
        "--remerge",
        *geometry_merge_args(SITES[site]["kind"]),
    ]
    print(f"[geometry-remerge] {site}", flush=True)
    completed = subprocess.run(cmd, cwd=str(ARTIFACT_ROOT), env=_env())
    if completed.returncode != 0:
        raise SystemExit(f"geometry remerge {site} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, choices=sorted(SITES))
    parser.add_argument("--camera", default="")
    parser.add_argument("--model", default=os.environ.get("HSG_YOLO_WEIGHTS", "data/models/yolo26l.pt"))
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if not args.merge_only:
        cameras = [args.camera] if args.camera else list(SITES[args.site]["cameras"])
        for camera in cameras:
            detect_camera(args.site, camera, args.model)
    merge_site(args.site)


if __name__ == "__main__":
    main()
