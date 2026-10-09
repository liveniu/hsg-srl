#!/usr/bin/env python3
"""Load ground-truth and prediction bins for the TrackEval BEV-HOTA path.

The published tracking score is `evaluation.hota_trackeval`. This module
loads the 10 Hz bins, applies the reference-offset verdict, and does not
score HOTA itself.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

VEHICLE_CLASSES = ("car", "bus", "truck", "van")
# M6 labels stop at 360 s because camera C05 is black after about 362 s.
MAX_T = {"lumpi_M6": 360.0}
MAX_T.update({
    f"lumpi_M{index}": 441.5 for index in range(6)
})

BinMap = dict[int, dict[int, tuple[float, float]]]
TimeMap = dict[str, tuple[float, float]]
SplitKey = tuple[str, int, int]


def load_split_keys(path: Path, split: str = "test") -> set[SplitKey]:
    if not path.is_file():
        raise FileNotFoundError(f"missing split manifest {path}")
    keys: set[SplitKey] = set()
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("split") != split:
                continue
            cam = r.get("camera_id") or r.get("camera_name") or ""
            keys.add((cam, int(r["local_track_id"]), int(r["frame_index"])))
    if not keys:
        raise ValueError(f"split manifest has no {split!r} rows: {path}")
    return keys


def load_time_maps_from_match_report(path: Path) -> TimeMap:
    blob = json.loads(path.read_text(encoding="utf-8"))
    cameras = blob.get("cameras", blob)
    tm: TimeMap = {}
    for cam, rec in cameras.items():
        if not isinstance(rec, dict) or "error" in rec:
            continue
        tm[cam] = (
            float(rec.get("dt_applied_sec", 0.0))
            + float(rec.get("drift_const_ms", 0.0)) / 1000.0,
            float(rec.get("drift_slope_ms_per_min", 0.0)) / 60000.0,
        )
    return tm


def parse_reference_bias(text: str) -> np.ndarray:
    """Apply the common ground-truth offset only when the verdict starts with REFERENCE OFFSET.

    A file that does not start with that phrase may still contain a ``common_bias=``
    line. Scoring ignores that stored value.
    """
    if not text.startswith("REFERENCE OFFSET"):
        return np.zeros(2)
    m = re.search(r"common_bias=\(([-+0-9.]+),([-+0-9.]+)\)", text)
    if not m:
        return np.zeros(2)
    return np.array([float(m.group(1)), float(m.group(2))])


def load_gt_bins(
    origin_db: Path,
    site: str,
    bias: np.ndarray,
    t_min: float,
    *,
    gt_batch: str = "l0hn_gt_001",
    cameras: set[str] | None = None,
    max_t: float | None = None,
    vehicle_classes: tuple[str, ...] = VEHICLE_CLASSES,
) -> BinMap:
    conn = sqlite3.connect(str(origin_db))
    ph = ",".join("?" * len(vehicle_classes))
    cam_sql = ""
    cam_bind: list[Any] = []
    if cameras:
        cam_sql = " AND camera_name IN (" + ",".join("?" * len(cameras)) + ")"
        cam_bind = list(sorted(cameras))
    sql = (
        f"""SELECT global_id, timestamp_sec, outside_reference_x,
                   outside_reference_y FROM trajectory_observations
            WHERE scene_id=? AND batch_id=?
              AND class_name IN ({ph}){cam_sql}
              AND outside_reference_x IS NOT NULL"""
    )
    rows = conn.execute(sql, [site, gt_batch, *vehicle_classes, *cam_bind]).fetchall()
    if not rows:
        sql2 = (
            f"""SELECT global_id, timestamp_sec, outside_reference_x,
                       outside_reference_y FROM trajectory_observations
                WHERE scene_id=? AND batch_id=?
                  AND outside_reference_x IS NOT NULL{cam_sql}"""
        )
        rows = conn.execute(sql2, [site, gt_batch, *cam_bind]).fetchall()
    conn.close()
    if max_t is None:
        max_t = MAX_T.get(site)
    gt_by_bin: BinMap = defaultdict(dict)
    for gid, t, x, y in rows:
        if t < t_min or (max_t is not None and t > max_t):
            continue
        gt_by_bin[int(round(t * 10))][gid] = (x + bias[0], y + bias[1])
    return gt_by_bin


def load_predictions(
    db: Path,
    site: str,
    coord: str,
    tm: TimeMap,
    test_keys: set[SplitKey],
    *,
    pred_batch: str,
    cameras: set[str] | None = None,
) -> tuple[BinMap, float]:
    if coord not in {"world", "final_world"}:
        raise ValueError("coord must be 'world' or 'final_world'")
    conn = sqlite3.connect(str(db))
    acc: dict[tuple[int, int], list] = defaultdict(list)
    t_min_label = float("inf")
    for cam, tid, fr, gidp, t, x, y in conn.execute(
        f"""SELECT camera_name, local_track_id, frame_index, global_id,
                   timestamp_sec, {coord}_x, {coord}_y
            FROM trajectory_observations
            WHERE scene_id=? AND batch_id=? AND {coord}_x IS NOT NULL""",
        (site, pred_batch),
    ):
        if cameras is not None and cam not in cameras:
            continue
        if (cam, int(tid), int(fr)) not in test_keys:
            continue
        a, b = tm.get(cam, (0.0, 0.0))
        lt = t + a + b * t
        t_min_label = min(t_min_label, lt)
        acc[(int(round(lt * 10)), gidp)].append((x, y))
    conn.close()
    out: BinMap = defaultdict(dict)
    for (bn, gidp), pts in acc.items():
        arr = np.asarray(pts)
        out[bn][gidp] = (float(arr[:, 0].mean()), float(arr[:, 1].mean()))
    if t_min_label == float("inf"):
        t_min_label = 0.0
    return out, t_min_label


def self_check() -> None:
    """Tiny deterministic check that does not need licensed data."""
    assert np.all(parse_reference_bias("NO COMMON OFFSET\ncommon_bias=(-0.1,+0.2)\n") == 0)
    applied = parse_reference_bias("REFERENCE OFFSET\ncommon_bias=(-0.0385,+0.0393)\n")
    assert abs(applied[0] + 0.0385) < 1e-12 and abs(applied[1] - 0.0393) < 1e-12


if __name__ == "__main__":
    self_check()
    print("bev_hota self-check ok")
