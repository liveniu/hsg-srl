"""HSG-SRL overlap: valid ground-FOV MST + strong chords. No yaml/ROI/AABB.

Candidate edges: H-projected FOV convex hulls, intersection area >= A_min.
Written graph: maximum spanning tree on those areas, plus chords whose area
is >= rho * (largest tree-edge area). GT co-visibility is never a training edge.

Does not read config/*.yaml or image_roi. Does not rewrite L0 origin.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.scene_geometry_db import save_scene_overlap_pairs

A_MIN_M2 = 1000.0
RHO_CHORD = 0.5
MAX_WORLD_M = 250.0
GRID_N = 25

# Frozen fusion thresholds used with this graph (C–E remerge).
MERGE_SCORE_THRESHOLD = 0.70
MERGE_MAX_DISTANCE_M = 2.5


def _project_pts(H: np.ndarray, uv: np.ndarray, scale: float) -> np.ndarray:
    n = len(uv)
    pts = np.concatenate([uv, np.ones((n, 1))], axis=1)
    p = pts @ H.T
    w = p[:, 2]
    ok = np.abs(w) > 1e-8
    xy = np.full((n, 2), np.nan, dtype=np.float64)
    xy[ok] = p[ok, :2] / w[ok, None]
    xy /= float(scale)
    finite = np.isfinite(xy).all(axis=1)
    near = np.linalg.norm(xy, axis=1) < MAX_WORLD_M
    xy[~(ok & finite & near)] = np.nan
    return xy


def valid_fov_hull(
    H: np.ndarray, scale: float, wh: tuple[int, int], n: int = GRID_N
) -> tuple[np.ndarray | None, int]:
    w, h = wh
    xs = np.linspace(0.0, w - 1.0, n)
    ys = np.linspace(0.0, h - 1.0, n)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    bottom = np.stack(
        [np.linspace(0.0, w - 1.0, 2 * n), np.full(2 * n, h - 1.0)], axis=1
    )
    xy = _project_pts(H, np.vstack([grid, bottom]), scale)
    good = xy[np.isfinite(xy).all(axis=1)]
    if len(good) < 3:
        return None, int(len(good))
    hull = cv.convexHull(good.astype(np.float32)).reshape(-1, 2)
    return hull, int(len(good))


def hull_intersect_area(ha: np.ndarray | None, hb: np.ndarray | None) -> float:
    if ha is None or hb is None or len(ha) < 3 or len(hb) < 3:
        return 0.0
    pa = ha.astype(np.float32).reshape(-1, 1, 2)
    pb = hb.astype(np.float32).reshape(-1, 1, 2)
    _ret, inter = cv.intersectConvexConvex(pa, pb)
    if inter is None or len(inter) < 3:
        return 0.0
    return float(cv.contourArea(inter))


def _load_h_scale(
    conn: sqlite3.Connection, site: str, cam: str
) -> tuple[np.ndarray, float] | None:
    row = conn.execute(
        """
        SELECT homography, scale_px_per_meter
          FROM camera_calibrations
         WHERE camera_name=? AND scene_id=?
         ORDER BY calibrated_at DESC LIMIT 1
        """,
        (cam, site),
    ).fetchone()
    if row is None:
        return None
    H = np.frombuffer(row[0], dtype=np.float64).reshape(3, 3).copy()
    return H, float(row[1])


def _load_calib_wh(
    conn: sqlite3.Connection, site: str, batch: str, cam: str
) -> tuple[int, int] | None:
    row = conn.execute(
        """
        SELECT calib_w, calib_h, infer_w, infer_h
          FROM trajectory_camera_config
         WHERE scene_id=? AND batch_id=? AND camera_name=?
        """,
        (site, batch, cam),
    ).fetchone()
    if not row:
        return None
    cw, ch, iw, ih = row
    if cw and ch and int(cw) > 0 and int(ch) > 0:
        return int(cw), int(ch)
    if iw and ih and int(iw) > 0 and int(ih) > 0:
        return int(iw), int(ih)
    return None


def candidate_edges(
    conn: sqlite3.Connection,
    site: str,
    cameras: list[str],
    *,
    batch: str,
    a_min: float = A_MIN_M2,
) -> list[tuple[str, str, float]]:
    hulls: dict[str, np.ndarray | None] = {}
    for cam in cameras:
        cal = _load_h_scale(conn, site, cam)
        wh = _load_calib_wh(conn, site, batch, cam)
        if cal is None or wh is None:
            hulls[cam] = None
            continue
        H, scale = cal
        hull, _n = valid_fov_hull(H, scale, wh)
        hulls[cam] = hull
    edges: list[tuple[str, str, float]] = []
    for i, a in enumerate(cameras):
        for b in cameras[i + 1 :]:
            area = hull_intersect_area(hulls.get(a), hulls.get(b))
            if area >= a_min:
                aa, bb = sorted((a, b))
                edges.append((aa, bb, float(area)))
    edges.sort(key=lambda t: -t[2])
    return edges


def mst_plus_chords(
    cameras: list[str],
    edges: list[tuple[str, str, float]],
    *,
    rho: float = RHO_CHORD,
) -> dict[str, Any]:
    parent = {c: c for c in cameras}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree: list[tuple[str, str, float]] = []
    leftover: list[tuple[str, str, float]] = []
    for a, b, area in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            leftover.append((a, b, area))
            continue
        parent[ra] = rb
        tree.append((a, b, area))

    max_tree = max((e[2] for e in tree), default=0.0)
    thresh = rho * max_tree if max_tree > 0 else float("inf")
    chords = [e for e in leftover if e[2] >= thresh]
    dropped = [e for e in leftover if e[2] < thresh]
    written = {(a, b) for a, b, _ in tree} | {(a, b) for a, b, _ in chords}
    on_tree = {x for a, b, _ in tree for x in (a, b)}
    return {
        "tree": tree,
        "chords": chords,
        "dropped_chords": dropped,
        "pairs": sorted(written),
        "max_tree_area_m2": max_tree,
        "chord_threshold_m2": thresh if max_tree > 0 else None,
        "n_components": len({find(c) for c in cameras}),
        "isolated": sorted(c for c in cameras if c not in on_tree),
    }


def compute_hsg_overlap(
    conn: sqlite3.Connection,
    site: str,
    cameras: list[str],
    *,
    batch: str,
    a_min: float = A_MIN_M2,
    rho: float = RHO_CHORD,
) -> dict[str, Any]:
    cams = [str(c) for c in cameras]
    edges = candidate_edges(conn, site, cams, batch=batch, a_min=a_min)
    rec = mst_plus_chords(cams, edges, rho=rho)
    rec.update({
        "site": site,
        "cameras": cams,
        "a_min_m2": a_min,
        "rho": rho,
        "candidates": [(a, b, round(ar, 1)) for a, b, ar in edges],
        "rule": "valid_ground_FOV_MST_plus_chords_ge_rho_max_tree",
    })
    rec["tree"] = [(a, b, round(ar, 1)) for a, b, ar in rec["tree"]]
    rec["chords"] = [(a, b, round(ar, 1)) for a, b, ar in rec["chords"]]
    rec["dropped_chords"] = [
        (a, b, round(ar, 1)) for a, b, ar in rec["dropped_chords"]
    ]
    return rec


def write_hsg_overlap(
    conn: sqlite3.Connection,
    site: str,
    cameras: list[str],
    *,
    batch: str,
    a_min: float = A_MIN_M2,
    rho: float = RHO_CHORD,
) -> dict[str, Any]:
    rec = compute_hsg_overlap(
        conn, site, cameras, batch=batch, a_min=a_min, rho=rho
    )
    save_scene_overlap_pairs(conn, site, rec["pairs"], replace=True)
    return rec
