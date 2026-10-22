"""Projected ground-field overlap used by HSG-SRL.

configs/overlap is the published graph. Pair mining and final tracking read
that frozen edge list and do not apply another area cutoff.
scripts/build_overlap.py can write a working copy for a from-video run.
That builder keeps LUMPI and Synthehicle edges of at least 1000 m^2 and
Ko-PER edges of at least 500 m^2. A_MIN_M2 below is the LUMPI/Synthehicle
default passed through unless the caller sets a_min. The graph is not a
spanning tree and does not read a site YAML or an image ROI.
configs/pseudo_identity_overlap is only the edge set already stored in the
released geometry databases.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

A_MIN_M2 = 1000.0
MAX_WORLD_M = 250.0
GRID_N = 25

# Published final-tracking gates. Replay scripts pass these on the command
# line; they do not read them from here.
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
