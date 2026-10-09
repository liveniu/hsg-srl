"""Per-point features from the valid-ground hull. No learned parameters.

Bearing is the ground angle from the bottom-center projection, relative to the
bisector of that camera's hull, divided by the hull's half-angle.
Boundary and overlap depths are signed meters divided by s0. Positive is inside.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2 as cv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.fov_overlap import (  # noqa: E402
    _load_calib_wh,
    _load_h_scale,
    valid_fov_hull,
)
from pipeline.multi_camera_trajectory_fusion import (  # noqa: E402
    adapt_homography_to_resolution,
    image_point_to_world,
)

S0_M = 60.0


def _wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _segment_ts(a: np.ndarray, b: np.ndarray, poly: np.ndarray) -> list[float]:
    ts = [0.0, 1.0]
    d = b - a
    n = len(poly)
    for i in range(n):
        c = poly[i]
        e = poly[(i + 1) % n] - c
        denom = d[0] * e[1] - d[1] * e[0]
        if abs(denom) < 1e-12:
            continue
        ac = c - a
        t = (ac[0] * e[1] - ac[1] * e[0]) / denom
        u = (ac[0] * d[1] - ac[1] * d[0]) / denom
        if -1e-8 <= t <= 1.0 + 1e-8 and -1e-8 <= u <= 1.0 + 1e-8:
            ts.append(float(np.clip(t, 0.0, 1.0)))
    ts = sorted(set(np.round(ts, 8)))
    return [float(t) for t in ts]


def _outside_pieces(a: np.ndarray, b: np.ndarray, poly: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    ts = _segment_ts(a, b, poly.astype(np.float64))
    contour = poly.astype(np.float32).reshape(-1, 1, 2)
    pieces = []
    for t0, t1 in zip(ts, ts[1:]):
        if t1 - t0 < 1e-8:
            continue
        mid = a + 0.5 * (t0 + t1) * (b - a)
        if cv.pointPolygonTest(contour, (float(mid[0]), float(mid[1])), False) < 0:
            pieces.append((a + t0 * (b - a), a + t1 * (b - a)))
    return pieces


def exterior_segments(polys: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    for i, poly in enumerate(polys):
        n = len(poly)
        for k in range(n):
            pieces = [(poly[k].astype(np.float64), poly[(k + 1) % n].astype(np.float64))]
            for j, other in enumerate(polys):
                if i == j:
                    continue
                nxt = []
                for a, b in pieces:
                    nxt.extend(_outside_pieces(a, b, other))
                pieces = nxt
                if not pieces:
                    break
            segs.extend(pieces)
    return segs


def _dist_point_seg(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    den = float(np.dot(ab, ab))
    if den < 1e-18:
        return np.linalg.norm(p - a, axis=1)
    t = np.clip(((p - a) @ ab) / den, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=1)


def signed_distance(points: np.ndarray, poly: np.ndarray | None, segs: list | None = None) -> np.ndarray:
    """Positive inside. ``segs`` is the boundary to measure when already inside a union."""
    out = np.zeros(len(points), dtype=np.float64)
    if poly is None or len(poly) < 3:
        return out
    contour = poly.astype(np.float32).reshape(-1, 1, 2)
    if segs is None:
        for i, p in enumerate(points):
            out[i] = cv.pointPolygonTest(contour, (float(p[0]), float(p[1])), True)
        return out
    inside = np.zeros(len(points), dtype=bool)
    for i, p in enumerate(points):
        inside[i] = cv.pointPolygonTest(contour, (float(p[0]), float(p[1])), False) >= 0
    if np.any(inside) and segs:
        dist = np.full(len(points), np.inf)
        for a, b in segs:
            dist = np.minimum(dist, _dist_point_seg(points, a, b))
        out[inside] = dist[inside]
    if np.any(~inside):
        for i in np.flatnonzero(~inside):
            out[i] = cv.pointPolygonTest(contour, (float(points[i, 0]), float(points[i, 1])), True)
    return out


class CameraGeom:
    def __init__(self, origin, angle0, half, hull, other_hulls, union_segs):
        self.origin = np.asarray(origin, dtype=np.float64)
        self.angle0 = float(angle0)
        self.half = float(half)
        self.hull = hull
        self.other_hulls = other_hulls
        self.union_segs = union_segs

    def extras(self, xy: np.ndarray) -> np.ndarray:
        p = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        v = p - self.origin
        ang = _wrap(np.arctan2(v[:, 1], v[:, 0]) - self.angle0)
        bearing = ang / self.half
        boundary = signed_distance(p, self.hull) / S0_M
        overlap = signed_distance_union(p, self.other_hulls, self.union_segs) / S0_M
        return np.column_stack([bearing, boundary, overlap]).astype(np.float32)


def signed_distance_union(points: np.ndarray, polys: list[np.ndarray], segs: list) -> np.ndarray:
    """Signed distance to the union. Positive inside any polygon."""
    out = np.zeros(len(points), dtype=np.float64)
    if not polys:
        return out
    inside = np.zeros(len(points), dtype=bool)
    outside = np.full(len(points), np.inf)
    for poly in polys:
        contour = poly.astype(np.float32).reshape(-1, 1, 2)
        for i, p in enumerate(points):
            d = cv.pointPolygonTest(contour, (float(p[0]), float(p[1])), True)
            if d >= 0.0:
                inside[i] = True
            else:
                outside[i] = min(outside[i], -d)
    out[~inside] = -outside[~inside]
    if segs and inside.any():
        dist = np.full(len(points), np.inf)
        for a, b in segs:
            dist = np.minimum(dist, _dist_point_seg(points, a, b))
        out[inside] = dist[inside]
    return out


def build_frames(conn, site: str, cameras: list[str], batch: str) -> dict[str, CameraGeom]:
    from pipeline.geometry_guided_residual_mlp import _infer_and_calib_wh_from_db

    infer_wh, calib_wh = _infer_and_calib_wh_from_db(conn, site, batch, cameras)
    hulls: dict[str, np.ndarray | None] = {}
    raw = {}
    for cam in cameras:
        cal = _load_h_scale(conn, site, cam)
        wh = _load_calib_wh(conn, site, batch, cam)
        if cal is None or wh is None:
            hulls[cam] = None
            continue
        H, scale = cal
        iw, cw = infer_wh.get(cam), calib_wh.get(cam)
        if iw and cw and iw != cw:
            H = adapt_homography_to_resolution(H, cw, iw)
            wh = iw
        hull, _n = valid_fov_hull(H, scale, wh)
        hulls[cam] = None if hull is None else hull.astype(np.float64)
        raw[cam] = (H, scale, wh)
    frames = {}
    for cam in cameras:
        hull = hulls.get(cam)
        if cam not in raw or hull is None:
            frames[cam] = CameraGeom(np.zeros(2), 0.0, 1.0, None, [], [])
            continue
        H, scale, wh = raw[cam]
        w, h = wh
        samples = []
        for u in np.linspace(0.0, w - 1.0, 9):
            xy = image_point_to_world(H, (float(u), float(h - 1.0)), scale)
            if np.isfinite(xy).all() and np.linalg.norm(xy) < 250.0:
                samples.append(xy)
        if samples:
            origin = np.mean(samples, axis=0)
        else:
            origin = hull.mean(axis=0)
        center = image_point_to_world(H, (0.5 * (w - 1.0), 0.5 * (h - 1.0)), scale)
        forward = np.asarray(center, dtype=np.float64) - origin
        if (not np.isfinite(forward).all()) or np.linalg.norm(forward) < 1e-3:
            forward = hull.mean(axis=0) - origin
        angle0 = float(np.arctan2(forward[1], forward[0]))
        rel = []
        for p in hull:
            v = p - origin
            rel.append(float(_wrap(np.array([np.arctan2(v[1], v[0]) - angle0]))[0]))
        half = max(max(abs(a) for a in rel), 1e-6)
        others = [hulls[c] for c in cameras if c != cam and hulls.get(c) is not None]
        segs = exterior_segments(others) if others else []
        frames[cam] = CameraGeom(origin, angle0, half, hull, others, segs)
    return frames
