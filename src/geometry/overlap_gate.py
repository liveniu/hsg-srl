"""Overlap polygons and the published single-cue anchor terms.

Published scoring uses the frozen edges in ``configs/overlap``. Those edges
are not filtered again by intersection area. A ground point is inside a
camera's overlap union when its maximum signed distance to the incident
polygons is at least -1e-6 m. Membership is evaluated at the uncorrected
HSG point.
"""

from __future__ import annotations

import cv2
import numpy as np

POLYGON_TOL_M = -1e-6
S0_M = 60.0
ETA = 0.75


def polygons_from_hulls(
    hulls: dict[str, np.ndarray | None],
    edges: list[tuple[str, str]],
) -> dict[str, list[np.ndarray]]:
    """Intersect frozen-edge footprints. Empty when a hull is missing."""
    polygons = {camera: [] for camera in hulls}
    for left, right in edges:
        hull_left = hulls.get(left)
        hull_right = hulls.get(right)
        if hull_left is None or hull_right is None:
            continue
        if len(hull_left) < 3 or len(hull_right) < 3:
            continue
        _area, poly = cv2.intersectConvexConvex(
            np.asarray(hull_left, dtype=np.float32),
            np.asarray(hull_right, dtype=np.float32),
        )
        if poly is None or len(poly) < 3:
            continue
        shape = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        polygons.setdefault(left, []).append(shape)
        polygons.setdefault(right, []).append(shape)
    return polygons


def signed_depth(points: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray:
    """Maximum signed distance in meters. Empty polygon list is outside."""
    depth = np.full(len(points), -np.inf, dtype=np.float64)
    if len(points) == 0 or not polygons:
        return depth
    for polygon in polygons:
        contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
        for index, point in enumerate(points):
            distance = cv2.pointPolygonTest(
                contour, (float(point[0]), float(point[1])), True,
            )
            if distance > depth[index]:
                depth[index] = distance
    return depth


def inside_mask(points: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    return signed_depth(points, polygons) >= POLYGON_TOL_M


def hsg_terms(
    box_xywh: np.ndarray,
    metric_h: np.ndarray,
    wh: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Paper anchor terms for one camera frame of xywh boxes.

    ``metric_h`` is the homography after converting BEV pixels to meters.
    ``alpha`` is sensitivity only, ``phi`` is the shape prior alone, and
    ``beta`` is the published HSG blend. Positions are ground meters.
    """
    box = np.asarray(box_xywh, dtype=np.float64)
    width = np.maximum(box[:, 2], 1e-6)
    height = np.maximum(box[:, 3], 1e-6)
    center = box[:, :2] + box[:, 2:] / 2.0
    _center_xy, jac = _project(metric_h, center)
    sensitivity = np.linalg.norm(
        jac * np.asarray(wh, dtype=np.float64)[None, None], axis=(1, 2),
    )
    alpha = sensitivity / (sensitivity + S0_M)
    if np.linalg.norm(metric_h[2, :2]) >= 1e-9:
        probe = metric_h[2] @ np.array([wh[0] / 2.0, wh[1] - 1.0, 1.0])
        ground = (
            np.column_stack([center, np.ones(len(center))]) @ metric_h[2]
        ) * (1.0 if probe >= 0.0 else -1.0) > 0.0
        alpha = alpha.copy()
        alpha[~ground] = 1.0
    rho = np.clip(height / width, 0.5, 3.0)
    phi = rho ** 2 / (1.0 + rho ** 2)
    gate = 4.0 * alpha * (1.0 - alpha)
    beta = alpha + ETA * gate * (phi - alpha)
    xy, _anchor_jac = _project(metric_h, _place(center, height, beta))
    alpha_xy, _jac = _project(metric_h, _place(center, height, alpha))
    shape_xy, _jac = _project(metric_h, _place(center, height, phi))
    return {
        "alpha": alpha,
        "phi": phi,
        "beta": beta,
        "hsg": xy,
        "alpha_only": alpha_xy,
        "shape_only": shape_xy,
    }


def _place(center: np.ndarray, height: np.ndarray, beta: np.ndarray) -> np.ndarray:
    anchor = center.copy()
    anchor[:, 1] += np.asarray(beta, dtype=np.float64) * height / 2.0
    return anchor


def _project(homography: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hom = np.column_stack([points, np.ones(len(points))]) @ homography.T
    den = hom[:, 2]
    xy = hom[:, :2] / den[:, None]
    jac = (
        homography[:2, :2][None]
        - xy[:, :, None] * homography[2, :2][None, None]
    ) / den[:, None, None]
    return xy, jac
