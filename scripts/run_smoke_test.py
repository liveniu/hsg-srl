#!/usr/bin/env python3
"""Check the published localizer without licensed video.

Confirms the HSG anchor, the 4,802-parameter bound, the paper loss, the
retained-support deployment rule, and the TrackEval self-check.
"""

from __future__ import annotations

import sqlite3
import sys

import numpy as np

from bootstrap import SRC_ROOT

sys.path.insert(0, str(SRC_ROOT))

from evaluation.hota_trackeval import _self_check  # noqa: E402
from pipeline.geometry_guided_residual_mlp import (  # noqa: E402
    GeometryGuidedResidualMLP,
    TrainConfig,
    compute_geometry_base,
    xywh_to_xyxy,
)
from pipeline.multi_camera_trajectory_fusion import box_anchor  # noqa: E402
from geometry.overlap_gate import POLYGON_TOL_M, hsg_terms, inside_mask  # noqa: E402
from geometry.fov_overlap import candidate_edges  # noqa: E402
from protocol import (  # noqa: E402
    AREA_MIN_M2,
    LAMBDA_MAG,
    LAMBDA_MEAN,
    LAMBDA_PAIR,
    LAMBDA_TEMP,
    MIN_TRAIN_PAIRS,
    PAIR_DIST_M,
    R_MAX_M,
    SITES,
)
from train_recording import mvc_loss, support_cameras  # noqa: E402


def check_overlap_graph() -> None:
    """Keep an edge at or above the area cutoff and drop disjoint or small overlaps."""
    with sqlite3.connect(":memory:") as conn:
        conn.executescript("""
            CREATE TABLE camera_calibrations (
                camera_name TEXT, scene_id TEXT, homography BLOB,
                scale_px_per_meter REAL, calibrated_at TEXT
            );
            CREATE TABLE trajectory_camera_config (
                scene_id TEXT, batch_id TEXT, camera_name TEXT,
                calib_w INTEGER, calib_h INTEGER, infer_w INTEGER, infer_h INTEGER
            );
        """)
        for camera, offset in (("A", 0.0), ("B", 8.0), ("C", 25.0)):
            matrix = np.eye(3, dtype=np.float64)
            matrix[0, 2] = offset
            conn.execute("INSERT INTO camera_calibrations VALUES (?, ?, ?, ?, ?)",
                         (camera, "smoke", matrix.tobytes(), 1.0, "0"))
            conn.execute("INSERT INTO trajectory_camera_config VALUES (?, ?, ?, ?, ?, ?, ?)",
                         ("smoke", "test", camera, 10, 10, 10, 10))
        kept = candidate_edges(conn, "smoke", ["A", "B", "C"], batch="test", a_min=5.0)
        dropped = candidate_edges(conn, "smoke", ["A", "B", "C"], batch="test", a_min=10.0)
    if len(kept) != 1 or kept[0][:2] != ("A", "B") or not np.isclose(kept[0][2], 9.0):
        raise SystemExit(f"overlap graph at 5 m^2 returned {kept}")
    if dropped:
        raise SystemExit(f"overlap graph at 10 m^2 kept {dropped}")
    if AREA_MIN_M2 != {"lumpi": 1000.0, "synth": 1000.0, "koper": 500.0}:
        raise SystemExit(f"area thresholds drifted: {AREA_MIN_M2}")


def main() -> None:
    homography = np.array([
        [0.01, 0.0, -2.0],
        [0.0, -0.02, 8.0],
        [0.0001, 0.0004, 1.0],
    ], dtype=float)
    box = xywh_to_xyxy((400.0, 200.0, 80.0, 120.0))
    anchor, world, feature, risk = compute_geometry_base(
        box, homography, 4.0, (1920, 1080),
    )
    if feature.shape != (7,):
        raise SystemExit(f"feature dimension {feature.shape}")
    if not 0.0 <= risk.alpha <= 1.0:
        raise SystemExit("alpha left [0, 1]")
    beta_anchor = box_anchor(
        box, "multiplicative_gating_offsetted",
        homography=homography, frame_wh=(1920, 1080), scale=4.0, risk=risk,
    )
    if abs(beta_anchor[0] - anchor[0]) > 1e-6:
        raise SystemExit("anchor x is not the box center")
    y1, y2 = 200.0, 320.0
    center_y = 0.5 * (y1 + y2)
    if not (center_y - 1e-6) <= beta_anchor[1] <= y2 + 1e-6:
        raise SystemExit("HSG anchor left the center-to-bottom segment")
    for mode, beta in (("fixed_beta_050", 0.50), ("fixed_beta_070", 0.70)):
        point = box_anchor(box, mode, homography=homography, frame_wh=(1920, 1080), scale=4.0)
        expected = center_y + beta * (y2 - center_y)
        if abs(point[1] - expected) > 1e-6:
            raise SystemExit(f"{mode} anchor {point[1]} != {expected}")
    if not np.isfinite(world).all():
        raise SystemExit("world projection is not finite")

    library = GeometryGuidedResidualMLP(in_dim=7, hidden=64)
    if library.max_residual_m != R_MAX_M:
        raise SystemExit("library residual default is not the published 1 m bound")
    model = GeometryGuidedResidualMLP(in_dim=7, hidden=64, max_residual_m=R_MAX_M)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != 4802:
        raise SystemExit(f"parameter count {count}")
    if TrainConfig().max_pair_dist_m != PAIR_DIST_M or TrainConfig().lambda_reg != LAMBDA_MAG:
        raise SystemExit("library TrainConfig no longer matches the published loss")
    if TrainConfig().lambda_mean_delta != LAMBDA_MEAN or TrainConfig().lambda_pair_center != LAMBDA_PAIR:
        raise SystemExit("library TrainConfig no longer matches the published loss")
    if (R_MAX_M, PAIR_DIST_M, LAMBDA_MAG, LAMBDA_MEAN, LAMBDA_PAIR, LAMBDA_TEMP, MIN_TRAIN_PAIRS) != (
        1.0, 2.0, 0.10, 1.0, 0.50, 0.02, 32,
    ):
        raise SystemExit("paper constants drifted")

    import torch

    torch.manual_seed(0)
    features = torch.randn(8, 7)
    residual = model(features)
    if float(residual.abs().max()) > R_MAX_M + 1e-5:
        raise SystemExit("residual left the 1 m bound")
    loss = mvc_loss(
        residual[:4], residual[4:],
        torch.zeros(4, 2) + residual[:4],
        torch.zeros(4, 2) + residual[4:],
        torch.full((4,), 0.8),
        torch.full((4,), 0.3),
    )
    if not torch.isfinite(loss):
        raise SystemExit("paper loss is not finite")
    opened = support_cameras(["C05", "C06", "C07"], np.array([0, 1]), np.array([1, 1]))
    if opened != ["C05", "C06"]:
        raise SystemExit(f"support rule returned {opened}")
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    points = np.array([[1.0, 1.0], [3.0, 1.0], [0.0, 0.0]])
    mask = inside_mask(points, [square])
    if mask.tolist() != [True, False, True]:
        raise SystemExit(f"overlap mask {mask.tolist()}")
    if POLYGON_TOL_M != -1e-6:
        raise SystemExit("polygon tolerance drifted")
    metric = np.diag([1.0 / 4.0, 1.0 / 4.0, 1.0]) @ homography
    terms = hsg_terms(np.array([[400.0, 200.0, 80.0, 120.0]]), metric, (1920, 1080))
    if not (0.0 <= float(terms["beta"][0]) <= 1.0):
        raise SystemExit("HSG beta left [0, 1]")
    if abs(float(terms["alpha"][0]) - float(terms["beta"][0])) > 1.0:
        raise SystemExit("anchor terms are not on one detection")
    check_overlap_graph()
    if SITES["ko_per"]["cameras"] != ("SK1", "SK4"):
        raise SystemExit("Ko-PER cameras drifted")
    _self_check()
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
